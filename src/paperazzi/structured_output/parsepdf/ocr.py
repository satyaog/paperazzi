import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from paperazzi import CFG
from paperazzi.platforms.openai.utils import Message
from paperazzi.platforms.utils import get_platform
from paperazzi.structured_output.utils import get_structured_output, make_disk_store
from paperazzi.utils import PaperTxt, disk_cache
from paperazzi.utils import disk_store as disk_store_decorator

PROG = f"{Path(__file__).stem.replace('_', '-')}"


def paper_make_key(paper: PaperTxt):
    return lambda a, k: paper.id


def main(argv: list = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paperoni",
        nargs="*",
        type=Path,
        default=[],
        help="Paperoni json report of papers to analyse",
    )
    options = parser.parse_args(argv)

    paperoni = sum([json.loads(_p.read_text()) for _p in options.paperoni], [])

    metadata = get_structured_output().METADATA.model_copy()
    metadata.llm_model = CFG[CFG.platform.select].model

    disk_store = make_disk_store(metadata)

    papers = [PaperTxt(p, disk_store=disk_store) for p in paperoni]

    client = get_platform().client()
    disk_store_prompt = disk_store_decorator(
        disk_cache(get_platform().ocr),
        disk_store,
        serializer=get_platform().OCRResponseSerializer,
    )

    LOG_FILE = CFG.dir.log / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        filename=LOG_FILE.with_suffix(
            f".{PROG}.{metadata.model_id}.{disk_store.prefix}.{metadata.model_id}.dbg"
        ),
        level=logging.DEBUG,
        force=True,
    )

    for paper in papers:
        messages = [
            Message(type="application/pdf", prompt=paper.pdfs[0]),
        ]

        prompt = disk_store_prompt.update(make_key=paper_make_key(paper), index=0)
        prompt(client=client, messages=messages, model=metadata.llm_model)


if __name__ == "__main__":
    main()

import argparse
import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from paperazzi import CFG, Config
from paperazzi.platforms.openai.utils import Message
from paperazzi.platforms.utils import get_platform
from paperazzi.structured_output.utils import (
    Metadata,
    get_structured_output,
    make_disk_store,
)
from paperazzi.utils import PaperTxt, disk_cache
from paperazzi.utils import disk_store as disk_store_decorator

PROG = f"{Path(__file__).stem.replace('_', '-')}"


def paper_make_key(paper: PaperTxt, parser: str):
    return lambda a, k: f"{parser}-{paper.id}"


def main(argv: list = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paperoni",
        nargs="*",
        type=Path,
        default=[],
        help="Paperoni json report of papers to analyse",
    )
    parser.add_argument(
        "--parser",
        type=str,
        help="PDF parser to use",
    )
    parser.add_argument(
        "--parser-version",
        type=str,
        help="PDF parser version to use",
    )
    parser.add_argument(
        "--parsed-format",
        type=str,
        default="md",
        choices=["md", "txt"],
        help="Conversion format to use",
    )
    options = parser.parse_args(argv)

    paperoni = sum([json.loads(_p.read_text()) for _p in options.paperoni], [])

    metadata = get_structured_output().METADATA.model_copy()
    metadata.llm_model = CFG[CFG.platform.select].model

    disk_store = make_disk_store(metadata)

    papers = [PaperTxt(p, disk_store=disk_store) for p in paperoni]

    client = get_platform().client()
    disk_store_prompt = disk_store_decorator(
        disk_cache(get_platform().prompt),
        disk_store,
        serializer=get_platform().ParsedResponseSerializer(
            get_structured_output(metadata).Analysis
        ),
    )

    document_metadata = Metadata(
        model_id="parsepdf", model_version="0.0.0", llm_model=options.parser_version
    )
    with Config.push():
        CFG.platform.select = options.parser
        document_disk_store = make_disk_store(
            document_metadata, prefix=CFG.platform.select
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
        document = None
        match options.parser:
            case "mistralai":
                with next(document_disk_store.iter_files(key=paper.id)).open("rb") as f:
                    document = get_platform(options.parser).OCRResponseSerializer.load(
                        f
                    )
                match options.parsed_format:
                    case "md":
                        document = [page.markdown for page in document.pages]

        for i, page in enumerate(document[:10]):
            if (
                re.search(r"(^|[^a-zA-Z])(abstract|summary)($|[^a-zA-Z])", page.lower())
                is not None
            ):
                break

        first_pages = document[: i + 2]

        messages = [
            Message(
                type="system", prompt=get_structured_output(metadata).SYSTEM_MESSAGE
            ),
            Message(
                type="user",
                prompt=get_structured_output(metadata).FIRST_MESSAGE,
                args=("\n---\n".join(first_pages),),
            ),
        ]

        prompt = disk_store_prompt.update(
            make_key=paper_make_key(paper, document_disk_store.prefix), index=0
        )

        prompt(
            client=client,
            messages=messages,
            model=metadata.llm_model,
            structured_model=get_structured_output(metadata).Analysis,
            structured_metadata=metadata,
        )


if __name__ == "__main__":
    main()

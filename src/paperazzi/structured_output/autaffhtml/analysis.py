import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import google
import requests.exceptions
import requests_cache
from fake_useragent import UserAgent
from requests.adapters import HTTPAdapter, Retry

from paperazzi import CFG
from paperazzi.log import logger
from paperazzi.platforms.openai.utils import Message
from paperazzi.platforms.utils import get_platform
from paperazzi.structured_output.utils import get_structured_output, make_disk_store
from paperazzi.utils import PaperTxt, disk_cache
from paperazzi.utils import disk_store as disk_store_decorator

PROG = f"{Path(__file__).stem.replace('_', '-')}"


def paper_make_key(paper: PaperTxt, doi: str):
    return lambda a, k: f"{doi}-{paper.id}"


def cached_session() -> requests_cache.CachedSession:
    return requests_cache.CachedSession(CFG.dir.cache / "requests", stale_if_error=True)


def download_html(url: str, output: Path):
    retries = Retry(total=5, backoff_factor=2)

    for additional_request_args in (
        {"headers": {"User-Agent": UserAgent(os="Linux", platforms="desktop").firefox}},
        {},
    ):
        response = cached_session().get(url, retries=retries, **additional_request_args)

        try:
            response.raise_for_status()
            break

        except requests.exceptions.HTTPError as e:
            error = e
            continue

    else:
        raise error

    output.write_text(response.text)


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
        disk_cache(get_platform().prompt),
        disk_store,
        serializer=get_platform().ParsedResponseSerializer(
            get_structured_output(metadata).Analysis
        ),
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
        pdf = paper.pdfs[0]
        for doi in filter(
            lambda x: x["type"].split(".")[0] == "doi", paper.paper_info["links"]
        ):
            link_type = doi["type"].split(".")[0]
            url = f"https://doi.org/{doi['link']}"
            url_hash = hashlib.sha256(doi["link"].encode()).hexdigest()[:8]
            html = pdf.with_stem(f"{paper.id}-{url_hash}").with_suffix(".html")

            try:
                logger.info(f"Downloading HTML for {url}")
                download_html(url, html)
            except requests.exceptions.HTTPError as e:
                logger.error(f"Failed to download HTML for {url}: {e}", exc_info=True)
                continue

            messages = [
                Message(
                    type="system", prompt=get_structured_output(metadata).SYSTEM_MESSAGE
                ),
                Message(
                    type="user",
                    prompt=get_structured_output(metadata).FIRST_MESSAGE,
                    args=(html.read_text(),),
                ),
            ]

            query_metadata = metadata.model_copy()
            query_metadata.query = {"url": url}

            prompt = disk_store_prompt.update(
                make_key=paper_make_key(paper, f"{link_type}-{url_hash}"), index=0
            )
            try:
                prompt(
                    client=client,
                    messages=messages,
                    model=query_metadata.llm_model,
                    structured_model=get_structured_output(query_metadata).Analysis,
                    structured_metadata=query_metadata,
                )
            except google.genai.errors.ClientError as e:
                logger.error(f"Failed to prompt for {url}: {e}", exc_info=True)


if __name__ == "__main__":
    main()

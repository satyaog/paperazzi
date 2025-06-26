import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import List, Optional

from paperazzi.config import CFG
from paperazzi.log import logger
from paperazzi.platforms.utils import get_platform
from paperazzi.structured_output.autaff.evaluate import (
    evaluate_response,
    format_differences,
    output_report,
)
from paperazzi.structured_output.utils import get_structured_output, make_disk_store
from paperazzi.utils import PaperTxt


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Evaluate LLM author/affiliation extraction."
    )
    parser.add_argument(
        "--paperoni",
        nargs="*",
        type=Path,
        default=[],
        help="Paperoni json reports of papers to evaluate",
    )
    parser.add_argument(
        "--format", choices=["csv", "json"], default="csv", help="Output format"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output file (default: stdout)"
    )
    parser.add_argument(
        "--show-differences",
        action="store_true",
        help="Show detailed differences when there are mismatches",
    )
    options = parser.parse_args(argv)

    # Load paperoni papers
    paperoni = sum([json.loads(_p.read_text()) for _p in options.paperoni], [])
    logger.info(f"Found {len(paperoni)} papers in paperoni reports")

    metadata = get_structured_output().METADATA
    metadata.llm_model = CFG[CFG.platform.select].model
    autaff_metadata = metadata.model_copy()
    autaff_metadata.model_id = "autaff"

    disk_store = make_disk_store(metadata)

    # Create Paper objects for all paperoni papers
    papers = [PaperTxt(_p, disk_store=disk_store) for _p in paperoni]

    # Find all validated YAMLs
    results = []
    stats = Counter()

    for paper in papers:
        val_file = CFG.dir.validated / autaff_metadata.model_id / f"{paper.id}.yaml"

        if not val_file.exists() or not paper.queries:
            continue

        analysis = None
        urls = []
        for query in paper.queries:
            with query.open("rb") as _f:
                response = (
                    get_platform()
                    .ParsedResponseSerializer(get_structured_output(metadata).Analysis)
                    .load(_f)
                )

            _analysis = None
            try:
                _analysis = _analysis or response.parsed
            except AttributeError:
                pass
            try:
                _analysis = _analysis or response.output[0].content[0].parsed
            except AttributeError:
                pass

            analysis = analysis or _analysis

            if _analysis is None or (
                not _analysis.authors_affiliations and not _analysis.affiliations
            ):
                # if there is no authors and affiliations, chances are the
                # analysed html file does not contain the info about the paper
                # but is rather a redirection page using javascript, preventing
                # us from accessing the target page
                logger.warning(
                    f"No authors and affiliations found in {response._metadata.query['url']} for {paper.id}, skipping"
                )
                continue

            urls.append(response._metadata.query["url"])

            if not analysis.affiliations:
                analysis.affiliations = _analysis.affiliations
            if not analysis.authors_affiliations:
                analysis.authors_affiliations = _analysis.authors_affiliations

            for author_affiliation in analysis.authors_affiliations:
                if not author_affiliation.affiliations:
                    _author_affiliation = next(
                        filter(
                            lambda x: x.author == author_affiliation.author,
                            _analysis.authors_affiliations,
                        ),
                        None,
                    )

                    if _author_affiliation is None:
                        continue

                    author_affiliation.affiliations = _author_affiliation.affiliations

        if not analysis.authors_affiliations or not analysis.affiliations:
            continue

        with tempfile.NamedTemporaryFile("wt") as _f:
            _f.write(json.dumps(analysis.model_dump()))
            _f.flush()
            report_entry = evaluate_response(paper, stats, val_file, Path(_f.name))

        report_entry.pdf_txt_file = urls
        report_entry.llm_file = [
            report_entry.llm_file,
            *[str(query) for query in paper.queries],
        ]

        if not options.show_differences:
            report_entry.authors_order_diff = None
            report_entry.affiliations_diff = None
            report_entry.author_affiliations_diff = None
            report_entry.typo_warnings = None

        if options.show_differences:
            format_differences(report_entry)

        results.append(report_entry.to_dict())

    # No results case
    if not results:
        logger.warning("No papers were found for evaluation.")
        return

    output_report(stats, results, options.format, options.output)


if __name__ == "__main__":
    main()

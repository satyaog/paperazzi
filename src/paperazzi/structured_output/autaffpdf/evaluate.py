import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Optional

from paperazzi.config import CFG
from paperazzi.log import logger
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

        if not val_file.exists():
            continue

        report_entry = evaluate_response(paper, stats, val_file, paper.queries[-1])

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

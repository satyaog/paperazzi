from __future__ import annotations

from pathlib import Path
from typing import Generic, TypeVar

from packaging.version import Version
from pydantic import BaseModel, Field

from paperazzi.structured_output.utils import Metadata

METADATA = Metadata(model_id=Path(__file__).parent.name, model_version=Version("2.0.0"))

SYSTEM_MESSAGE = """You are a Deep Learning expert specializing in scientific text analysis. Your task is to extract the authors and their corresponding affiliations from the provided scientific paper. Ensure that all affiliations are accurately associated with each author, especially when authors have multiple affiliations. Pay attention to symbols, superscripts, or any references that indicate institutional connections.

### Instructions:

- Extract Author Names:
  - Identify and list all author names in full (e.g., first and last name).
- Extract Affiliations:
  - For each author, extract all institutions they are affiliated with.
  - Authors may have multiple affiliations, so ensure all are captured.
- Associate Authors with Institutions:
  - Match each author with their correct institution(s).
  - Handle superscript symbols or numbers that indicate institutional affiliation, ensuring accuracy in matching authors to institutions.
- Affiliation Matching:
  - Verify that all authors are paired with the correct number of affiliations (as indicated by superscripts or numeric references in the text).
  - Ensure no author or institution is missed, even if multiple affiliations are provided.
- Check Completeness:
  - Ensure no author is omitted from the list.
  - Ensure all affiliations are listed correctly for each author.

### Key Considerations:

Some authors may have multiple affiliations. Pay special attention to superscripts or numbers that may link authors to different institutions.
Each affiliation should be captured and accurately paired with the corresponding author."""

FIRST_MESSAGE = """### The first pages of the scientific paper:

{}"""


T = TypeVar("T")


class Explained(BaseModel, Generic[T]):
    value: T
    reasoning: str = Field(
        description="A detailed explanation for the choice of the value.",
    )
    quote: str = Field(
        description="The best literal quote from the paper which supports the value",
    )


class AuthorAffiliations(BaseModel):
    author: Explained[str] = Field(
        description=("An author present in the Deep Learning scientific paper")
    )
    affiliations: list[Explained[str]] = Field(
        description=(
            "List of the author's affiliations present in the Deep Learning scientific paper"
        )
    )


class Analysis(BaseModel):
    authors_affiliations: list[AuthorAffiliations] = Field(
        description=(
            "List of all authors present in the Deep Learning scientific paper with theirs affiliations"
        )
    )
    affiliations: list[Explained[str]] = Field(
        description=(
            "List of all affiliations present in the Deep Learning scientific paper"
        )
    )

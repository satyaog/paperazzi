from __future__ import annotations

from pathlib import Path
from typing import Generic, TypeVar

from packaging.version import Version
from pydantic import BaseModel, Field

from paperazzi.structured_output.utils import Metadata

METADATA = Metadata(model_id=Path(__file__).parent.name, model_version=Version("3.0.0"))

SYSTEM_MESSAGE = """You are a Deep Learning expert specializing in scientific text analysis. Your task is to extract the authors and their corresponding affiliations from the provided scientific paper. Ensure that all affiliations are accurately associated with each author, especially when authors have multiple affiliations. Pay attention to symbols, superscripts, or any references that indicate institutional connections.

### Instructions:

- Extract Author Names:
  - Identify and list all author names in full (e.g., first and last names). Ensure you account for any middle initials or multi-part names (e.g., "John Doe Smith").
- Extract Affiliations:
  - For each author, extract all affiliated institutions.
  - If an author has multiple affiliations, capture each institution accurately.
- Associate Authors with Institutions:
  - Correctly pair each author with their corresponding affiliation(s).
  - Pay attention to superscript numbers, symbols (e.g., †), or any other references that indicate specific institutional ties.
  - Some affiliations might be explicitly stated near the author’s name without superscripts—be sure to capture those as well.
- Affiliation Accuracy:
  - Verify that all authors are paired with the correct number of affiliations (as indicated by superscripts or numeric references in the text).
  - Ensure no author or institution is missed, even if multiple affiliations are provided.
- Check Completeness:
  - Ensure no author is omitted from the list.
  - Ensure all affiliations are listed correctly for each author.

### Key Considerations:

- Multiple Affiliations: Be vigilant when an author has more than one affiliation. These should be accurately paired with the corresponding institution(s) and clearly noted.
- Superscripts or Symbols: Pay careful attention to superscripts, asterisks, or other symbols that indicate affiliation links. Ensure these are handled correctly when matching authors with institutions.
- Affiliation Clarity: Ensure all affiliations are clearly listed and paired with the corresponding author, even if the affiliation is explicitly listed without a superscript."""

FIRST_MESSAGE = """### The first pages of the scientific paper:

{}"""


T = TypeVar("T")


class Explained(BaseModel, Generic[T]):
    value: T
    reasoning: str = Field(
        description="A detailed explanation for the choice of the value",
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

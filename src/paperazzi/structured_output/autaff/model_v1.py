from __future__ import annotations

from pathlib import Path
from typing import Generic, TypeVar

from packaging.version import Version
from pydantic import BaseModel, Field

from paperazzi.structured_output.utils import Metadata

METADATA = Metadata(model_id=Path(__file__).parent.name, model_version=Version("1.0.0"))


SYSTEM_MESSAGE = """You are an expert in Deep Learning Research. Your task is to identify all the authors of a scientific paper along with their respective affiliations, ensuring that each author is correctly associated with the relevant institution(s). An author can have multiple affiliations.

### Instructions:
- Identify all authors listed in the paper.
- Identify the corresponding affiliations for each author.
- Correctly associate the affiliations with each author, ensuring accuracy."""

FIRST_MESSAGE = """### The first page of the scientific paper:
{}"""


T = TypeVar("T")


class Explained(BaseModel, Generic[T]):
    value: T
    justification: str = Field(
        description="A detailed explanation for the choice of the value.",
    )
    quote: str = Field(
        description="The best literal quote from the paper which supports the value",
    )


class AuthorAffiliations(BaseModel):
    author: Explained[str] = Field(
        description=("An author found in the Deep Learning scientific paper")
    )
    affiliations: list[Explained[str]] = Field(
        description=(
            "List of the author affiliations found in the Deep Learning scientific paper"
        )
    )


class Analysis(BaseModel):
    authors_affiliations: list[AuthorAffiliations]

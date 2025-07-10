from __future__ import annotations

from pathlib import Path

from packaging.version import Version

from paperazzi.structured_output.utils import Metadata

METADATA = Metadata(model_id=Path(__file__).parent.name, model_version=Version("0.0.0"))

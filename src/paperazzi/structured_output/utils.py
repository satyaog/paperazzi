import functools
import importlib
from pathlib import Path
from typing import Callable, Optional

from packaging.version import Version
from pydantic import BaseModel, ConfigDict, field_serializer, model_validator

from paperazzi import CFG
from paperazzi.platforms.utils import get_platform
from paperazzi.utils import DiskStore, _make_key


class Metadata(BaseModel):
    model_id: str
    model_version: Version = Version("0.0.0")
    llm_model: Optional[str] = None

    @model_validator(mode="before")
    def parse_model_version(cls, values):
        # Convert the model_version from string to Version if it is a string
        if "model_version" in values and isinstance(values["model_version"], str):
            values["model_version"] = Version(values["model_version"])
        return values

    @field_serializer("model_version")
    def serialize_model_version(self, model_version: Version):
        return str(model_version)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )


def make_disk_store(
    metadata: Metadata,
    make_key: Callable = _make_key,
    prefix: str = None,
    index: int = None,
) -> DiskStore:
    cache_dir = CFG.dir.analyses / metadata.model_id / CFG.platform.select

    # TODO: fix the prefix variable name. The prefix arg is not really a prefix
    # if used only through the get_platform utils. platform seems to be a better
    # choice.
    prefix = get_platform(prefix).CODE.lower()
    if metadata.llm_model:
        # Remove underscores from the LLM model name to simplify parsing of the
        # filename (prefix_id_index)
        prefix = "-".join(
            [
                prefix,
                metadata.llm_model.replace("models", "")
                .replace("model", "")
                .replace("_", "")
                .replace("/", ""),
            ]
        )

    return DiskStore(
        cache_dir=cache_dir,
        make_key=make_key,
        prefix=prefix,
        version=metadata.model_version,
        index=index,
    )


@functools.cache
def _structured():
    return {
        f"{model_id.parent.name}.{model_version.stem}": importlib.import_module(
            f"paperazzi.structured_output.{model_id.parent.name}.{model_version.stem}"
        )
        for model_id in sorted(Path(__file__).parent.glob("*/__init__.py"))
        for model_version in [model_id.parent / "model.py"]
        + sorted(
            model_id.parent.glob("model_v*.py"),
            key=lambda x: int(x.stem.split("_")[-1][1:]),
        )
    }


def get_structured_output(metadata: Metadata = None):
    if metadata is None:
        metadata = Metadata(
            model_id=CFG.structured.model,
            model_version=CFG.structured.version or "0.0.0",
        )

    model_version = (
        f"model_v{metadata.model_version.major}"
        if metadata.model_version.major > 0
        else "model"
    )

    if metadata.model_id == "autaff" and model_version == "model":
        # Only used for typehints
        import paperazzi.structured_output.autaff.model as _model

        return _model

    return _structured()[f"{metadata.model_id}.{model_version}"]

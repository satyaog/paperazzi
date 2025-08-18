import functools
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from paperazzi.config import CFG
from paperazzi.log import logger


@dataclass
class Message:
    type: Literal["system", "user", "assistant", "application/pdf"]
    prompt: str
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str | Path:
        if self.type == "application/pdf":
            return Path(self.prompt)
        if self.args or self.kwargs:
            return self.prompt.format(*self.args, **self.kwargs)
        return self.prompt

    def format_message(self) -> dict[str, str | Path]:
        return {
            "role": self.type,
            "content": self.content,
        }


@functools.cache
def _platforms():
    platforms = {}
    for platform in sorted(Path(__file__).parent.glob("*/__init__.py")):
        try:
            platforms[platform.parent.name] = importlib.import_module(
                f"paperazzi.platforms.{platform.parent.name}"
            )
        except ImportError:
            logger.warning(
                f"{platform.parent.name} optional dependency is not installed.",
                exc_info=True,
            )
            continue
    return platforms


def iter_platforms():
    return _platforms().keys()


def get_platform(platform: str = None):
    platform = platform or (
        "instructor" if CFG.platform.instructor else CFG.platform.select
    )

    if platform == "openai":
        # Only used for typehints
        import paperazzi.platforms.openai as _openai

        return _openai.utils

    return _platforms()[platform].utils

import functools
import importlib
from pathlib import Path

from paperazzi.config import CFG


@functools.cache
def _platform():
    return {
        platform.parent.name: importlib.import_module(
            f"paperazzi.platform.{platform.parent.name}"
        )
        for platform in sorted(Path(__file__).parent.glob("*/__init__.py"))
    }


def get_platform(platform: str = None):
    platform = platform or (
        "instructor" if CFG.platform.instructor else CFG.platform.select
    )

    if platform == "openai":
        # Only used for typehints
        import paperazzi.platform.openai as _openai

        return _openai.utils

    return _platform()[platform].utils

import functools
import importlib
from pathlib import Path

import paperazzi
from paperazzi.config import CFG
from paperazzi.log import logger


@functools.cache
def _platforms():
    platforms = {}
    for platform in sorted(Path(__file__).parent.glob("*/__init__.py")):
        try:
            platforms[platform.parent.name] = importlib.import_module(
                f"paperazzi.platform.{platform.parent.name}"
            )
        except ImportError:
            logger.warning(
                f"{platform.parent.name} optional dependencies are not installed.",
                exc_info=True,
            )
            continue
    return platforms


def get_platform(platform: str = None):
    platform = platform or (
        "instructor" if CFG.platform.instructor else CFG.platform.select
    )

    if platform == "openai":
        # Only used for typehints
        import paperazzi.platform.openai as _openai

        return _openai.utils

    return _platforms()[platform].utils

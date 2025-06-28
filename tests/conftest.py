import os

import pytest

from . import CONFIG_FILE

os.environ["PAPERAZZI_CFG"] = str(CONFIG_FILE)

from paperazzi.config import Config

with Config.push() as config:
    # Correctly load files from project's data directory
    # TODO: use a path from the config file instead of assuming the file is
    # located in the config's data dir
    config.dir.data = config.dir.root / "../data"


@pytest.fixture(scope="function", autouse=True)
def cfg():
    with Config.push() as config:
        yield config

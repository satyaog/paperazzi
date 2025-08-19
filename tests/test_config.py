import logging
import os
from pathlib import Path

import pytest

from paperazzi.config import CFG, Config
from paperazzi.log import logger as main_logger

from . import CONFIG_FILE


def test_global_config():
    """Test that the global config instance is a singleton"""
    global_config = Config.get_global_config()
    assert global_config == Config.get_global_config()

    config = Config(str(CONFIG_FILE))
    config.testsection.testoption = "new test value"
    assert config != global_config

    Config.apply_global_config(config)
    assert config == Config.get_global_config()
    assert global_config != Config.get_global_config()


def test_stack_config():
    """Test that the stack config method correctly stack the config"""
    global_config = Config.get_global_config()
    config = Config(str(CONFIG_FILE))
    config.testsection.testoption = "new test value"

    assert global_config == CFG

    with Config.push(config) as cfg:
        assert cfg is config
        assert config == CFG
        assert global_config != Config.get_global_config()
        assert CFG == Config.get_global_config()

    with Config.push() as cfg:
        cfg.testsection.testoption = "new test value"
        cfg == Config.get_global_config()

        cfg.testsection.testoption = "renew test value"
        cfg == Config.get_global_config()

    assert CFG == global_config


def test_config_setattr():
    """Test setting an existing config attribute"""
    config = Config(str(CONFIG_FILE))
    assert config.testsection.testoption != "new test value"
    config.testsection.testoption = "new test value"
    assert config.testsection.testoption == "new test value"

    with pytest.raises(KeyError):
        config.nosection = {}

    with pytest.raises(KeyError):
        config.testsection.noopt = ""


def test_config_env_vars(monkeypatch, caplog):
    """Test that environment variable starting with 'PAPERAZZI_' are correctly
    added to the config. Make sure environment variables do not add entry to
    config"""

    monkeypatch.setenv("PAPERAZZI_DIR_TESTPATH", "path/to/file")
    monkeypatch.setenv("PAPERAZZI_TESTSECTION_TESTOPTION", "updated test value")

    monkeypatch.setenv("PAPERAZZI_NOSECTION_OPT", "")
    monkeypatch.setenv("PAPERAZZI_SECTION_NOOPT", "")

    config = Config(str(CONFIG_FILE))

    assert config.testsection.testoption == "updated test value"
    assert config.dir.testpath == CONFIG_FILE.parent / "path/to/file"
    assert "nosection" not in config
    assert "noopt" not in config.testsection

    assert any(
        "PAPERAZZI_NOSECTION_OPT" in rec.message
        for rec in filter(lambda r: r.levelname == "WARNING", caplog.records)
    )
    assert any(
        "PAPERAZZI_SECTION_NOOPT" in rec.message
        for rec in filter(lambda r: r.levelname == "WARNING", caplog.records)
    )


def test_config_empty_env_vars_dont_override_existing(monkeypatch):
    """Test that empty values in config.ini file do not replace existing environment variables"""

    # Set up existing environment variables
    existing_value = "not empty anymore"
    monkeypatch.setenv("EMPTY_VAR_ENV", existing_value)
    monkeypatch.setenv("NOT_EMPTY_VAR_ENV", "")
    monkeypatch.delenv("OPENAI_API_KEY")

    # Create a config with empty values for these env vars
    config = Config(str(CONFIG_FILE))

    assert config.env.empty_var_env == ""

    # Apply the global config which should set env vars from config
    Config.apply_global_config(config)

    # Verify that existing empty environment variables are preserved
    assert os.environ["EMPTY_VAR_ENV"] == existing_value

    # Test that the not empty environment variable is overridden by the config
    assert os.environ["NOT_EMPTY_VAR_ENV"] == config.env.not_empty_var_env

    # Test that missing environment variables from existing environment are
    # added empty
    assert os.environ["OPENAI_API_KEY"] == ""


def test_config_dir_section_resolve():
    """Test that the 'dir' section of the config get parsed as Path objects then
    resolved"""

    config = Config(str(CONFIG_FILE))

    for k in config.dir:
        assert isinstance(config.dir[k], Path)

    assert str(config.dir.testpath.relative_to(CONFIG_FILE.parent)) == "."


def test_config_logging_level(monkeypatch):
    """Test that the logging level is correctly set from the config file"""

    monkeypatch.setenv("PAPERAZZI_LOGGING_LEVEL", "CRITICAL")
    config = Config(str(CONFIG_FILE))
    assert config.logging.level == "CRITICAL"

    monkeypatch.setenv("PAPERAZZI_LOGGING_LEVEL", "DEBUG")
    config = Config(str(CONFIG_FILE))
    assert config.logging.level == "DEBUG"

    main_logger.level = logging.NOTSET
    Config.apply_global_config(config)
    assert main_logger.level == logging.DEBUG

    with pytest.raises(ValueError):
        monkeypatch.setenv("PAPERAZZI_LOGGING_LEVEL", "NOT A LEVEL")
        Config.apply_global_config(Config(str(CONFIG_FILE)))
        Config.apply_global_config(Config(str(CONFIG_FILE)))

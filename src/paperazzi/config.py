import configparser
import copy
import logging
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Union

from paperazzi.log import logger as main_logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

_PREFIX = Path(__file__).parent.name.upper()
_CFG_VARENV = f"{_PREFIX}_CFG"
CONFIG_FILE = os.environ.get(
    _CFG_VARENV, Path(__file__).parent.parent.parent / "config.ini"
)


def config_to_dict(config):
    """Converts a configparser.ConfigParser object to a dictionary."""
    config_dict = {}
    for section in config.sections():
        # Create a dictionary for each section
        section_dict = {}
        for option in config.options(section):
            section_dict[option] = config.get(section, option)
        config_dict[section] = section_dict
    return config_dict


class Config:
    _main_instance: "Config" = None
    _instance = threading.local()

    def __init__(self, config_file: str = CONFIG_FILE, config: dict = None) -> None:
        """Create a Config object from a config file or a dictionary."""
        if config:
            self._config = config

        else:
            # Parse and apply environment variables to config prior loading it
            with tempfile.NamedTemporaryFile("wt") as tmp_file:
                _config = configparser.ConfigParser()
                assert _config.read(
                    config_file
                ), f"Could not read config file [{config_file}]"
                _config = self._parse_env_vars(_config)
                _config.write(tmp_file.file)
                tmp_file.flush()

                _config = configparser.ConfigParser(
                    interpolation=configparser.ExtendedInterpolation()
                )
                _config.read(tmp_file.name)

            self._config = config_to_dict(_config)

            self._resolve(config_file)

    def __deepcopy__(self, memo):
        _config = Config(config=copy.deepcopy(self._config, memo))
        memo[id(self)] = _config
        return _config

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Config):
            return value._config == self._config

        elif isinstance(value, dict):
            return value == self._config

        else:
            raise TypeError(f"Cannot compare {type(self)} with {type(value)}")

    def __getattribute__(self, name: str) -> Union["Config", Path]:
        try:
            return object.__getattribute__(self, name)

        except AttributeError:
            return self[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_config":
            return object.__setattr__(self, name, value)

        if isinstance(self._config[name], dict):
            raise ValueError("Can only set values for nested Config objects")

        self._config[name] = value

    def __getitem__(self, key: str) -> Union["Config", Path]:
        value = self._config[key]
        if isinstance(value, dict):
            return Config(config=value)
        else:
            match value:
                case "true":
                    return True
                case "false":
                    return False
                case _:
                    return value

    def __iter__(self):
        return iter(self._config)

    def _resolve(self, config_file: str):
        config_parent = Path(config_file).resolve().parent

        for k, v in self._config["dir"].items():
            v = Path(v)
            if not v.is_absolute():
                v = config_parent / v
            self._config["dir"][k] = v

    @classmethod
    def _parse_env_vars(
        cls, config: configparser.ConfigParser
    ) -> configparser.ConfigParser:
        for envvar, value in os.environ.items():
            if not envvar.startswith(f"{_PREFIX}_") or envvar == _CFG_VARENV:
                continue

            conf_key = envvar.lower().split("_")[1:]

            try:
                section = config[conf_key.pop(0)]

                option = "_".join(conf_key)
                # Do not create options
                if option not in section:
                    raise KeyError(option)

                section[option] = value
            except KeyError:
                logger.warning(
                    f"Could not find env var {envvar} in config", exc_info=True
                )

        return config

    @staticmethod
    def get_global_config() -> "Config":
        """Returns the global instance of Config."""
        try:
            Config._instance.value
        except AttributeError:
            Config.apply_global_config(
                copy.deepcopy(Config._main_instance)
            )  # Set env vars and logging level

        return copy.deepcopy(Config._instance.value)

    @staticmethod
    def apply_global_config(config: "Config") -> None:
        """Apply the global instance of Config."""
        Config._instance.value = config

        try:
            main_logger.setLevel(config._config["logging"]["level"])
        except KeyError:
            pass

        for varenv, val in config._config["env"].items():
            os.environ[varenv.upper()] = val

    @contextmanager
    @staticmethod
    def push(config: "Config" = None) -> Generator["Config", None, None]:
        """Context manager to temporarily change the global config."""
        _config: Config = Config._instance.value

        try:
            if config is None:
                config = Config.get_global_config()

            Config.apply_global_config(config)

            yield config

        finally:
            Config._instance.value = _config


class GlobalConfigProxy(Config):
    def __init__(self) -> None:
        pass

    @property
    def _config(self) -> dict:
        self.get_global_config()
        return Config._instance.value._config


if Config._main_instance is None:
    Config._main_instance = Config()

CFG = GlobalConfigProxy()

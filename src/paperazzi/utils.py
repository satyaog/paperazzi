import functools
import json
import pickle
import tempfile
from dataclasses import dataclass
from multiprocessing import Lock
from pathlib import Path
from typing import Any, BinaryIO, Callable

from packaging.version import Version

import paperazzi


@dataclass
class CacheSerializer:
    dump: Callable
    load: Callable


def _make_key(args, kwargs):
    return hash(functools._make_key(args, kwargs, typed=False))


@dataclass
class DiskStore:
    cache_dir: Path
    make_key: Callable
    prefix: str = None
    version: Version = None
    index: int = None

    def __post_init__(self):
        if self.cache_dir is not None:
            self.cache_dir = self.cache_dir.resolve()

    def get_file(
        self, args: tuple = None, kwargs: dict = None, *, key: str = None
    ) -> Path:
        index = self.index
        key = key or self.make_key(args, kwargs)
        cache_dir = self.cache_dir

        if self.version is not None:
            cache_dir = cache_dir / str(self.version)
        if index is not None:
            index = f"{index:04d}"

        filename = "_".join(
            [part for part in (self.prefix, str(key), index) if part is not None]
        )

        cache_file = cache_dir / filename

        return cache_file

    def iter_files(self, args: tuple = None, kwargs: dict = None, *, key: str = None):
        no_index_kwargs = vars(self).copy()
        no_index_kwargs.pop("index")

        cache_file = DiskStore(**no_index_kwargs).get_file(args, kwargs, key=key)

        if cache_file.exists():
            yield cache_file

        for file in cache_file.parent.glob(f"{cache_file.name}_*"):
            yield file

    def move_to(
        self,
        store: "DiskStore",
        args: tuple = None,
        kwargs: dict = None,
        *,
        key: str = None,
        force: bool = False,
    ) -> Path:
        _here = self.get_file(args, kwargs, key=key)
        _there = store.get_file(args, kwargs, key=key)

        if _there.exists() and not force:
            raise FileExistsError(f"Destination file {_there} already exists")

        _there.parent.mkdir(parents=True, exist_ok=True)
        return _here.rename(_there)


class DiskCachedFunc:
    @dataclass
    class _Info:
        func: Callable
        cache_dir: Path = (
            Path(tempfile.gettempdir()) / paperazzi.__package__
        ).resolve()
        serializer: CacheSerializer = pickle
        make_key: Callable = _make_key

    _DEFAULTS = _Info(func=lambda: None)

    def __init__(self, func, cache_dir=None, serializer=None, make_key=None):
        if isinstance(func, DiskCachedFunc):
            func = func.info.func

        if cache_dir is None:
            cache_dir = self._DEFAULTS.cache_dir
        if serializer is None:
            serializer = self._DEFAULTS.serializer
        if make_key is None:
            make_key = self._DEFAULTS.make_key
        cache_dir = cache_dir.resolve()

        self._func = func
        self._cache_dir = cache_dir
        self._serializer = serializer
        self._make_key = make_key
        self._lock = Lock()

    @property
    def info(self):
        return self._Info(
            func=self._func,
            cache_dir=self._cache_dir,
            serializer=self._serializer,
            make_key=self._make_key,
        )

    def exists(self, *args, **kwargs):
        key = self._make_key(args, kwargs)
        cache_file = self._cache_dir / f"{self._func.__name__}_{key}"
        return cache_file.exists(), cache_file

    def __call__(self, *args, **kwargs):
        cache_exists, cache_file = self.exists(*args, **kwargs)

        if cache_exists:
            with self._lock, cache_file.open("rb") as _file:
                return self._serializer.load(_file)

        result = self._func(*args, **kwargs)

        with self._lock:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with cache_file.open("wb") as _file:
                self._serializer.dump(result, _file)
        assert self._serializer.load(cache_file.open("rb"))

        return result

    def update(
        self,
        cache_dir: Path = None,
        serializer=None,
        make_key: Callable = None,
    ):
        cache_kwargs = vars(self.info).copy()
        func = cache_kwargs.pop("func")

        if cache_dir is not None:
            cache_kwargs["cache_dir"] = cache_dir
        if serializer is not None:
            cache_kwargs["serializer"] = serializer
        if make_key is not None:
            cache_kwargs["make_key"] = make_key

        return DiskCachedFunc(func, **cache_kwargs)


class DiskStoreFunc(DiskCachedFunc):
    @dataclass
    class _Info(DiskCachedFunc._Info):
        store: DiskStore = None

    def __init__(self, func: DiskCachedFunc, store: DiskStore):
        store = DiskStore(**vars(store))

        if store.cache_dir is None:
            store.cache_dir = func.info.cache_dir
        if store.prefix is None:
            store.prefix = func.info.func.__name__
        if store.make_key is None:
            store.make_key = func.info.make_key

        super().__init__(func.info.func, None, func.info.serializer, None)

        self._store = store

    @property
    def info(self):
        return self._Info(
            func=self._func,
            cache_dir=self._store.cache_dir,
            serializer=self._serializer,
            make_key=self._store.make_key,
            store=self._store,
        )

    def exists(self, *args, **kwargs):
        cache_file = self._store.get_file(args, kwargs)
        return cache_file.exists(), cache_file

    def update(
        self,
        store: DiskStore = None,
        *,
        cache_dir: Path = None,
        make_key: Callable = None,
        prefix: str = None,
        version: Version = None,
        index: int = None,
    ):
        if store is not None:
            store_kwargs = {k: v for k, v in vars(store).items() if v is not None}
        else:
            store_kwargs = {}

        if cache_dir is not None:
            store_kwargs["cache_dir"] = cache_dir
        if make_key is not None:
            store_kwargs["make_key"] = make_key
        if prefix is not None:
            store_kwargs["prefix"] = prefix
        if version is not None:
            store_kwargs["version"] = version
        if index is not None:
            store_kwargs["index"] = index

        store_kwargs = {**vars(self.info.store), **store_kwargs}

        return DiskStoreFunc(self, DiskStore(**store_kwargs))


def disk_cache(
    func=None,
    *,
    cache_dir=DiskCachedFunc._DEFAULTS.cache_dir,
    serializer=DiskCachedFunc._DEFAULTS.serializer,
    make_key=DiskCachedFunc._DEFAULTS.make_key,
):
    """Cache the result of a function to a file on disk.

    Args:
        func: Function to cache
        cache_dir: Directory to cache the result
        serializer: Serializer to use. Must have a load and dump method
        make_key: Function to generate cache keys
    """

    def decorator(f) -> DiskCachedFunc:
        return DiskCachedFunc(f, cache_dir, serializer, make_key)

    if func is None:
        # Called as @disk_cache(params)
        return decorator
    else:
        # Called as @disk_cache
        return decorator(func)


def disk_store(
    func: DiskCachedFunc = None,
    store: DiskStore = None,
    *,
    cache_dir=None,
    serializer=None,
    make_key=None,
):
    """Wrap a function to use a disk store.

    Args:
        func: Function to wrap
        store: DiskStore to use
        cache_dir: Directory to cache the result. Overrides store.cache_dir
        serializer: Serializer to use. Must have a load and dump method.
        make_key: Function to generate cache keys. Overrides store.make_key
    """

    if store is None:
        store = DiskStore(cache_dir=None, make_key=None)

    if cache_dir is not None:
        store.cache_dir = cache_dir
    if make_key is not None:
        store.make_key = make_key

    if cache_dir is None:
        cache_dir = DiskCachedFunc._DEFAULTS.cache_dir
    if serializer is None:
        serializer = DiskCachedFunc._DEFAULTS.serializer
    if make_key is None:
        make_key = DiskCachedFunc._DEFAULTS.make_key

    def decorator(f) -> DiskStoreFunc:
        disk_cached_func = DiskCachedFunc(f, cache_dir, serializer, make_key)
        return DiskStoreFunc(disk_cached_func, store)

    if func is None:
        # Called as @disk_store(params)
        return decorator
    else:
        # Called as @disk_store
        return decorator(func)

import functools
import hashlib
import pickle
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from multiprocessing import Lock
from pathlib import Path
from typing import Callable, Generator

from packaging.version import Version

import paperazzi
from paperazzi import CFG
from paperazzi.log import logger


@dataclass
class CacheSerializer:
    dump: Callable
    load: Callable


def _make_key(args, kwargs):
    key = pickle.dumps((args, kwargs))
    return hashlib.sha256(key).hexdigest()


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

    def iter_files(
        self, args: tuple = None, kwargs: dict = None, *, key: str = None
    ) -> Generator[Path, None, None]:
        no_index_kwargs = vars(self).copy()
        no_index_kwargs.pop("index")

        cache_file = DiskStore(**no_index_kwargs).get_file(args, kwargs, key=key)

        if cache_file.exists():
            yield cache_file

        yield from sorted(
            cache_file.parent.glob(f"{cache_file.name}_*"),
            # extract the index from the filename (*_{index}.ext)
            key=lambda x: int(Path(x.name.split("_")[-1]).stem),
        )

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
            with self._lock, cache_file.open("rb") as f:
                return self._serializer.load(f)

        result = self._func(*args, **kwargs)

        with self._lock:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with cache_file.open("wb") as f:
                self._serializer.dump(result, f)
            with cache_file.open("rb") as f:
                assert self._serializer.load(f)

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
        # Disable DiskCachedFunc cache_dir and make_key to use the store ones
        self._cache_dir = None
        self._make_key = None

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
    ) -> "DiskStoreFunc":
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

    Returns:
        DiskCachedFunc: A wrapped function that caches its results

    Example:
        @disk_cache
        def expensive_function(x):
            return x ** 2

        # Or with parameters
        @disk_cache(cache_dir=Path('/tmp/cache'))
        def expensive_function(x):
            return x ** 2
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

    def decorator(f: DiskCachedFunc) -> DiskStoreFunc:
        disk_cached_func = DiskCachedFunc(
            f,
            f.info.cache_dir if cache_dir is None else cache_dir,
            f.info.serializer if serializer is None else serializer,
            f.info.make_key if make_key is None else make_key,
        )
        return DiskStoreFunc(disk_cached_func, store)

    if func is None:
        # Called as @disk_store(params)
        return decorator
    else:
        # Called as @disk_store
        return decorator(func)


@functools.lru_cache(maxsize=64)
def _glob(path: str, *args, **kwargs):
    return sorted(Path(path).glob(*args, **kwargs))


class PaperBase:
    # Original form of the converted pdf to txt. eg data/cache/*/ARXIV_ID.txt
    LINK_ID_TEMPLATE = "*/{link_id}.txt"
    # Extended form of the converted pdf to txt. eg data/cache/*/PAPER_ID.txt
    PAPER_ID_TEMPLATE = LINK_ID_TEMPLATE.format(link_id="{paper_id}")
    # The the up-to-date form of the converted pdf (by paperoni)
    # eg data/cache/fulltext/PAPER_ID/fulltext.txt
    PAPER_ID_FULLTEXT_TEMPLATE = "fulltext/{paper_id}/fulltext.txt"

    def __init__(self, paper_info: dict, disk_store: DiskStore) -> None:
        self.paper_info = paper_info

        self._selected_id = None
        self._queries = []
        self._pdfs = []
        self._pdf_txts = []
        link_ids = [self._paper_id]
        pdf_txts = []

        for l in paper_info["links"]:
            link_id = l.get("link", None)
            if link_id and link_id not in link_ids:
                link_ids.append(link_id)

                pdf_txts += _glob(
                    str(CFG.dir.cache), self.LINK_ID_TEMPLATE.format(link_id=link_id)
                )

        # Find existing analyses and infer the paper id from them
        _ids = []
        for link_id in link_ids:
            analyses = list(disk_store.iter_files(key=link_id)) + list(
                disk_store.iter_files(key=f"*-{link_id}")
            )
            self._queries.extend(analyses)
            if analyses:
                _ids.append(link_id)

        if _ids:
            if len(set(_ids)) > 1:
                logger.warning(
                    f"Multiple paper queries found for {paper_info['title']}:\n  "
                    + "\n  ".join(map(str, sorted(set(self._queries))))
                )
            self._selected_id = _ids[0]

        # Try to find the downloaded/converted pdf using the original form of
        # the converted pdf to txt
        if not self._selected_id and pdf_txts:
            # Favor the first pdf found, it's usually the most relevent and
            # easiest to download / access
            self._selected_id = pdf_txts[0].stem

        if not self._selected_id and self.pdf_txt:
            self._selected_id = self._paper_id

        self._pdf_txts = (
            _glob(str(CFG.dir.cache), self.LINK_ID_TEMPLATE.format(link_id=self.id))
            + _glob(
                str(CFG.dir.cache),
                self.PAPER_ID_TEMPLATE.format(paper_id=self._paper_id),
            )
            + _glob(
                str(CFG.dir.cache),
                self.PAPER_ID_FULLTEXT_TEMPLATE.format(paper_id=self._paper_id),
            )
        )
        self._pdfs = (
            [
                p.with_suffix(".pdf")
                for p in self.pdf_txts
                if p.with_suffix(".pdf").exists()
            ]
            + _glob(
                str(CFG.dir.cache),
                self.LINK_ID_TEMPLATE.replace(".txt", ".pdf").format(link_id=self.id),
            )
            + _glob(
                str(CFG.dir.cache),
                self.PAPER_ID_TEMPLATE.replace(".txt", ".pdf").format(
                    paper_id=self._paper_id
                ),
            )
            + _glob(
                str(CFG.dir.cache),
                self.PAPER_ID_FULLTEXT_TEMPLATE.replace(".txt", ".pdf").format(
                    paper_id=self._paper_id
                ),
            )
        )

    @property
    def id(self):
        return self._selected_id or self._paper_id

    @property
    def queries(self):
        return self._queries

    @property
    def pdf_txts(self):
        return self._pdf_txts

    @property
    def pdf_txt(self):
        return next(
            iter(self.pdf_txts),
            None,
        )

    @property
    def pdfs(self):
        return self._pdfs

    @property
    def _paper_id(self):
        return self.paper_info["paper_id"]

    def get_link_id_pdf_txt(self):
        """Return a hardlink, with selected id as name, to the pdf. Currently,
        the pdf file name is used as an id to check if the query should be done
        or not. As the pdf file name changed with the up-to-date paperoni cache
        structure, a hardlink might be created and returned to avoid redoing the
        query
        """
        link_id_pdf = None

        if self.pdf_txt:
            link_id_pdf = self.pdf_txt.with_stem(self.id)

        if link_id_pdf and not link_id_pdf.exists():
            link_id_pdf.hardlink_to(self.pdf_txt)

        return link_id_pdf


class PaperMD(PaperBase):
    # Original form of the converted pdf to txt. eg data/cache/*/ARXIV_ID.txt
    LINK_ID_TEMPLATE = "*/{link_id}.md"
    # Extended form of the converted pdf to txt. eg data/cache/*/PAPER_ID.txt
    PAPER_ID_TEMPLATE = LINK_ID_TEMPLATE.format(link_id="{paper_id}")
    # The the up-to-date form of the converted pdf (by paperoni)
    # eg data/cache/fulltext/PAPER_ID/fulltext.txt
    PAPER_ID_FULLTEXT_TEMPLATE = "fulltext/{paper_id}/fulltext.md"


class PaperTxt(PaperBase):
    # Original form of the converted pdf to txt. eg data/cache/*/ARXIV_ID.txt
    LINK_ID_TEMPLATE = "*/{link_id}.txt"
    # Extended form of the converted pdf to txt. eg data/cache/*/PAPER_ID.txt
    PAPER_ID_TEMPLATE = LINK_ID_TEMPLATE.format(link_id="{paper_id}")
    # The the up-to-date form of the converted pdf (by paperoni)
    # eg data/cache/fulltext/PAPER_ID/fulltext.txt
    PAPER_ID_FULLTEXT_TEMPLATE = "fulltext/{paper_id}/fulltext.txt"


def str_normalize(string):
    # Normalize to NFD (decomposed form) and filter out combining characters
    # This converts accented characters to their base form (é -> e, ñ -> n, etc.)
    string = unicodedata.normalize("NFD", string)
    string = "".join(c for c in string if not unicodedata.combining(c))
    string = string.lower()
    string = [_s.split("}}") for _s in string.split("{{")]
    string = sum(string, [])
    exclude = string[1:2]
    string = list(
        map(
            lambda _s: re.sub(pattern=r"[^a-z0-9]", string=_s, repl=""),
            string[:1] + string[2:],
        )
    )
    string = "".join(string[:1] + exclude + string[1:])
    return string

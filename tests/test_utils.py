"""Tests for paperazzi.utils module."""

import json
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
from packaging.version import Version

from paperazzi.utils import (
    DiskCachedFunc,
    DiskStore,
    DiskStoreFunc,
    disk_cache,
    disk_store,
)


def make_key(args, kwargs):
    return "default_key"


class TestDiskStore:
    """Test DiskStore functionality."""

    def test_get_file(self):
        """Test basic get_file functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            store = DiskStore(cache_dir=cache_dir, make_key=make_key)

            # Test basic file generation
            cache_file = store.get_file((), {})
            expected_file = cache_dir.resolve() / "default_key"
            assert cache_file == expected_file

            # Test file generation with custom key
            cache_file = store.get_file(key="custom_key")
            expected_file = cache_dir.resolve() / "custom_key"
            assert cache_file == expected_file

    def test_get_file_with_prefix_version_index(self):
        """Test get_file with both version and index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            version = Version("0.1.2")

            store = DiskStore(
                cache_dir=cache_dir,
                make_key=make_key,
                prefix="prefix",
                version=version,
                index=5,
            )

            cache_file = store.get_file((), {})
            expected_file = (
                cache_dir.resolve() / f"{version}" / "prefix_default_key_0005"
            )
            assert cache_file == expected_file

    def test_get_file_with_custom_key(self):
        """Test get_file with custom key parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            store = DiskStore(cache_dir=cache_dir, make_key=make_key)

            cache_file = store.get_file(key="custom_key")
            expected_file = cache_dir.resolve() / "custom_key"
            assert cache_file == expected_file

    def test_iter_files(self):
        """Test iter_files functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            store = DiskStore(cache_dir=cache_dir, make_key=make_key)

            # Create some test files
            base_file = store.get_file((), {})
            base_file.parent.mkdir(parents=True, exist_ok=True)

            test_files = [
                base_file.parent / f"{base_file.name}",
                base_file.parent / f"{base_file.name}_0000",
                base_file.parent / f"{base_file.name}_0001",
                base_file.parent / f"{base_file.name}_0002",
                base_file.parent / "other_file_0000",  # This shouldn't be included
            ]

            for file in test_files:
                file.touch()

            # Test iteration
            found_files = list(store.iter_files((), {}))

            # Should find the first 4 files but not the "other_file"
            assert sorted(set(test_files) & set(found_files)) == test_files[:-1]
            assert list(set(test_files) - set(found_files)) == test_files[-1:]

    def test_move_to(self):
        """Test move_to functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            # Create source and destination stores
            source_store = DiskStore(
                cache_dir=cache_dir, prefix="src", make_key=make_key
            )
            dest_store = DiskStore(
                cache_dir=cache_dir,
                prefix="dst",
                make_key=lambda a, k: "new_key",
                version=Version("1.0.0"),
                index=0,
            )

            # Create source file
            source_file = source_store.get_file((), {})
            source_file.parent.mkdir(parents=True, exist_ok=True)
            source_file.write_text("test content")

            # Move file
            dest_file = source_store.move_to(dest_store, (), {})

            # Check that file was moved
            assert not source_file.exists()
            assert dest_file.exists()
            assert dest_file.read_text() == "test content"

            # Check destination path is correct
            assert dest_file == dest_store.get_file((), {})

            # Check that move_to fails when destination exists
            source_file.write_text("retest content")
            with pytest.raises(FileExistsError):
                source_store.move_to(dest_store, (), {})

            # Check that move_to succeeds when force is True
            source_store.move_to(dest_store, (), {}, force=True)

            # Check that file was moved
            assert not source_file.exists()
            assert dest_file.exists()
            assert dest_file.read_text() == "retest content"

    def test_move_to_with_custom_key(self):
        """Test move_to with custom key parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            # Create source and destination stores
            source_store = DiskStore(
                cache_dir=cache_dir, prefix="src", make_key=make_key
            )
            dest_store = DiskStore(cache_dir=cache_dir, prefix="dst", make_key=make_key)

            # Create source file with custom key
            source_file = source_store.get_file(key="custom_key")
            source_file.parent.mkdir(parents=True, exist_ok=True)
            source_file.write_text("test content")

            # Move file with custom key
            dest_file = source_store.move_to(dest_store, key="custom_key")

            # Check that file was moved correctly
            assert not source_file.exists()
            assert dest_file.exists()
            assert dest_file.read_text() == "test content"

            # Check destination path uses custom key
            expected_dest = dest_store.get_file(key="custom_key")
            assert dest_file == expected_dest


class TestDiskCacheDecorator:
    @disk_cache
    def cached_randint(a, b=100):
        return random.randint(a, b)

    def test_disk_cached(self):
        random.seed(42)
        assert self.cached_randint(0, 100) == self.cached_randint(0, 100)
        assert self.cached_randint(0, 100) != self.cached_randint(1, 100)

        exists, cache_file = self.cached_randint.exists(0, 100)
        assert exists and cache_file.exists()

    def test_disk_cache_decorator(self):
        # Test default parameters
        @disk_cache
        def _():
            return None

        assert disk_cache(_).info.cache_dir == disk_cache(lambda: None).info.cache_dir
        assert disk_cache(_).info.serializer == disk_cache(lambda: None).info.serializer
        assert disk_cache(_).info.make_key == disk_cache(lambda: None).info.make_key

        assert _.info.cache_dir == disk_cache(lambda: None).info.cache_dir
        assert _.info.serializer == disk_cache(lambda: None).info.serializer
        assert _.info.make_key == disk_cache(lambda: None).info.make_key

        # Test custom parameters
        def custom_make_key(args, kwargs):
            return "custom_key"

        @disk_cache(
            cache_dir=Path(tempfile.gettempdir()) / "1234",
            serializer=json,
            make_key=custom_make_key,
        )
        def _():
            pass

        assert _.info.cache_dir == (Path(tempfile.gettempdir()) / "1234").resolve()
        assert _.info.serializer == json
        assert _.info.make_key is custom_make_key

    def test_disk_cached_update(self):
        @disk_cache
        def _():
            return None

        def new_make_key(args, kwargs):
            return "new_make_key"

        _: DiskCachedFunc = _.update(
            cache_dir=Path(tempfile.gettempdir()) / "1234",
            serializer=None,
            make_key=new_make_key,
        )

        assert _.info.cache_dir == (Path(tempfile.gettempdir()) / "1234").resolve()
        assert _.info.make_key is new_make_key
        # Make sure None values are not updated
        assert _.info.serializer == disk_cache(lambda: None).info.serializer

    def test_disk_cached_exists(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            cache_dir = Path(cache_dir)
            cached_func: DiskCachedFunc = self.cached_randint.update(
                cache_dir=cache_dir
            )

            exists, cache_file_before = cached_func.exists(1, y=2)

            # Test same arguments produce the same cache file
            assert cached_func.exists(1, y=2)[1] == cache_file_before

            # Test cache_file does not exist
            assert not exists and not cache_file_before.exists()
            assert cache_file_before.parent == cache_dir.resolve()
            assert cache_file_before.name.startswith(
                self.cached_randint.info.func.__name__
            )

            cache_file_before.touch()

            exists, cache_file = cached_func.exists(1, y=2)

            # Test cache_file does exist
            assert exists and cache_file.exists()
            assert cache_file == cache_file_before

            # Test different file is produced for different arguments
            assert cache_file != cached_func.exists(2, y=2)[1].resolve()

    def test_disk_cached_exists_custom_make_key(self):
        """Test exists method with custom make_key function."""
        with tempfile.TemporaryDirectory() as cache_dir:
            cache_dir = Path(cache_dir)

            def custom_make_key(args, kwargs):
                return "custom_key"

            cached_func: DiskCachedFunc = self.cached_randint.update(
                make_key=custom_make_key
            )

            _, cache_file = cached_func.exists(1, y=2)

            assert cache_file.name.split("_")[-2:] == ["custom", "key"]

    def test_disk_cached_with_custom_serializer(self):
        """Test disk_cached with custom serializer."""

        @dataclass
        class JsonSerializer:
            @staticmethod
            def dump(data, file_obj):
                return file_obj.write(json.dumps(data).encode("utf-8"))

            load = json.load

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            @disk_cache(cache_dir=cache_dir, serializer=JsonSerializer)
            def json_func(data):
                return {"result": data * 2}

            result = json_func(5)
            assert result == {"result": 10}

            # Verify it works on subsequent calls (cached)
            result2 = json_func(5)
            assert result2 == {"result": 10}

            assert json_func(5) == json.loads(json_func.exists(5)[1].read_text())


class TestDiskStoreDecorator:
    """Test disk_store decorator and DiskStoreFunc functionality."""

    @disk_store
    @disk_cache
    def disk_store_randint(a, b=100):
        return random.randint(a, b)

    def test_disk_store(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            disk_store_randint: DiskStoreFunc = self.disk_store_randint.update(
                cache_dir=cache_dir
            )

            for index in range(2):
                indexed_disk_store_randint = disk_store_randint.update(index=index)
                assert not indexed_disk_store_randint.exists(0, 100)[0]
                random.seed(42 + index)
                assert indexed_disk_store_randint(0, 100) == indexed_disk_store_randint(
                    0, 100
                )

            assert disk_store_randint.update(index=0).exists(0, 100)[0]
            assert disk_store_randint.update(index=0)(
                0, 100
            ) != disk_store_randint.update(index=1)(0, 100)

    def test_disk_store_decorator(self):
        """Test basic disk_store decorator functionality."""

        # Test default parameters
        @disk_store
        @disk_cache
        def _():
            return None

        assert _.info.cache_dir == disk_store(_).info.cache_dir
        assert _.info.serializer == disk_store(_).info.serializer
        assert _.info.make_key == disk_store(_).info.make_key
        assert _.info.store == disk_store(_).info.store

        # Test custom store parameters
        def custom_store_make_key(args, kwargs):
            return "custom_store_key"

        custom_store = DiskStore(
            cache_dir=Path(tempfile.gettempdir()) / "5678",
            make_key=custom_store_make_key,
            prefix="custom_store_prefix",
            version=Version("0.1.2"),
            index=10,
        )

        @disk_store(store=custom_store)
        @disk_cache
        def _():
            pass

        assert _.info.cache_dir == custom_store.cache_dir
        assert _.info.make_key is custom_store_make_key
        assert _.info.store == custom_store

        # Test custom store lower priority
        def custom_make_key(args, kwargs):
            return "custom_key"

        @disk_store(
            cache_dir=Path(tempfile.gettempdir()) / "1234",
            serializer=json,
            make_key=custom_make_key,
            store=custom_store,
        )
        @disk_cache
        def _():
            pass

        assert _.info.cache_dir == (Path(tempfile.gettempdir()) / "1234").resolve()
        assert _.info.serializer == json
        assert _.info.make_key is custom_make_key
        assert (
            _.info.store.cache_dir == (Path(tempfile.gettempdir()) / "1234").resolve()
        )
        assert _.info.store.make_key is custom_make_key

    def test_disk_store_decorator_on_top_of_disk_cache(self):
        def custom_make_key(args, kwargs):
            return "custom_key"

        @disk_store
        @disk_cache(
            cache_dir=Path(tempfile.gettempdir()) / "1234",
            serializer=json,
            make_key=custom_make_key,
        )
        def _():
            pass

        # Check that custom parameters of disk_cache are retained
        assert _.info.cache_dir == (Path(tempfile.gettempdir()) / "1234").resolve()
        assert _.info.serializer == json
        assert _.info.make_key is custom_make_key
        assert (
            _.info.store.cache_dir == (Path(tempfile.gettempdir()) / "1234").resolve()
        )
        assert _.info.store.make_key is custom_make_key

    def test_disk_store_exists(self):
        """Test exists method of DiskStoreFunc."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            disk_store_randint: DiskCachedFunc = self.disk_store_randint.update(
                cache_dir=cache_dir
            )

            exists, cache_file = disk_store_randint.exists(5)
            assert not exists
            assert not cache_file.exists()
            assert cache_file == disk_store_randint.info.store.get_file((5,), {})

            store = DiskStore(
                cache_dir=cache_dir,
                make_key=make_key,
                prefix="prefix",
                version=Version("0.1.2"),
                index=10,
            )

            disk_store_randint: DiskCachedFunc = self.disk_store_randint.update(
                store=store
            )

            exists, cache_file = disk_store_randint.exists(5)
            assert not exists
            assert not cache_file.exists()
            assert cache_file == disk_store_randint.info.store.get_file((5,), {})

    def test_disk_store_update(self):
        """Test updating DiskStoreFunc with new store."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)

            def custom_make_key(args, kwargs):
                return "custom_key"

            def custom_store_make_key(args, kwargs):
                return "custom_store_key"

            custom_store = DiskStore(
                cache_dir=cache_dir / "1234",
                make_key=custom_store_make_key,
                prefix="custom_store_prefix",
                version=Version("0.1.2"),
                index=10,
            )

            assert (
                self.disk_store_randint.update(cache_dir=cache_dir).info.store.cache_dir
                == cache_dir.resolve()
            )
            assert (
                self.disk_store_randint.update(
                    make_key=custom_make_key
                ).info.store.make_key
                is custom_make_key
            )
            assert (
                self.disk_store_randint.update(prefix="custom_prefix").info.store.prefix
                == "custom_prefix"
            )
            assert self.disk_store_randint.update(
                version=Version("2.1.0")
            ).info.store.version == Version("2.1.0")
            assert self.disk_store_randint.update(index=10).info.store.index == 10
            assert (
                self.disk_store_randint.update(custom_store).info.store == custom_store
            )

            # check store lower priority
            assert (
                self.disk_store_randint.update(
                    custom_store, cache_dir=cache_dir
                ).info.store.cache_dir
                == cache_dir.resolve()
            )

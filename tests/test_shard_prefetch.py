"""Exercise prefetch scheduling and cache safety without Azure requests."""

from __future__ import annotations

import io
import json
import multiprocessing
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

import pytest

from dl_azure.storage import AzureShardCache, CacheCapacityError, ShardPrefetcher


def _url(name: str) -> str:
    return f"https://demo.blob.core.windows.net/data/{name}.tar?sig=secret"


@pytest.fixture
def blobs(monkeypatch: Any) -> Any:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        member = tarfile.TarInfo("sample.json")
        member.size = 2
        archive.addfile(member, io.BytesIO(b"{}"))
    payload = buffer.getvalue()
    state = SimpleNamespace(
        payload=payload,
        calls=[],
        gate=threading.Event(),
        entered=threading.Event(),
        active=0,
        peak=0,
        fail=False,
        lock=threading.Lock(),
    )
    state.gate.set()

    class Client:
        def __init__(self, url: str, **kwargs: Any) -> None:
            self.url = url

        def get_blob_properties(self, **kwargs: Any) -> Any:
            return SimpleNamespace(size=len(payload), etag='"v1"')

        def download_blob(self, **kwargs: Any) -> Any:
            def readinto(handle: Any) -> int:
                with state.lock:
                    state.calls.append(self.url)
                    state.active += 1
                    state.peak = max(state.peak, state.active)
                    state.entered.set()
                try:
                    assert state.gate.wait(5), "Test did not release download"
                    if state.fail:
                        raise OSError("Request failed: sig=secret")
                    return handle.write(payload)
                finally:
                    with state.lock:
                        state.active -= 1

            return SimpleNamespace(readinto=readinto)

        def close(self) -> None:
            pass

    monkeypatch.setattr("dl_azure.storage.shard_cache.BlobClient.from_blob_url", Client)
    monkeypatch.setattr("dl_azure.storage.download.shutil.which", lambda executable: None)
    from dl_azure.storage.download import _AZCOPY_PATHS

    _AZCOPY_PATHS.clear()
    yield state
    state.gate.set()
    _AZCOPY_PATHS.clear()


def _cache(tmp_path: Path, *, shards: int = 20) -> AzureShardCache:
    return AzureShardCache(
        str(tmp_path / "cache"),
        cache_size_bytes=10240 * shards,
        download_retries=0,
        lock_timeout_seconds=2,
    )


def _wait_state(prefetch: ShardPrefetcher, state: str, key: str | None = None) -> None:
    deadline = time.monotonic() + 3
    while state not in prefetch.status(key).values():
        assert time.monotonic() < deadline, prefetch.status()
        time.sleep(0.01)


def test_independent_thresholds_and_duplicate_plans(tmp_path: Path, blobs: Any) -> None:
    cache = _cache(tmp_path)
    with ShardPrefetcher(cache, trigger_fraction=0.5) as prefetch:
        prefetch.plan("A", current=[_url("a")], upcoming=[_url("d")])
        prefetch.plan("B", current=[_url("b")], upcoming=[_url("e")])
        prefetch.plan("C", current=[_url("c")], upcoming=[_url("d")])
        prefetch.advance("A", consumed=49, total=100)
        assert not blobs.calls
        assert set(prefetch.status().values()) == {"planned"}
        prefetch.advance("A", consumed=50, total=100)
        prefetch.wait("A", timeout=3)
        assert blobs.calls == [_url("d")]
        prefetch.advance("C", consumed=80, total=100)
        prefetch.wait("C", timeout=3)
        assert blobs.calls == [_url("d")]
        prefetch.release("A")
        assert set(prefetch.status("C").values()) == {"ready"}
        prefetch.advance("B", consumed=1, total=2)
        prefetch.wait("B", timeout=3)
        assert blobs.calls == [_url("d"), _url("e")]
        assert "sig=" not in repr(prefetch.status())
    assert not list(cache.pin_dir.glob("*.pin"))


def test_cycle_concurrency_queue_bound_and_foreground_join(
    tmp_path: Path, blobs: Any
) -> None:
    cache = _cache(tmp_path)
    blobs.gate.clear()
    with ShardPrefetcher(
        cache, max_concurrent_downloads=2, max_pending_shards=3
    ) as prefetch:
        prefetch.plan("cycle", current=[], upcoming=[_url(name) for name in "abc"])
        with pytest.raises(ValueError, match="queue is full"):
            prefetch.plan("overflow", current=[], upcoming=[_url("d")])
        prefetch.advance("cycle", consumed=5, total=10)
        assert blobs.entered.wait(2)
        try:
            with ThreadPoolExecutor(max_workers=1) as readers:
                reading = readers.submit(cache.ensure, _url("a"))
                assert not reading.done()
                blobs.gate.set()
                assert reading.result(timeout=3).is_file()
        finally:
            blobs.gate.set()
        assert len(prefetch.wait("cycle", timeout=3)) == 3
        assert len(blobs.calls) == 3
        assert blobs.peak <= 2


def test_active_and_upcoming_files_are_protected(tmp_path: Path, blobs: Any) -> None:
    cache = _cache(tmp_path, shards=2)
    first = cache.ensure(_url("active"))
    other_cache = _cache(tmp_path, shards=2)
    with ShardPrefetcher(cache) as prefetch:
        prefetch.plan("slot", current=[_url("active")], upcoming=[_url("next")])
        upcoming = prefetch.wait("slot", timeout=3)[0]
        with pytest.raises(CacheCapacityError):
            other_cache.ensure(_url("third"))
        assert first.exists() and upcoming.exists()
        # A reader's own reservation outlives the trainer's plan reservation.
        stream = cache([_url("active")])
        sample = next(stream)
        try:
            prefetch.release("slot")
            other_cache.ensure(_url("third"))
            assert first.exists()
            assert not upcoming.exists()
            assert sample["stream"].read(1)
        finally:
            stream.close()


def test_capacity_wait_resumes_when_another_plan_releases(
    tmp_path: Path, blobs: Any
) -> None:
    cache = _cache(tmp_path, shards=1)
    with ShardPrefetcher(cache) as prefetch:
        prefetch.plan("previous", current=[], upcoming=[_url("previous")])
        prefetch.wait("previous", timeout=3)
        prefetch.plan("new", current=[], upcoming=[_url("new")])
        prefetch.advance("new", consumed=1, total=1)
        _wait_state(prefetch, "waiting_for_space")
        with pytest.raises(CacheCapacityError):
            prefetch.wait("new", timeout=3)
        prefetch.release("previous")
        _wait_state(prefetch, "ready", "new")
        assert prefetch.wait("new", timeout=3)[0].exists()


def test_inflight_bytes_count_against_cache_budget(tmp_path: Path, blobs: Any) -> None:
    cache = _cache(tmp_path, shards=1)
    blobs.gate.clear()
    with ShardPrefetcher(cache, max_concurrent_downloads=2) as prefetch:
        prefetch.plan("two", current=[], upcoming=[_url("a"), _url("b")])
        prefetch.advance("two", consumed=1, total=1)
        try:
            assert blobs.entered.wait(2)
            _wait_state(prefetch, "waiting_for_space")
            assert len(blobs.calls) == 1
        finally:
            blobs.gate.set()
    assert not list(cache.part_dir.iterdir())
    assert not list(cache.pin_dir.iterdir())


def test_failure_and_timeout_are_visible_without_credentials(
    tmp_path: Path, blobs: Any
) -> None:
    cache = _cache(tmp_path)
    blobs.fail = True
    with ShardPrefetcher(cache) as prefetch:
        prefetch.plan("bad", current=[], upcoming=[_url("bad")])
        with pytest.raises(RuntimeError, match="Azure shard download failed") as error:
            prefetch.wait("bad", timeout=3)
        assert "sig=" not in str(error.value)
        assert set(prefetch.status("bad").values()) == {"failed"}
        assert not list(cache.part_dir.iterdir())
    blobs.fail = False
    blobs.gate.clear()
    with ShardPrefetcher(cache) as prefetch:
        prefetch.plan("slow", current=[], upcoming=[_url("slow")])
        try:
            with pytest.raises(TimeoutError):
                prefetch.wait("slow", timeout=0.02)
        finally:
            blobs.gate.set()
        assert prefetch.wait("slow", timeout=3)[0].exists()


@pytest.mark.parametrize(
    "options",
    [
        {"trigger_fraction": -0.1},
        {"trigger_fraction": 1.1},
        {"trigger_fraction": float("nan")},
        {"max_concurrent_downloads": 0},
        {"max_pending_shards": 1.5},
        {"max_pending_shards": True},
    ],
)
def test_rejects_invalid_options(tmp_path: Path, options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ShardPrefetcher(_cache(tmp_path), **options)


def test_disabled_prefetch_preserves_lazy_loading(tmp_path: Path, blobs: Any) -> None:
    with ShardPrefetcher(_cache(tmp_path), enabled=False) as prefetch:
        prefetch.plan("slot", current=[_url("a")], upcoming=[_url("b")])
        assert prefetch.advance("slot", consumed=1, total=1) == {}
        assert prefetch.wait("slot") == []
        prefetch.release("slot")
    assert blobs.calls == []


def _hold_reservation(
    cache_dir: str, ready: Any, release: Any, path_resolver: Any = None
) -> None:
    cache = AzureShardCache(
        cache_dir, cache_size_bytes=10240, path_resolver=path_resolver
    )
    with cache.reserve(_url("a")):
        ready.set()
        release.wait(5)


def test_reservations_protect_files_across_processes(
    tmp_path: Path, blobs: Any
) -> None:
    cache = _cache(tmp_path, shards=1)
    path = cache.ensure(_url("a"))
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(
        target=_hold_reservation, args=(str(cache.cache_dir), ready, release)
    )
    process.start()
    try:
        assert ready.wait(15)
        with pytest.raises(CacheCapacityError):
            cache.ensure(_url("b"))
        assert path.exists()
    finally:
        release.set()
        process.join(15)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0
    cache.ensure(_url("b"))
    assert not path.exists()


def _nested_destination(public_url: str) -> Path:
    return Path(urlsplit(public_url).path.lstrip("/").split("/", 1)[1])


def test_custom_destinations_keep_same_basenames_distinct(
    tmp_path: Path, blobs: Any
) -> None:
    cache = AzureShardCache(
        str(tmp_path / "cache"),
        cache_size_bytes=20480,
        path_resolver=_nested_destination,
    )
    urls = [_url("first/same"), _url("second/same")]
    first, second = [cache.ensure(url) for url in urls]
    assert first == tmp_path / "cache/first/same.tar"
    assert second == tmp_path / "cache/second/same.tar"
    assert first.is_file() and second.is_file()
    with cache.reserve(urls[0]):
        cache.ensure(_url("third/same"))
        assert first.is_file()
        assert not second.exists()
    assert len(blobs.calls) == 3


def test_azcopy_download_keeps_cache_paths_and_reservations(
    tmp_path: Path, blobs: Any, monkeypatch: Any,
) -> None:
    calls = []
    monkeypatch.setattr("dl_azure.storage.download.shutil.which", lambda name: "/installed/azcopy")

    def run(command: list[str], **kwargs: Any) -> Any:
        calls.append(command[2])
        Path(command[3]).write_bytes(blobs.payload)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("dl_azure.storage.download.subprocess.run", run)
    cache = AzureShardCache(
        str(tmp_path / "cache"), cache_size_bytes=20480,
        path_resolver=_nested_destination,
    )
    first_url = _url("first/same")
    first = cache.ensure(first_url)
    with cache.reserve(first_url):
        second = cache.ensure(_url("second/same"))
        cache.ensure(_url("third/same"))
        assert first.exists() and not second.exists()
    assert first == tmp_path / "cache/first/same.tar"
    assert calls == [first_url, _url("second/same"), _url("third/same")]
    assert not blobs.calls
    assert not list(cache.part_dir.iterdir())
    assert not list(cache.cache_dir.rglob("*.part"))


def test_absolute_destination_and_state_directory(
    tmp_path: Path, blobs: Any, monkeypatch: Any
) -> None:
    destination = tmp_path / "project/data/custom.tar"
    received = []

    def resolve(public_url: str) -> Path:
        received.append(public_url)
        return destination

    cache = AzureShardCache(
        str(tmp_path / "default"),
        cache_size_bytes=10240,
        path_resolver=resolve,
        state_dir=tmp_path / "state",
    )
    checked = []
    from dl_azure.storage import shard_cache

    original_disk_usage = shard_cache.shutil.disk_usage
    monkeypatch.setattr(
        shard_cache.shutil,
        "disk_usage",
        lambda path: (checked.append(path), original_disk_usage(path))[1],
    )
    assert cache.ensure(_url("custom")) == destination
    assert destination.is_file()
    assert checked == [destination.parent]
    assert cache.lock_dir == tmp_path / "state/locks"
    assert all("?" not in url for url in received)
    assert not list(destination.parent.glob("*.part"))


def test_custom_destination_conflicts_are_rejected(tmp_path: Path, blobs: Any) -> None:
    cache = AzureShardCache(
        str(tmp_path / "cache"),
        cache_size_bytes=20480,
        path_resolver=lambda url: "fixed.tar",
    )
    first = cache.ensure(_url("first"))
    with pytest.raises(ValueError, match="different shard"):
        cache.ensure(_url("second"))
    assert first.exists()
    assert blobs.calls == [_url("first")]


def test_cache_hits_preserve_content_mtime_and_unmanaged_files(
    tmp_path: Path, blobs: Any
) -> None:
    cache = AzureShardCache(
        str(tmp_path / "cache"),
        cache_size_bytes=10240,
        path_resolver=_nested_destination,
    )
    unmanaged = cache.cache_dir / "user-owned.tar"
    unmanaged.write_bytes(b"leave this file alone")
    first = cache.ensure(_url("first"))
    before = first.stat().st_mtime_ns
    cache.ensure(_url("first").replace("secret", "renewed"))
    assert first.stat().st_mtime_ns == before
    cache.ensure(_url("second"))
    assert not first.exists()
    assert unmanaged.read_bytes() == b"leave this file alone"


def test_abandoned_partial_at_custom_destination_is_reclaimed(
    tmp_path: Path, blobs: Any
) -> None:
    cache = AzureShardCache(
        str(tmp_path / "cache"),
        cache_size_bytes=10240,
        path_resolver=_nested_destination,
    )
    destination = cache.local_path(_url("abandoned"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.parent / ".abandoned.part"
    partial.write_bytes(b"partial download")
    size_record = cache.part_dir / f"{cache._cache_key(destination)}.size"
    size_record.write_text(
        json.dumps(
            {
                "size": 10240,
                "partial": str(partial),
                "device": destination.parent.stat().st_dev,
            }
        )
    )
    cache.ensure(_url("different"))
    assert not partial.exists()
    assert not size_record.exists()


def test_custom_destinations_are_protected_across_processes(
    tmp_path: Path, blobs: Any
) -> None:
    cache = AzureShardCache(
        str(tmp_path / "cache"),
        cache_size_bytes=10240,
        path_resolver=_nested_destination,
    )
    first = cache.ensure(_url("a"))
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(
        target=_hold_reservation,
        args=(str(cache.cache_dir), ready, release, _nested_destination),
    )
    process.start()
    try:
        assert ready.wait(15)
        with pytest.raises(CacheCapacityError):
            cache.ensure(_url("b"))
        assert first.is_file()
    finally:
        release.set()
        process.join(15)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0
    cache.ensure(_url("b"))
    assert not first.exists()


def test_wrapper_custom_cache_is_shared_by_prefetch_and_indexed_reads(
    tmp_path: Path, blobs: Any, monkeypatch: Any
) -> None:
    from dl_azure.datasets import AzureStreamingTarShardWrapper
    from dl_core.datasets import IndexedTarDataset
    from torch.utils.data import DataLoader

    class Wrapper(AzureStreamingTarShardWrapper):
        def get_shard_cache_path(self, blob_path: str) -> Path:
            return tmp_path / "project" / blob_path

        def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
            return {"key": file_dict["key"]}

    service = SimpleNamespace(
        get_blob_sas_url=lambda container, path, **kwargs: _url(Path(path).stem)
    )
    monkeypatch.setattr(
        "dl_azure.datasets.base.AzureClientService", lambda config: service
    )
    wrapper = Wrapper(
        {
            "account_name": "demo",
            "container_name": "data",
            "azure_config_path": str(tmp_path / "missing.json"),
            "auto_split": False,
            "track_shard_progress": True,
            "cache": {"cache_dir": str(tmp_path / "cache")},
            "indexed_tar": {"index_dir": str(tmp_path / "indexes")},
            "prefetch": {"enabled": True},
            "shards": {"train": ["a.tar"]},
        }
    )
    data = wrapper.get_shard_sources("train")
    with wrapper.cached_shard_sources(data) as local:
        assert local[0]["shards"][0]["path"] == str(tmp_path / "project/a.tar")
        dataset = wrapper.build_indexed_dataset(local, "train")
        assert isinstance(dataset, IndexedTarDataset)
        wrapper.reset_shard_progress(dataset.shard_totals)
        with wrapper.create_shard_prefetcher() as prefetch:
            prefetch.plan("a.tar", current=["a.tar"], upcoming=["b.tar"])
            for batch in DataLoader(dataset, batch_size=1):
                wrapper.record_shard_consumption(batch["shard_id"])
                progress = wrapper.get_shard_progress("a.tar")
                prefetch.advance(
                    "a.tar", consumed=progress["consumed"], total=progress["total"]
                )
            assert prefetch.wait("a.tar", timeout=3) == [tmp_path / "project/b.tar"]
            assert "sig=" not in repr(dataset[0])
        dataset.close()
    assert blobs.calls == [_url("a"), _url("b")]
    assert next(iter(wrapper.get_split("train")))["shard_id"] == ["a.tar"]
    assert blobs.calls == [_url("a"), _url("b")]
    assert not list(wrapper.create_shard_cache().pin_dir.glob("*.pin"))


def test_indexed_source_context_reserves_whole_selection_before_download(
    tmp_path: Path, blobs: Any, monkeypatch: Any
) -> None:
    from dl_azure.datasets import AzureStreamingTarShardWrapper

    class Wrapper(AzureStreamingTarShardWrapper):
        def get_shard_cache_path(self, blob_path: str) -> Path:
            return Path(blob_path)

        def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
            return file_dict

    service = SimpleNamespace(
        get_blob_sas_url=lambda container, path, **kwargs: _url(Path(path).stem)
    )
    monkeypatch.setattr(
        "dl_azure.datasets.base.AzureClientService", lambda config: service
    )
    wrapper = Wrapper(
        {
            "account_name": "demo",
            "container_name": "data",
            "azure_config_path": str(tmp_path / "missing.json"),
            "auto_split": False,
            "cache": {
                "cache_dir": str(tmp_path / "cache"),
                "cache_size_gb": 20480 / 1024**3,
            },
            "shards": {"train": ["a.tar", "b.tar"]},
        }
    )
    cache = wrapper.create_shard_cache()
    existing = cache.ensure(_url("b"))
    unrelated = cache.ensure(_url("unrelated"))
    with wrapper.cached_shard_sources(wrapper.get_shard_sources("train")) as local:
        assert existing.is_file()
        assert not unrelated.exists()
        assert all(Path(shard["path"]).is_file() for shard in local[0]["shards"])
    assert blobs.calls == [_url("b"), _url("unrelated"), _url("a")]


def test_wrapper_path_hook_decodes_blob_name_once(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from dl_azure.datasets import AzureStreamingTarShardWrapper

    class Wrapper(AzureStreamingTarShardWrapper):
        def get_shard_cache_path(self, blob_path: str) -> Path:
            return Path(blob_path)

        def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
            return file_dict

    monkeypatch.setattr(
        "dl_azure.datasets.base.AzureClientService", lambda config: SimpleNamespace()
    )
    wrapper = Wrapper(
        {
            "account_name": "demo",
            "container_name": "data",
            "azure_config_path": str(tmp_path / "missing.json"),
            "auto_split": False,
            "cache": {"cache_dir": str(tmp_path / "cache")},
        }
    )
    cache = wrapper.create_shard_cache()
    url = "https://demo.blob.core.windows.net/data/folder/literal%252F.tar?sig=secret"
    assert cache.local_path(url) == tmp_path / "cache/folder/literal%2F.tar"


def test_old_sas_tokens_share_download_identity(tmp_path: Path, blobs: Any) -> None:
    cache = _cache(tmp_path)
    with ShardPrefetcher(cache) as prefetch:
        prefetch.plan("a", current=[], upcoming=[_url("a")])
        prefetch.plan(
            "renewed", current=[], upcoming=[_url("a").replace("secret", "renewed")]
        )
        assert prefetch.wait("a", timeout=3) == prefetch.wait("renewed", timeout=3)
        assert len(blobs.calls) == 1


def test_full_cache_does_not_starve_already_cached_request(
    tmp_path: Path, blobs: Any
) -> None:
    cache = _cache(tmp_path, shards=1)
    cache.ensure(_url("cached"))
    with ShardPrefetcher(cache, max_concurrent_downloads=1) as prefetch:
        prefetch.plan("blocked", current=[_url("cached")], upcoming=[_url("new")])
        prefetch.advance("blocked", consumed=1, total=1)
        _wait_state(prefetch, "waiting_for_space")
        prefetch.plan("cached", current=[], upcoming=[_url("cached")])
        assert prefetch.wait("cached", timeout=3)[0].exists()


def test_wrapper_prefetch_keeps_training_order(
    tmp_path: Path, blobs: Any, monkeypatch: Any
) -> None:
    from dl_azure.datasets import AzureStreamingTarShardWrapper

    class Wrapper(AzureStreamingTarShardWrapper):
        def transform(self, file_dict: dict[str, Any], split: str) -> dict[str, Any]:
            return {"source_path": file_dict["source_path"], "key": file_dict["key"]}

    service = SimpleNamespace(
        get_blob_sas_url=lambda container, path, **kwargs: _url(Path(path).stem)
    )
    monkeypatch.setattr(
        "dl_azure.datasets.base.AzureClientService", lambda config: service
    )
    wrapper = Wrapper(
        {
            "account_name": "demo",
            "container_name": "data",
            "azure_config_path": str(tmp_path / "missing.json"),
            "shards": {"train": ["a.tar", "b.tar"]},
            "auto_split": False,
            "batch_size": 1,
            "num_workers": 0,
            "shuffle": False,
            "cache": {"cache_dir": str(tmp_path / "cache")},
            "prefetch": {"enabled": True, "trigger_fraction": 0.75},
        }
    )
    with wrapper.create_shard_prefetcher() as prefetch:
        prefetch.plan("slot", current=["a.tar"], upcoming=["b.tar"])
        prefetch.advance("slot", consumed=2, total=4)
        assert not blobs.calls
        prefetch.advance("slot", consumed=3, total=4)
        prefetch.wait("slot", timeout=3)
        assert blobs.calls == [_url("b")]
        loader = wrapper.get_split("train")
        assert [batch["source_path"][0] for batch in loader] == ["a.tar", "b.tar"]
        assert blobs.calls == [_url("b"), _url("a")]


def test_cancelled_queued_jobs_keep_the_admission_bound(
    tmp_path: Path, blobs: Any
) -> None:
    cache = _cache(tmp_path)
    blobs.gate.clear()
    with ShardPrefetcher(
        cache, max_concurrent_downloads=1, max_pending_shards=2
    ) as prefetch:
        try:
            prefetch.plan("running", current=[], upcoming=[_url("running")])
            prefetch.advance("running", consumed=1, total=1)
            assert blobs.entered.wait(2)
            prefetch.plan("queued", current=[], upcoming=[_url("queued")])
            prefetch.advance("queued", consumed=1, total=1)
            prefetch.release("queued")
            with pytest.raises(ValueError, match="queue is full"):
                prefetch.plan("extra", current=[], upcoming=[_url("extra")])
        finally:
            blobs.gate.set()
    assert _url("queued") not in blobs.calls
    assert not list(cache.pin_dir.iterdir())


def test_recovers_abandoned_download_and_reservation(
    tmp_path: Path, blobs: Any
) -> None:
    cache = _cache(tmp_path, shards=1)
    old = cache.ensure(_url("old"))
    (cache.pin_dir / f"{old.name}.abandoned.pin").touch()
    upcoming = cache.local_path(_url("next"))
    (cache.part_dir / f"{upcoming.name}.size").write_text(str(len(blobs.payload)))
    (cache.part_dir / f".{upcoming.name}.abandoned.part").write_bytes(b"partial")

    assert cache.ensure(_url("next")) == upcoming
    assert upcoming.exists() and not old.exists()
    assert not list(cache.part_dir.iterdir())
    assert not list(cache.pin_dir.iterdir())

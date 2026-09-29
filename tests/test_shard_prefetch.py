"""Exercise prefetch scheduling and cache safety without Azure requests."""

from __future__ import annotations

import io
import multiprocessing
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
        def __init__(self, url: str) -> None:
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
    yield state
    state.gate.set()


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


def _hold_reservation(cache_dir: str, ready: Any, release: Any) -> None:
    cache = AzureShardCache(cache_dir, cache_size_bytes=10240)
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


def test_recovers_abandoned_download_and_reservation(tmp_path: Path, blobs: Any) -> None:
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

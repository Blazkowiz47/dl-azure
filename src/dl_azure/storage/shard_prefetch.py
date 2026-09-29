"""Bounded background downloads driven by explicit shard replacement plans."""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import CancelledError, Future, TimeoutError
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from urllib.parse import urlsplit, urlunsplit

from dl_azure.storage.shard_cache import AzureShardCache, CacheCapacityError


@dataclass
class _Download:
    url: str = field(repr=False)
    state: str = "planned"
    future: Future[Path] | None = None
    stop: threading.Event = field(default_factory=threading.Event)


@dataclass
class _Plan:
    downloads: list[Path]
    reservations: ExitStack


class ShardPrefetcher:
    """Own one training process's plans, queue, and upcoming-shard reservations.

    Call plan/advance/wait/release from the owning trainer thread. Downloads
    run in a bounded thread pool; cache locks coordinate with other processes.
    """

    def __init__(
        self,
        cache: AzureShardCache,
        *,
        enabled: bool = True,
        trigger_fraction: float = 0.5,
        max_concurrent_downloads: int = 2,
        max_pending_shards: int = 32,
        resolve_url: Callable[[str], str] | None = None,
    ) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("prefetch.enabled must be a boolean")
        if not math.isfinite(trigger_fraction) or not 0 <= trigger_fraction <= 1:
            raise ValueError("prefetch.trigger_fraction must be between 0 and 1")
        for name, value in (
            ("max_concurrent_downloads", max_concurrent_downloads),
            ("max_pending_shards", max_pending_shards),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"prefetch.{name} must be a positive integer")
        self.cache = cache
        self.enabled = enabled
        self.trigger_fraction = trigger_fraction
        self.max_concurrent_downloads = max_concurrent_downloads
        self.max_pending_shards = max_pending_shards
        self.resolve_url = resolve_url
        self._owner = (os.getpid(), threading.get_ident())
        self._plans: dict[str, _Plan] = {}
        self._downloads: dict[Path, _Download] = {}
        self._threads: list[threading.Thread] = []
        self._queue: Queue[_Download] = Queue(maxsize=max_pending_shards)
        self._closing = threading.Event()
        self._closed = False

    def _check_owner(self) -> None:
        if self._owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError(
                "Use the shard prefetcher only from its owning trainer thread"
            )
        if self._closed:
            raise RuntimeError("Shard prefetcher is closed")

    def plan(
        self, key: str, *, current: Iterable[str], upcoming: Iterable[str]
    ) -> None:
        """Reserve an immutable replacement plan; release its key before reuse."""
        self._check_owner()
        if not self.enabled:
            return
        if key in self._plans:
            raise ValueError(
                "Prefetch plan already exists; release it before reusing its key"
            )
        if isinstance(current, str) or isinstance(upcoming, str):
            raise TypeError("current and upcoming must be collections of shard paths")
        current_urls = list(dict.fromkeys(current))
        upcoming_urls = list(dict.fromkeys(upcoming))
        if not upcoming_urls:
            raise ValueError("A prefetch plan must have at least one upcoming shard")
        if self.resolve_url is not None:
            current_urls = [self.resolve_url(path) for path in current_urls]
            upcoming_urls = [self.resolve_url(path) for path in upcoming_urls]
        upcoming_by_path = {self.cache.local_path(url): url for url in upcoming_urls}
        # Released jobs may still be finishing an HTTP request. Keep them in
        # the admission count until finished so repeated plans cannot grow it.
        referenced = {path for plan in self._plans.values() for path in plan.downloads}
        for path, download in list(self._downloads.items()):
            if path not in referenced and (
                download.future is None or download.future.done()
            ):
                del self._downloads[path]
        if (
            len(self._downloads.keys() | upcoming_by_path.keys())
            > self.max_pending_shards
        ):
            raise ValueError(
                "Prefetch queue is full; release plans or increase max_pending_shards"
            )
        if any(
            self._downloads[path].stop.is_set()
            for path in upcoming_by_path
            if path in self._downloads
        ):
            raise RuntimeError(
                "A released download is still stopping; retry the plan after it finishes"
            )
        with ExitStack() as reservations:
            for url in dict.fromkeys(current_urls + upcoming_urls):
                reservations.enter_context(self.cache.reserve(url))
            self._plans[key] = _Plan(list(upcoming_by_path), reservations.pop_all())
        for path, url in upcoming_by_path.items():
            self._downloads.setdefault(path, _Download(url))

    def _work(self) -> None:
        while not self._closing.is_set():
            try:
                download = self._queue.get(timeout=0.1)
            except Empty:
                continue
            future = download.future
            assert future is not None
            if download.stop.is_set():
                download.state = "cancelled"
                future.set_exception(CancelledError("Shard prefetch was cancelled"))
                continue
            download.state = "downloading"
            try:
                path = self.cache.ensure(download.url)
            except CacheCapacityError:
                download.state = "waiting_for_space"
                # Rotate blocked requests so they cannot starve cached shards
                # or smaller downloads waiting behind them.
                self._queue.put_nowait(download)
                self._closing.wait(0.25)
            except Exception as exc:
                download.state = "failed"
                future.set_exception(exc)
            else:
                download.state = "ready"
                future.set_result(path)

    def advance(self, key: str, *, consumed: float, total: float) -> dict[str, str]:
        """Queue every replacement once consumption reaches the configured fraction."""
        self._check_owner()
        if not self.enabled:
            return {}
        if not math.isfinite(total) or total <= 0:
            raise ValueError(
                "Prefetch progress total must be finite and greater than zero"
            )
        if not math.isfinite(consumed) or consumed < 0:
            raise ValueError(
                "Prefetch consumed progress must be finite and nonnegative"
            )
        plan = self._plans[key]
        if consumed / total >= self.trigger_fraction:
            if not self._threads:
                for index in range(self.max_concurrent_downloads):
                    thread = threading.Thread(
                        target=self._work, name=f"azure-shard-{index}", daemon=True
                    )
                    thread.start()
                    self._threads.append(thread)
            for path in plan.downloads:
                download = self._downloads[path]
                if download.future is None:
                    download.state = "queued"
                    download.future = Future()
                    self._queue.put_nowait(download)
        return self.status(key)

    def status(self, key: str | None = None) -> dict[str, str]:
        """Return SAS-free shard URLs and planned/queued/downloading/ready/error states."""
        self._check_owner()
        if not self.enabled:
            return {}
        paths = self._plans[key].downloads if key is not None else self._downloads
        result = {}
        for path in paths:
            download = self._downloads[path]
            parsed = urlsplit(download.url)
            result[urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))] = (
                download.state
            )
        return result

    def wait(self, key: str, *, timeout: float | None = None) -> list[Path]:
        """Require a plan now; surface failures or insufficient cache capacity.

        Reservations remain held after this call until release() or close().
        A timeout leaves the background downloads running.
        """
        self._check_owner()
        if not self.enabled:
            return []
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("Prefetch timeout must be finite and nonnegative")
        self.advance(key, consumed=1, total=1)
        deadline = None if timeout is None else time.monotonic() + timeout
        downloads = [self._downloads[path] for path in self._plans[key].downloads]
        while True:
            pending = False
            paths = []
            for download in downloads:
                future = download.future
                assert future is not None
                if future.done():
                    paths.append(future.result())
                else:
                    pending = True
                    if download.state == "waiting_for_space":
                        raise CacheCapacityError(
                            "Prefetch is waiting for cache space; release finished plans "
                            "or increase cache.cache_size_gb"
                        )
            if not pending:
                return paths
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for shard prefetch")
            time.sleep(0.01)

    def release(self, key: str) -> None:
        """Release a finished plan; cancel jobs no remaining plan needs."""
        self._check_owner()
        if not self.enabled:
            return
        plan = self._plans.pop(key)
        plan.reservations.close()
        referenced = {path for plan in self._plans.values() for path in plan.downloads}
        for path in plan.downloads:
            if path not in referenced:
                download = self._downloads[path]
                download.stop.set()
                if download.future is None:
                    download.state = "cancelled"

    def close(self) -> None:
        """Cancel queued work, finish in-flight requests, and release reservations."""
        if self._closed:
            return
        self._check_owner()
        try:
            self._closing.set()
            for download in self._downloads.values():
                download.stop.set()
            for thread in self._threads:
                thread.join()
        finally:
            for download in self._downloads.values():
                if download.future is not None and not download.future.done():
                    download.state = "cancelled"
                    download.future.set_exception(
                        CancelledError("Shard prefetch was cancelled")
                    )
            for plan in self._plans.values():
                plan.reservations.close()
            self._plans.clear()
            self._downloads.clear()
            self._closed = True

    def __enter__(self) -> ShardPrefetcher:
        self._check_owner()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

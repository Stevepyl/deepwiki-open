"""Process-local cache for prepared RAG retrievers."""

from __future__ import annotations

import hashlib
import gc
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote

from api.config import get_embedder_type
from api.rag import RAG

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagPrepareResult:
    rag: RAG
    cache_hit: bool
    documents_count: int
    prepare_latency_sec: float
    cache_key: str


@dataclass
class _CacheEntry:
    rag: RAG
    documents_count: int
    created_at: float
    last_accessed_at: float


_cache: OrderedDict[str, _CacheEntry] = OrderedDict()
_cache_lock = threading.Lock()
_build_locks: dict[str, threading.Lock] = {}


def parse_filter_list(value: str | None) -> list[str] | None:
    """Parse UI/API filter text into a stable list.

    Existing clients typically send newline-separated values, while the API
    descriptions say comma-separated. Support both without changing callers.
    """
    if value is None:
        return None
    items = [unquote(item.strip()) for item in re.split(r"[\n,]+", value) if item.strip()]
    return items or None


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return max(minimum, int(raw_value))
    except ValueError:
        logger.warning("Invalid integer for %s=%r. Using default %s.", name, raw_value, default)
        return default


def _cache_enabled() -> bool:
    return _env_bool("DEEPWIKI_RAG_CACHE_ENABLED", True)


def _max_entries() -> int:
    return _env_int("DEEPWIKI_RAG_CACHE_MAX_ENTRIES", 2, 1)


def _ttl_seconds() -> int:
    return _env_int("DEEPWIKI_RAG_CACHE_TTL_SECONDS", 3600, 1)


def _normalize_list(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(item.strip() for item in values if item and item.strip())


def _token_hash(token: str | None) -> str:
    if not token:
        return ""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def build_rag_cache_key(
    *,
    repo_url: str,
    repo_type: str | None,
    provider: str | None,
    model: str | None,
    token: str | None,
    excluded_dirs: list[str] | None,
    excluded_files: list[str] | None,
    included_dirs: list[str] | None,
    included_files: list[str] | None,
) -> str:
    parts: tuple[Any, ...] = (
        repo_url.strip(),
        (repo_type or "github").strip(),
        (provider or "google").strip(),
        (model or "").strip(),
        get_embedder_type(),
        _normalize_list(excluded_dirs),
        _normalize_list(excluded_files),
        _normalize_list(included_dirs),
        _normalize_list(included_files),
        _token_hash(token),
    )
    return repr(parts)


def _is_expired(entry: _CacheEntry, now: float) -> bool:
    return now - entry.created_at > _ttl_seconds()


def _evict_expired_locked(now: float) -> None:
    expired_keys = [key for key, entry in _cache.items() if _is_expired(entry, now)]
    for key in expired_keys:
        logger.info("Evicting expired RAG cache entry: %s", key)
        _cache.pop(key, None)


def _evict_lru_locked() -> None:
    while len(_cache) > _max_entries():
        evicted_key, _ = _cache.popitem(last=False)
        logger.info("Evicting LRU RAG cache entry: %s", evicted_key)


def _evict_lru_for_new_entry_locked() -> int:
    evicted_count = 0
    while len(_cache) >= _max_entries():
        evicted_key, _ = _cache.popitem(last=False)
        evicted_count += 1
        logger.info("Evicting LRU RAG cache entry before building new entry: %s", evicted_key)
    return evicted_count


def _get_cached_entry(key: str, now: float) -> _CacheEntry | None:
    with _cache_lock:
        _evict_expired_locked(now)
        entry = _cache.get(key)
        if entry is None:
            return None
        entry.last_accessed_at = now
        _cache.move_to_end(key)
        return entry


def _get_build_lock(key: str) -> threading.Lock:
    with _cache_lock:
        lock = _build_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _build_locks[key] = lock
        return lock


def _prepare_uncached_rag(
    *,
    repo_url: str,
    repo_type: str | None,
    token: str | None,
    provider: str | None,
    model: str | None,
    excluded_dirs: list[str] | None,
    excluded_files: list[str] | None,
    included_dirs: list[str] | None,
    included_files: list[str] | None,
) -> RAG:
    rag = RAG(provider=provider or "google", model=model)
    rag.prepare_retriever(
        repo_url,
        repo_type or "github",
        token,
        excluded_dirs,
        excluded_files,
        included_dirs,
        included_files,
    )
    return rag


def get_prepared_rag(
    *,
    repo_url: str,
    repo_type: str | None,
    token: str | None,
    provider: str | None,
    model: str | None,
    excluded_dirs: list[str] | None = None,
    excluded_files: list[str] | None = None,
    included_dirs: list[str] | None = None,
    included_files: list[str] | None = None,
) -> RagPrepareResult:
    started_at = time.perf_counter()
    key = build_rag_cache_key(
        repo_url=repo_url,
        repo_type=repo_type,
        provider=provider,
        model=model,
        token=token,
        excluded_dirs=excluded_dirs,
        excluded_files=excluded_files,
        included_dirs=included_dirs,
        included_files=included_files,
    )

    if not _cache_enabled():
        rag = _prepare_uncached_rag(
            repo_url=repo_url,
            repo_type=repo_type,
            token=token,
            provider=provider,
            model=model,
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files,
        )
        return RagPrepareResult(
            rag=rag,
            cache_hit=False,
            documents_count=len(getattr(rag, "transformed_docs", []) or []),
            prepare_latency_sec=round(time.perf_counter() - started_at, 3),
            cache_key=key,
        )

    now = time.monotonic()
    entry = _get_cached_entry(key, now)
    if entry is not None:
        logger.info("RAG cache hit for %s", repo_url)
        return RagPrepareResult(
            rag=entry.rag,
            cache_hit=True,
            documents_count=entry.documents_count,
            prepare_latency_sec=round(time.perf_counter() - started_at, 3),
            cache_key=key,
        )

    build_lock = _get_build_lock(key)
    with build_lock:
        now = time.monotonic()
        entry = _get_cached_entry(key, now)
        if entry is not None:
            logger.info("RAG cache hit after wait for %s", repo_url)
            return RagPrepareResult(
                rag=entry.rag,
                cache_hit=True,
                documents_count=entry.documents_count,
                prepare_latency_sec=round(time.perf_counter() - started_at, 3),
                cache_key=key,
            )

        with _cache_lock:
            _evict_expired_locked(time.monotonic())
            evicted_count = _evict_lru_for_new_entry_locked()
        if evicted_count:
            gc.collect()

        logger.info("RAG cache miss for %s. Preparing retriever.", repo_url)
        rag = _prepare_uncached_rag(
            repo_url=repo_url,
            repo_type=repo_type,
            token=token,
            provider=provider,
            model=model,
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files,
        )
        documents_count = len(getattr(rag, "transformed_docs", []) or [])

        with _cache_lock:
            stored_at = time.monotonic()
            _cache[key] = _CacheEntry(
                rag=rag,
                documents_count=documents_count,
                created_at=stored_at,
                last_accessed_at=stored_at,
            )
            _cache.move_to_end(key)
            _evict_lru_locked()

        return RagPrepareResult(
            rag=rag,
            cache_hit=False,
            documents_count=documents_count,
            prepare_latency_sec=round(time.perf_counter() - started_at, 3),
            cache_key=key,
        )


def clear_rag_cache() -> None:
    """Clear cache for tests and operational troubleshooting."""
    with _cache_lock:
        _cache.clear()
        _build_locks.clear()

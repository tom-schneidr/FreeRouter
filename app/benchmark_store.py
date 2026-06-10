from __future__ import annotations

import json
import os
from dataclasses import dataclass
from time import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.model_catalog import ModelRoute


_MIN_INDEX_SCORE = 1
_MAX_INDEX_SCORE = 60


@dataclass(frozen=True)
class BenchmarkScoreEntry:
    index: int
    source: str
    confidence: str
    updated_at: int


@dataclass(frozen=True)
class BenchmarkStoreSnapshot:
    updated_at: int | None
    source_url: str | None
    scores: dict[str, BenchmarkScoreEntry]


class BenchmarkStore:
    """Persisted Artificial Analysis index scores refreshed via FreeRouter web search."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._updated_at: int | None = None
        self._source_url: str | None = None
        self._scores: dict[str, BenchmarkScoreEntry] = {}
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.path):
            self._updated_at = None
            self._source_url = None
            self._scores = {}
            return

        with open(self.path, encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            self._scores = {}
            return

        updated_at = payload.get("updated_at")
        self._updated_at = int(updated_at) if isinstance(updated_at, (int, float)) else None
        source_url = payload.get("source_url")
        self._source_url = source_url if isinstance(source_url, str) else None

        raw_scores = payload.get("scores")
        scores: dict[str, BenchmarkScoreEntry] = {}
        if isinstance(raw_scores, dict):
            for key, value in raw_scores.items():
                if not isinstance(key, str) or not isinstance(value, dict):
                    continue
                index = value.get("index")
                if not isinstance(index, (int, float)):
                    continue
                index_int = int(index)
                if not _MIN_INDEX_SCORE <= index_int <= _MAX_INDEX_SCORE:
                    continue
                entry_updated = value.get("updated_at")
                scores[key.lower()] = BenchmarkScoreEntry(
                    index=index_int,
                    source=str(value.get("source") or "web-search"),
                    confidence=str(value.get("confidence") or "medium"),
                    updated_at=int(entry_updated) if isinstance(entry_updated, (int, float)) else 0,
                )
        self._scores = scores
        _notify_scores_changed()

    def save(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        payload = {
            "updated_at": self._updated_at,
            "source_url": self._source_url,
            "scores": {
                key: {
                    "index": entry.index,
                    "source": entry.source,
                    "confidence": entry.confidence,
                    "updated_at": entry.updated_at,
                }
                for key, entry in sorted(self._scores.items())
            },
        }
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        _notify_scores_changed()

    def snapshot(self) -> BenchmarkStoreSnapshot:
        return BenchmarkStoreSnapshot(
            updated_at=self._updated_at,
            source_url=self._source_url,
            scores=dict(self._scores),
        )

    def index_scores_map(self) -> dict[str, int]:
        return {key: entry.index for key, entry in self._scores.items()}

    def is_stale(self, *, now: int | None = None, max_age_seconds: int) -> bool:
        timestamp = int(time()) if now is None else now
        if self._updated_at is None:
            return True
        return timestamp - self._updated_at >= max_age_seconds

    def merge_scores(
        self,
        scores: dict[str, int],
        *,
        source: str,
        confidence: str,
        source_url: str | None = None,
        updated_at: int | None = None,
    ) -> int:
        timestamp = int(time()) if updated_at is None else updated_at
        merged = 0
        for raw_key, raw_index in scores.items():
            if not isinstance(raw_key, str):
                continue
            if not isinstance(raw_index, (int, float)):
                continue
            index = int(raw_index)
            if not _MIN_INDEX_SCORE <= index <= _MAX_INDEX_SCORE:
                continue
            key = raw_key.strip().lower()
            if not key:
                continue
            self._scores[key] = BenchmarkScoreEntry(
                index=index,
                source=source,
                confidence=confidence,
                updated_at=timestamp,
            )
            merged += 1

        if merged > 0:
            self._updated_at = timestamp
            if source_url:
                self._source_url = source_url
            self.save()
        return merged

    def routes_without_dynamic_match(self, routes: list[ModelRoute]) -> list[ModelRoute]:
        from app.model_ranking import route_has_dynamic_benchmark_match

        dynamic = {key: entry.index for key, entry in self._scores.items()}
        return [
            route for route in routes if not route_has_dynamic_benchmark_match(route, dynamic)
        ]


_store: BenchmarkStore | None = None
_store_path: str | None = None


def get_benchmark_store(path: str | None = None) -> BenchmarkStore:
    global _store, _store_path
    if path is not None and (_store is None or _store_path != path):
        _store = BenchmarkStore(path)
        _store_path = path
    if _store is None:
        raise RuntimeError("BenchmarkStore path is not configured")
    return _store


def _notify_scores_changed() -> None:
    from app.model_ranking import invalidate_dynamic_benchmark_cache

    invalidate_dynamic_benchmark_cache()


def reset_benchmark_store_for_tests() -> None:
    global _store, _store_path
    _store = None
    _store_path = None
    _notify_scores_changed()

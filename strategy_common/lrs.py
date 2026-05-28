from __future__ import annotations

import json
import hashlib
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class ResearchArtifactVersion:
    strategy_id: str
    artifact_date: date
    artifact_type: str
    artifact_hash: str
    version: int
    created_at_utc: str
    payload: dict[str, Any]


class LRSDatabase:
    """Small shared Local Research Store surface used by live/backtest parity.

    This shared layer adds strategy-scoped immutable research artifacts so
    candidate snapshots are not sourced from ad hoc JSON files.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS research_artifact (
        strategy_id TEXT NOT NULL,
        artifact_date TEXT NOT NULL,
        artifact_type TEXT NOT NULL,
        artifact_hash TEXT NOT NULL,
        version INTEGER NOT NULL,
        created_at_utc TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        PRIMARY KEY (strategy_id, artifact_date, artifact_type, version)
    );
    CREATE INDEX IF NOT EXISTS idx_research_artifact_latest
        ON research_artifact(strategy_id, artifact_date, artifact_type, version);
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(self.SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save_artifact(
        self,
        artifact_date: date,
        artifact: dict[str, Any],
        *,
        strategy_id: str,
        artifact_type: str = "candidate_snapshot",
        artifact_hash: str = "",
    ) -> ResearchArtifactVersion:
        strategy = strategy_id.upper().strip()
        payload = dict(artifact)
        resolved_hash = artifact_hash or str(payload.get("artifact_hash", "")) or _stable_payload_hash(payload)
        with self._conn() as conn:
            existing = conn.execute(
                """
                SELECT *
                FROM research_artifact
                WHERE strategy_id = ? AND artifact_date = ? AND artifact_type = ?
                ORDER BY version DESC
                LIMIT 1
                """,
                (strategy, artifact_date.isoformat(), artifact_type),
            ).fetchone()
            if existing is not None and str(existing["artifact_hash"]) == resolved_hash:
                return ResearchArtifactVersion(
                    strategy_id=existing["strategy_id"],
                    artifact_date=date.fromisoformat(existing["artifact_date"]),
                    artifact_type=existing["artifact_type"],
                    artifact_hash=existing["artifact_hash"],
                    version=int(existing["version"]),
                    created_at_utc=existing["created_at_utc"],
                    payload=json.loads(existing["payload_json"]),
                )
            version = int(existing["version"] or 0) + 1 if existing is not None else 1
            created_at = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO research_artifact
                    (strategy_id, artifact_date, artifact_type, artifact_hash, version, created_at_utc, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy,
                    artifact_date.isoformat(),
                    artifact_type,
                    resolved_hash,
                    version,
                    created_at,
                    json.dumps(payload, sort_keys=True, default=str),
                ),
            )
        return ResearchArtifactVersion(strategy, artifact_date, artifact_type, resolved_hash, version, created_at, payload)

    def load_artifact(
        self,
        artifact_date: date,
        *,
        strategy_id: str,
        artifact_type: str = "candidate_snapshot",
        version: int | None = None,
    ) -> ResearchArtifactVersion | None:
        strategy = strategy_id.upper().strip()
        with self._conn() as conn:
            if version is None:
                row = conn.execute(
                    """
                    SELECT *
                    FROM research_artifact
                    WHERE strategy_id = ? AND artifact_date = ? AND artifact_type = ?
                    ORDER BY version DESC
                    LIMIT 1
                    """,
                    (strategy, artifact_date.isoformat(), artifact_type),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT *
                    FROM research_artifact
                    WHERE strategy_id = ? AND artifact_date = ? AND artifact_type = ? AND version = ?
                    """,
                    (strategy, artifact_date.isoformat(), artifact_type, int(version)),
                ).fetchone()
        if row is None:
            return None
        return ResearchArtifactVersion(
            strategy_id=row["strategy_id"],
            artifact_date=date.fromisoformat(row["artifact_date"]),
            artifact_type=row["artifact_type"],
            artifact_hash=row["artifact_hash"],
            version=int(row["version"]),
            created_at_utc=row["created_at_utc"],
            payload=json.loads(row["payload_json"]),
        )


def _stable_payload_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

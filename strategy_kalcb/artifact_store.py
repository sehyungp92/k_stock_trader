from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path

from .models import KALCBDailySnapshot


_FINAL_ARTIFACT_STAGE = "daily_finalized_candidate"


class KALCBArtifactStore:
    """Source-fingerprinted daily candidate snapshot store for KALCB."""

    def __init__(self, root: str | Path = "data/strategy/kalcb"):
        self.root = Path(root)

    def save_snapshot(self, snapshot: KALCBDailySnapshot) -> Path:
        _require_finalized_snapshot(snapshot)
        path = self.path_for(snapshot.trade_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(snapshot.to_json_dict(), indent=2, sort_keys=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(payload, encoding="utf-8")
        for attempt in range(6):
            try:
                tmp.replace(path)
                return path
            except PermissionError:
                existing = self.load_snapshot(snapshot.trade_date) if path.exists() else None
                if existing is not None and existing.artifact_hash == snapshot.artifact_hash:
                    tmp.unlink(missing_ok=True)
                    return path
                if attempt >= 5:
                    raise
                time.sleep(0.05 * (attempt + 1))
        return path

    def load_snapshot(self, trade_date: date) -> KALCBDailySnapshot | None:
        path = self.path_for(trade_date)
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        snapshot = KALCBDailySnapshot.from_json_dict(payload)
        recorded_hash = str(payload.get("artifact_hash") or "")
        if recorded_hash and recorded_hash != snapshot.artifact_hash:
            raise ValueError(f"KALCB candidate artifact hash mismatch: {path}")
        return snapshot

    def path_for(self, trade_date: date) -> Path:
        return self.root / f"candidate_snapshot_{trade_date.isoformat()}.json"


def save_snapshot_to_lrs(snapshot: KALCBDailySnapshot, lrs) -> object:
    _require_finalized_snapshot(snapshot)
    return lrs.save_artifact(
        snapshot.trade_date,
        snapshot.to_json_dict(),
        strategy_id="KALCB",
        artifact_type="candidate_snapshot",
        artifact_hash=snapshot.artifact_hash,
    )


def load_snapshot_from_lrs(trade_date: date, lrs) -> KALCBDailySnapshot | None:
    version = lrs.load_artifact(
        trade_date,
        strategy_id="KALCB",
        artifact_type="candidate_snapshot",
    )
    if version is None:
        return None
    snapshot = KALCBDailySnapshot.from_json_dict(version.payload)
    if version.artifact_hash and snapshot.artifact_hash != version.artifact_hash:
        raise ValueError("KALCB LRS candidate artifact hash mismatch")
    return snapshot


def _require_finalized_snapshot(snapshot: KALCBDailySnapshot) -> None:
    stage = str((snapshot.metadata or {}).get("artifact_stage") or "").strip()
    if stage != _FINAL_ARTIFACT_STAGE:
        raise ValueError(
            "KALCBArtifactStore only persists finalized executable candidate snapshots; "
            "call strategy_kalcb.research.finalize_candidate_snapshot() first."
        )

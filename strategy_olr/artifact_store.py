from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path

from .models import OLRDailySnapshot


OLR_STAGE1_ARTIFACT_STAGE = "stage1_daily_candidate"
OLR_FINAL_ARTIFACT_STAGE = "final_afternoon_1430"
_STAGE_DIRS = {
    OLR_STAGE1_ARTIFACT_STAGE: "stage1",
    OLR_FINAL_ARTIFACT_STAGE: "final",
}
_STAGE_ARTIFACT_TYPES = {
    OLR_STAGE1_ARTIFACT_STAGE: "candidate_snapshot_stage1",
    OLR_FINAL_ARTIFACT_STAGE: "candidate_snapshot_final",
}


class OLRArtifactStore:
    """Source-fingerprinted daily candidate snapshot store for OLR."""

    def __init__(self, root: str | Path = "data/strategy/olr"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_snapshot(self, snapshot: OLRDailySnapshot, *, artifact_stage: str | None = None) -> Path:
        stage = artifact_stage or infer_olr_artifact_stage(snapshot)
        path = self.path_for(snapshot.trade_date, artifact_stage=stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(snapshot.to_json_dict(), indent=2, sort_keys=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(payload, encoding="utf-8")
        for attempt in range(6):
            try:
                tmp.replace(path)
                return path
            except PermissionError:
                existing = self.load_snapshot(snapshot.trade_date, artifact_stage=stage) if path.exists() else None
                if existing is not None and existing.artifact_hash == snapshot.artifact_hash:
                    tmp.unlink(missing_ok=True)
                    return path
                if attempt >= 5:
                    raise
                time.sleep(0.05 * (attempt + 1))
        return path

    def load_snapshot(self, trade_date: date, *, artifact_stage: str | None = None) -> OLRDailySnapshot | None:
        stages = [_normalize_artifact_stage(artifact_stage)] if artifact_stage is not None else [None, OLR_STAGE1_ARTIFACT_STAGE, OLR_FINAL_ARTIFACT_STAGE]
        for stage in stages:
            loaded = self._load_snapshot_path(self.path_for(trade_date, artifact_stage=stage))
            if loaded is not None:
                return loaded
        return None

    def _load_snapshot_path(self, path: Path) -> OLRDailySnapshot | None:
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        snapshot = OLRDailySnapshot.from_json_dict(payload)
        recorded_hash = str(payload.get("artifact_hash") or "")
        if recorded_hash and recorded_hash != snapshot.artifact_hash:
            raise ValueError(f"OLR candidate artifact hash mismatch: {path}")
        return snapshot

    def path_for(self, trade_date: date, *, artifact_stage: str | None = None) -> Path:
        stage = _normalize_artifact_stage(artifact_stage)
        base = self.root / _stage_dir(stage) if stage else self.root
        return base / f"candidate_snapshot_{trade_date.isoformat()}.json"


def save_snapshot_to_lrs(snapshot: OLRDailySnapshot, lrs, *, artifact_stage: str | None = None) -> object:
    stage = _normalize_artifact_stage(artifact_stage or infer_olr_artifact_stage(snapshot))
    return lrs.save_artifact(
        snapshot.trade_date,
        snapshot.to_json_dict(),
        strategy_id="OLR",
        artifact_type=_artifact_type_for_stage(stage),
        artifact_hash=snapshot.artifact_hash,
    )


def load_snapshot_from_lrs(trade_date: date, lrs, *, artifact_stage: str | None = None) -> OLRDailySnapshot | None:
    stages = [_normalize_artifact_stage(artifact_stage)] if artifact_stage is not None else [None, OLR_STAGE1_ARTIFACT_STAGE, OLR_FINAL_ARTIFACT_STAGE]
    version = None
    for stage in stages:
        version = lrs.load_artifact(
            trade_date,
            strategy_id="OLR",
            artifact_type=_artifact_type_for_stage(stage),
        )
        if version is not None:
            break
    if version is None:
        return None
    snapshot = OLRDailySnapshot.from_json_dict(version.payload)
    if version.artifact_hash and snapshot.artifact_hash != version.artifact_hash:
        raise ValueError("OLR LRS candidate artifact hash mismatch")
    return snapshot


def infer_olr_artifact_stage(snapshot: OLRDailySnapshot) -> str:
    metadata = dict(snapshot.metadata or {})
    explicit = str(metadata.get("artifact_stage") or "").strip()
    if explicit:
        return _normalize_artifact_stage(explicit) or explicit
    source = str(metadata.get("source") or "").strip()
    basis = str(metadata.get("selection_time_basis") or "").strip()
    if source in {"olr_afternoon_selection", "olr_shadow_same_day_reranker"} or basis == "14:30_decision_from_completed_5m_bars":
        return OLR_FINAL_ARTIFACT_STAGE
    if source == "olr_research_selection" or basis == "pre_session_from_prior_completed_daily_rows":
        return OLR_STAGE1_ARTIFACT_STAGE
    return ""


def _normalize_artifact_stage(value: str | None) -> str:
    stage = str(value or "").strip().lower()
    if stage in {"", "legacy", "candidate_snapshot"}:
        return ""
    if stage in {"stage1", "daily", "daily_candidate", "candidate_snapshot_stage1", OLR_STAGE1_ARTIFACT_STAGE}:
        return OLR_STAGE1_ARTIFACT_STAGE
    if stage in {"final", "afternoon", "afternoon_1430", "candidate_snapshot_final", OLR_FINAL_ARTIFACT_STAGE}:
        return OLR_FINAL_ARTIFACT_STAGE
    return stage


def _stage_dir(stage: str) -> str:
    try:
        return _STAGE_DIRS[stage]
    except KeyError as exc:
        raise ValueError(f"unsupported OLR artifact_stage={stage!r}") from exc


def _artifact_type_for_stage(stage: str) -> str:
    if not stage:
        return "candidate_snapshot"
    try:
        return _STAGE_ARTIFACT_TYPES[stage]
    except KeyError as exc:
        raise ValueError(f"unsupported OLR artifact_stage={stage!r}") from exc

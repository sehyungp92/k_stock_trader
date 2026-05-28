from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from kis_core.trading_calendar import get_trading_calendar
from strategy_kalcb.artifact_store import KALCBArtifactStore
from strategy_kalcb.models import KALCBDailySnapshot
from strategy_kalcb.research import KALCB_FINAL_ARTIFACT_STAGE
from strategy_olr.artifact_store import OLR_FINAL_ARTIFACT_STAGE, OLR_STAGE1_ARTIFACT_STAGE, OLRArtifactStore
from strategy_olr.models import OLRDailySnapshot
from strategy_olr.research import FINAL_CANDIDATE_CONFIG_HASH_VERSION

KST = ZoneInfo("Asia/Seoul")
PRESESSION_READY_CUTOFF = time(9, 0)
OLR_FINAL_READY_TIME = time(14, 30)
DEFAULT_ARTIFACT_ROOTS = {
    "KALCB": Path("data/strategy/kalcb"),
    "OLR": Path("data/strategy/olr"),
}


@dataclass(frozen=True, slots=True)
class ArtifactReadinessFailure:
    strategy_id: str
    stage: str
    detail: str


def krx_trade_date(timestamp: datetime | None = None) -> date:
    ts = timestamp or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    today = ts.astimezone(KST).date()
    calendar = get_trading_calendar()
    return today if calendar.is_trading_day(today) else calendar.previous_trading_day(today)


def load_strategy_artifact(
    strategy_id: str,
    trade_date: date,
    required_stage: str,
    mode: str = "paper",
    *,
    artifact_roots: dict[str, str | Path] | None = None,
    max_candidates: int | None = None,
) -> KALCBDailySnapshot | OLRDailySnapshot:
    sid = strategy_id.upper().strip()
    mode_name = str(mode or "").strip().lower()
    roots = {**DEFAULT_ARTIFACT_ROOTS, **{key.upper(): Path(value) for key, value in dict(artifact_roots or {}).items()}}
    if sid == "KALCB":
        _validate_required_stage_for_mode(sid, required_stage, mode_name)
        snapshot = KALCBArtifactStore(roots["KALCB"]).load_snapshot(trade_date)
        _validate_snapshot(snapshot, sid, trade_date, required_stage, max_candidates=max_candidates)
        return snapshot
    if sid == "OLR":
        stage = _normalize_olr_stage(required_stage)
        _validate_required_stage_for_mode(sid, stage, mode_name)
        snapshot = OLRArtifactStore(roots["OLR"]).load_snapshot(trade_date, artifact_stage=stage)
        _validate_snapshot(snapshot, sid, trade_date, stage, max_candidates=max_candidates)
        return snapshot
    raise ValueError(f"unsupported strategy_id={strategy_id!r}")


def validate_strategy_artifacts(
    strategy_ids: list[str] | tuple[str, ...],
    *,
    trade_date: date | None = None,
    mode: str = "paper",
    artifact_roots: dict[str, str | Path] | None = None,
) -> tuple[dict[str, KALCBDailySnapshot | OLRDailySnapshot], list[ArtifactReadinessFailure]]:
    current_trade_date = trade_date or krx_trade_date()
    artifacts: dict[str, KALCBDailySnapshot | OLRDailySnapshot] = {}
    failures: list[ArtifactReadinessFailure] = []
    for raw_sid in strategy_ids:
        sid = raw_sid.upper().strip()
        stage = required_stage_for(sid, mode)
        try:
            artifacts[sid] = load_strategy_artifact(
                sid,
                current_trade_date,
                stage,
                mode,
                artifact_roots=artifact_roots,
            )
        except Exception as exc:
            failures.append(ArtifactReadinessFailure(sid, stage, str(exc)))
    return artifacts, failures


def required_stage_for(strategy_id: str, mode: str) -> str:
    sid = strategy_id.upper().strip()
    mode_name = str(mode or "").lower()
    if sid == "KALCB":
        return KALCB_FINAL_ARTIFACT_STAGE
    if sid == "OLR":
        return OLR_STAGE1_ARTIFACT_STAGE if mode_name == "artifact_only_stage1" else OLR_FINAL_ARTIFACT_STAGE
    raise ValueError(f"unsupported strategy_id={strategy_id!r}")


def _validate_snapshot(
    snapshot: Any,
    strategy_id: str,
    trade_date: date,
    required_stage: str,
    *,
    max_candidates: int | None,
) -> None:
    if snapshot is None:
        raise FileNotFoundError("artifact is missing")
    if snapshot.strategy_id.upper().strip() != strategy_id:
        raise ValueError(f"strategy_id mismatch: {snapshot.strategy_id!r}")
    if snapshot.trade_date != trade_date:
        raise ValueError(f"trade_date mismatch: {snapshot.trade_date.isoformat()} != {trade_date.isoformat()}")
    if not snapshot.source_fingerprint:
        raise ValueError("source_fingerprint is blank")
    _validate_generated_at(snapshot.generated_at, strategy_id, trade_date, required_stage)
    metadata = dict(snapshot.metadata or {})
    recorded_stage = str(metadata.get("artifact_stage") or "").strip()
    if recorded_stage != required_stage:
        raise ValueError(f"artifact_stage mismatch: {recorded_stage!r} != {required_stage!r}")
    if snapshot.to_json_dict().get("artifact_hash") != snapshot.artifact_hash:
        raise ValueError("artifact hash validation failed")
    if max_candidates is not None and len(snapshot.candidates) > max_candidates:
        raise ValueError(f"candidate count {len(snapshot.candidates)} exceeds max {max_candidates}")
    _validate_stage_metadata(strategy_id, required_stage, metadata)


def _validate_required_stage_for_mode(strategy_id: str, required_stage: str, mode: str) -> None:
    expected = required_stage_for(strategy_id, mode)
    if required_stage != expected:
        raise ValueError(
            f"{strategy_id} required_stage={required_stage!r} is not valid for mode={mode or 'paper'!r}; "
            f"expected {expected!r}"
        )


def _validate_stage_metadata(
    strategy_id: str,
    required_stage: str,
    metadata: dict[str, Any],
) -> None:
    source = str(metadata.get("source") or "").strip()
    basis = str(metadata.get("selection_time_basis") or "").strip()
    if strategy_id == "KALCB":
        if required_stage != KALCB_FINAL_ARTIFACT_STAGE:
            raise ValueError(f"KALCB requires artifact_stage={KALCB_FINAL_ARTIFACT_STAGE!r}")
        if not source:
            raise ValueError("KALCB artifact source is blank")
        if not str(metadata.get("candidate_config_hash") or "").strip():
            raise ValueError("KALCB candidate_config_hash is blank")
        return
    if strategy_id == "OLR" and required_stage == OLR_STAGE1_ARTIFACT_STAGE:
        if source != "olr_research_selection":
            raise ValueError(f"OLR stage1 source mismatch: {source!r}")
        if basis != "pre_session_from_prior_completed_daily_rows":
            raise ValueError(f"OLR stage1 selection_time_basis mismatch: {basis!r}")
        return
    if strategy_id == "OLR" and required_stage == OLR_FINAL_ARTIFACT_STAGE:
        if source not in {"olr_afternoon_selection", "olr_shadow_same_day_reranker"}:
            raise ValueError(f"OLR final source mismatch: {source!r}")
        if basis != "14:30_decision_from_completed_5m_bars":
            raise ValueError(f"OLR final selection_time_basis mismatch: {basis!r}")
        candidate_config_hash = str(metadata.get("candidate_config_hash") or "").strip()
        final_config_hash = str(metadata.get("final_candidate_config_hash") or "").strip()
        if not candidate_config_hash:
            raise ValueError("OLR final candidate_config_hash is blank")
        if not final_config_hash:
            raise ValueError("OLR final final_candidate_config_hash is blank")
        if candidate_config_hash != final_config_hash:
            raise ValueError("OLR final candidate_config_hash does not match final_candidate_config_hash")
        version = str(metadata.get("final_candidate_config_hash_version") or "").strip()
        if version != FINAL_CANDIDATE_CONFIG_HASH_VERSION:
            raise ValueError(f"OLR final final_candidate_config_hash_version mismatch: {version!r}")


def _validate_generated_at(
    generated_at: datetime,
    strategy_id: str,
    trade_date: date,
    required_stage: str,
) -> None:
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")
    try:
        generated_kst = generated_at.astimezone(KST)
    except Exception as exc:
        raise ValueError("generated_at cannot be converted to KST") from exc
    if generated_kst.date() != trade_date:
        raise ValueError(f"generated_at KST date mismatch: {generated_kst.date().isoformat()} != {trade_date.isoformat()}")
    if strategy_id == "OLR" and required_stage == OLR_FINAL_ARTIFACT_STAGE and generated_kst.time() < OLR_FINAL_READY_TIME:
        raise ValueError("OLR final generated_at is before the 14:30 KST artifact cutoff")
    if _requires_pre_session_generated_at(strategy_id, required_stage) and generated_kst.time() >= PRESESSION_READY_CUTOFF:
        raise ValueError("pre-session artifact generated_at must be before the 09:00 KST session open")


def _requires_pre_session_generated_at(strategy_id: str, required_stage: str) -> bool:
    return (strategy_id == "KALCB" and required_stage == KALCB_FINAL_ARTIFACT_STAGE) or (
        strategy_id == "OLR" and required_stage == OLR_STAGE1_ARTIFACT_STAGE
    )


def _normalize_olr_stage(stage: str) -> str:
    value = str(stage or "").strip().lower()
    if value in {"stage1", "daily", "candidate_snapshot_stage1", OLR_STAGE1_ARTIFACT_STAGE}:
        return OLR_STAGE1_ARTIFACT_STAGE
    if value in {"final", "afternoon", "candidate_snapshot_final", OLR_FINAL_ARTIFACT_STAGE}:
        return OLR_FINAL_ARTIFACT_STAGE
    return stage

from __future__ import annotations

from datetime import date, datetime
from hashlib import sha256

from backtests.auto.shared.cache_keys import stable_signature
from backtests.core.replay_bundle import EventReplayBundle
from strategy_kalcb.models import KALCBDailyCandidate, KALCBDailySnapshot


def build_feature_bundle_hash(snapshot: KALCBDailySnapshot, source_fingerprint: str) -> str:
    raw = repr((snapshot.artifact_hash, source_fingerprint, len(snapshot.candidates)))
    return sha256(raw.encode("utf-8")).hexdigest()


def build_feature_bundle_hash_for_snapshots(snapshots: dict[date, KALCBDailySnapshot], source_fingerprint: str) -> str:
    return stable_signature(
        {
            "source_fingerprint": source_fingerprint,
            "candidate_hashes": {day.isoformat(): snapshot.artifact_hash for day, snapshot in sorted(snapshots.items())},
            "candidate_counts": {day.isoformat(): len(snapshot.candidates) for day, snapshot in sorted(snapshots.items())},
        }
    )


def snapshot_from_bundle(bundle: EventReplayBundle) -> KALCBDailySnapshot | None:
    metadata = dict(bundle.metadata or {})
    raw_snapshot = metadata.get("kalcb_candidate_snapshot")
    if isinstance(raw_snapshot, dict):
        return KALCBDailySnapshot.from_json_dict(raw_snapshot)
    raw_candidates = metadata.get("kalcb_candidates")
    if isinstance(raw_candidates, list):
        candidates = tuple(KALCBDailyCandidate.from_json_dict(row) for row in raw_candidates)
        if candidates:
            return KALCBDailySnapshot(
                trade_date=candidates[0].trade_date,
                candidates=candidates,
                source_fingerprint=bundle.source_fingerprint,
                generated_at=datetime.fromisoformat(str(metadata.get("generated_at"))) if metadata.get("generated_at") else datetime.now(),
                metadata={"bundle_metadata": metadata},
            )
    return None


def snapshots_from_bundle(bundle: EventReplayBundle) -> dict[date, KALCBDailySnapshot]:
    metadata = dict(bundle.metadata or {})
    raw_snapshots = metadata.get("kalcb_candidate_snapshots")
    if isinstance(raw_snapshots, dict):
        snapshots: dict[date, KALCBDailySnapshot] = {}
        for key, value in raw_snapshots.items():
            if isinstance(value, dict):
                snapshot = KALCBDailySnapshot.from_json_dict(value)
                snapshots[date.fromisoformat(str(key))] = snapshot
        if snapshots:
            return snapshots
    single = snapshot_from_bundle(bundle)
    if single is not None:
        return {single.trade_date: single}
    return {}


def require_kalcb_feature_metadata(bundle: EventReplayBundle, snapshot: KALCBDailySnapshot) -> None:
    metadata = dict(bundle.metadata or {})
    candidate_hash = metadata.get("kalcb_candidate_artifact_hash")
    feature_hash = metadata.get("kalcb_feature_bundle_hash")
    expected_feature_hash = build_feature_bundle_hash(snapshot, bundle.source_fingerprint)
    candidate_hashes = metadata.get("kalcb_candidate_artifact_hashes")
    if isinstance(candidate_hashes, dict):
        snapshots = snapshots_from_bundle(bundle)
        expected_hashes = {day.isoformat(): item.artifact_hash for day, item in snapshots.items()}
        expected_feature_hash = build_feature_bundle_hash_for_snapshots(snapshots, bundle.source_fingerprint)
        if any(item.source_fingerprint and item.source_fingerprint != bundle.source_fingerprint for item in snapshots.values()):
            raise ValueError("KALCB candidate source fingerprint mismatch")
        if candidate_hashes != expected_hashes:
            raise ValueError("KALCB candidate artifact hash mismatch")
        if feature_hash and feature_hash != expected_feature_hash:
            raise ValueError("KALCB feature bundle hash mismatch")
        if not feature_hash:
            raise ValueError("KALCB official replay requires candidate and feature bundle hashes")
        return
    if snapshot.source_fingerprint and snapshot.source_fingerprint != bundle.source_fingerprint:
        raise ValueError("KALCB candidate source fingerprint mismatch")
    if candidate_hash and candidate_hash != snapshot.artifact_hash:
        raise ValueError("KALCB candidate artifact hash mismatch")
    if feature_hash and feature_hash != expected_feature_hash:
        raise ValueError("KALCB feature bundle hash mismatch")
    if not candidate_hash or not feature_hash:
        raise ValueError("KALCB official replay requires candidate and feature bundle hashes")

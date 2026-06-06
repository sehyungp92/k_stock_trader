from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .hashing import canonical_json_hash, file_sha256, json_file_hash, jsonl_file_hash

try:
    from instrumentation.src.lineage import LineageContext
    from instrumentation.src.runtime_exporter import RuntimeAssistantExporter
except Exception:  # pragma: no cover - instrumentation must be fail-open
    LineageContext = Any  # type: ignore
    RuntimeAssistantExporter = None  # type: ignore

REQUIRED_JSONL = (
    "artifact_generation.jsonl",
    "subscription_events.jsonl",
    "decision_stream.jsonl",
    "strategy_actions.jsonl",
    "portfolio_arbitration.jsonl",
    "oms_intents.jsonl",
    "order_events.jsonl",
    "fill_events.jsonl",
    "trade_outcomes.jsonl",
    "state_snapshots.jsonl",
)
REQUIRED_DIRS = ("daily_snapshots", "olr_stage1_snapshots", "olr_final_snapshots")
MARKET_BARS_FILE = "market_bars_5m.parquet"
KIS_RESOURCE_PLAN_FILE = "kis_resource_plan.json"
HASH_CONTRACT_VERSION = "paper-session-hash-contract-v1"
REQUIRED_EXPECTED_HASH_GROUPS = (
    *(Path(filename).stem for filename in REQUIRED_JSONL),
    "runtime_events",
    "strategy_configs_manifest",
    *(f"{dirname}_manifest" for dirname in REQUIRED_DIRS),
    "end_of_day_positions",
    "kis_resource_plan",
    "market_bars_5m",
)
ARTIFACT_EVIDENCE_REQUIREMENTS = {
    "KALCB": (("daily_snapshots", "daily_finalized_candidate"),),
    "OLR": (
        ("olr_stage1_snapshots", "stage1_daily_candidate"),
        ("olr_final_snapshots", "final_afternoon_1430"),
    ),
}
STREAM_VOLATILE_HASH_KEYS = {
    "oms_intents": {
        "idempotency_key",
        "intent_id",
        "timestamp",
        "record_type",
        "dry_run",
        "submitted_to_broker",
        "intended_broker_submit",
        "broker_submit_possible",
        "actually_submitted_to_broker",
        "oms_status",
        "broker_order_id",
    },
    "order_events": {
        "intent_id",
        "order_id",
        "broker_order_id",
        "oms_received_at",
        "order_submitted_at",
        "record_type",
        "dry_run",
        "submitted_to_broker",
        "intended_broker_submit",
        "actually_submitted_to_broker",
    },
    "fill_events": {"order_id", "broker_order_id", "execution_id"},
}


@dataclass(frozen=True, slots=True)
class SessionPaths:
    root: Path
    trade_date: date

    @property
    def manifest(self) -> Path:
        return self.root / "session_manifest.json"


class PaperSessionRecorder:
    def __init__(self, root: str | Path, trade_date: date, *, assistant_event_dir: str | Path | None = None, lineage: LineageContext | None = None):
        self.paths = SessionPaths(Path(root), trade_date)
        self._market_bar_rows: dict[str, dict[str, Any]] = {}
        self._market_bars_dirty = False
        self.assistant_exporter: Any | None = None
        self.paths.root.mkdir(parents=True, exist_ok=True)
        for dirname in REQUIRED_DIRS:
            (self.paths.root / dirname).mkdir(parents=True, exist_ok=True)
        for filename in REQUIRED_JSONL:
            (self.paths.root / filename).touch(exist_ok=True)
        self._load_market_bars()
        self._event_sequence = self._load_event_sequence()
        if assistant_event_dir is not None:
            self.enable_assistant_export(assistant_event_dir, lineage=lineage)

    def enable_assistant_export(self, data_dir: str | Path, *, lineage: LineageContext | None = None) -> None:
        if RuntimeAssistantExporter is None:
            self.assistant_exporter = None
            return
        existing = self.assistant_exporter
        existing_writer = getattr(existing, "writer", None)
        if existing_writer is not None and Path(getattr(existing_writer, "data_dir", "")) == Path(data_dir) and lineage is None:
            return
        try:
            self.assistant_exporter = RuntimeAssistantExporter(data_dir, lineage=lineage)
        except Exception:
            self.assistant_exporter = None

    def write_manifest(self, payload: dict[str, Any] | None = None) -> Path:
        self.flush_market_bars()
        manifest = {
            "session_type": "paper_live_capture",
            "trade_date": self.paths.trade_date.isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "promotion_status": "paper_live_parity_pending",
            "paper_trading_approved": False,
            "live_capital_approved": False,
            **dict(payload or {}),
        }
        self.paths.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
        self._export_manifest(manifest)
        return self.paths.manifest

    def close_session(
        self,
        end_of_day_positions: dict[str, Any],
        session_metrics: dict[str, Any] | None = None,
        *,
        closeout_reason: str = "normal_eod",
    ) -> Path:
        existing = _read_json_file(self.paths.manifest)
        self.write_end_of_day_positions(end_of_day_positions)
        missing_files = _missing_required_files(self.paths.root)
        missing_dirs = [name for name in REQUIRED_DIRS if not (self.paths.root / name).is_dir()]
        missing_artifacts = missing_artifact_evidence(self.paths.root, existing.get("strategy_ids"))
        resource_plan_required = existing.get("kis_resource_plan_required") is True or str(existing.get("mode") or "").lower() in {"paper", "live"}
        missing_resource_plan = [KIS_RESOURCE_PLAN_FILE] if resource_plan_required and not (self.paths.root / KIS_RESOURCE_PLAN_FILE).is_file() else []
        sealed = not missing_files and not missing_dirs and not missing_artifacts and not missing_resource_plan
        hashes = session_hashes(self.paths.root)
        missing_hash_groups = [key for key in REQUIRED_EXPECTED_HASH_GROUPS if key not in hashes]
        expected_hashes = {key: hashes[key] for key in REQUIRED_EXPECTED_HASH_GROUPS if key in hashes}
        return self.write_manifest(
            {
                **existing,
                "expected_hashes": expected_hashes,
                "expected_hash_groups": list(REQUIRED_EXPECTED_HASH_GROUPS),
                "expected_hashes_complete": sealed and not missing_hash_groups,
                "hash_contract_version": HASH_CONTRACT_VERSION,
                "hash_contract_status": "sealed" if sealed else "unsealed_failure",
                "hash_contract_sealed_by": "PaperSessionRecorder.close_session",
                "closeout_reason": closeout_reason,
                "closeout_generated_at": datetime.now(timezone.utc).isoformat(),
                "closeout_missing_required_files": missing_files,
                "closeout_missing_required_dirs": missing_dirs,
                "closeout_missing_artifact_evidence": missing_artifacts,
                "closeout_missing_resource_plan": missing_resource_plan,
                "closeout_missing_hash_groups": missing_hash_groups,
                "session_metrics": dict(session_metrics or existing.get("session_metrics") or {}),
            }
        )

    def append_jsonl(self, filename: str, payload: dict[str, Any]) -> None:
        if filename not in REQUIRED_JSONL:
            raise ValueError(f"unsupported session jsonl file {filename!r}")
        with (self.paths.root / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        self._export_stream_row(filename, payload)

    def copy_snapshot(self, source: str | Path, bucket: str) -> Path:
        if bucket not in REQUIRED_DIRS:
            raise ValueError(f"unsupported snapshot bucket {bucket!r}")
        src = Path(source)
        target = self.paths.root / bucket / src.name
        shutil.copy2(src, target)
        return target

    def record_market_bar(self, bar: Any) -> Path:
        row = _market_bar_row(bar)
        key = market_bar_hash(row)
        if key not in self._market_bar_rows:
            self._market_bar_rows[key] = row
            self._market_bars_dirty = True
        return self.paths.root / MARKET_BARS_FILE

    def next_event_sequence(self) -> int:
        self._event_sequence += 1
        return self._event_sequence

    def flush_market_bars(self) -> Path:
        path = self.paths.root / MARKET_BARS_FILE
        if self._market_bar_rows and (self._market_bars_dirty or not path.is_file()):
            self._write_market_bars()
            self._market_bars_dirty = False
        return path

    def write_end_of_day_positions(self, payload: dict[str, Any]) -> Path:
        self.flush_market_bars()
        path = self.paths.root / "end_of_day_positions.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return path

    def write_resource_plan(self, payload: dict[str, Any]) -> Path:
        path = self.paths.root / KIS_RESOURCE_PLAN_FILE
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        self._export_resource_plan(payload)
        return path

    def _export_stream_row(self, filename: str, payload: dict[str, Any]) -> None:
        exporter = self.assistant_exporter
        if exporter is None:
            return
        try:
            exporter.export_stream_row(filename, payload, session_root=self.paths.root, trade_date=self.paths.trade_date)
        except Exception:
            pass

    def _export_manifest(self, manifest: dict[str, Any]) -> None:
        exporter = self.assistant_exporter
        if exporter is None:
            return
        try:
            exporter.export_manifest(manifest, session_root=self.paths.root, trade_date=self.paths.trade_date)
        except Exception:
            pass

    def _export_resource_plan(self, payload: dict[str, Any]) -> None:
        exporter = self.assistant_exporter
        if exporter is None:
            return
        try:
            exporter.export_resource_plan(payload, session_root=self.paths.root, trade_date=self.paths.trade_date)
        except Exception:
            pass

    def _load_market_bars(self) -> None:
        path = self.paths.root / MARKET_BARS_FILE
        if not path.is_file():
            return
        for row in _read_market_bar_rows(path):
            self._market_bar_rows[market_bar_hash(row)] = row

    def _load_event_sequence(self) -> int:
        value = 0
        for row in _read_jsonl(self.paths.root / "decision_stream.jsonl"):
            if str(row.get("record_type") or "") != "runtime_event_input":
                continue
            try:
                value = max(value, int(row.get("event_sequence") or 0))
            except (TypeError, ValueError):
                continue
        return value

    def _write_market_bars(self) -> None:
        rows = sorted(self._market_bar_rows.values(), key=lambda row: (row["timestamp"], row["symbol"], row["timeframe"]))
        _write_market_bar_rows(self.paths.root / MARKET_BARS_FILE, rows)


def session_hashes(session_root: str | Path) -> dict[str, str]:
    root = Path(session_root)
    manifest = _read_json_file(root / "session_manifest.json")
    hashes = {"session_manifest": json_file_hash(root / "session_manifest.json")}
    for filename in REQUIRED_JSONL:
        stem = Path(filename).stem
        hashes[stem] = jsonl_file_hash(root / filename, exclude_keys=STREAM_VOLATILE_HASH_KEYS.get(stem))
    hashes["runtime_events"] = canonical_json_hash(runtime_event_rows(root))
    hashes["strategy_configs_manifest"] = canonical_json_hash(
        manifest.get("strategy_configs") or manifest.get("captured_strategy_configs") or {}
    )
    for dirname in REQUIRED_DIRS:
        hashes[f"{dirname}_manifest"] = canonical_json_hash(snapshot_directory_manifest(root / dirname, dirname))
    eod = root / "end_of_day_positions.json"
    hashes["end_of_day_positions"] = json_file_hash(eod) if eod.exists() else canonical_json_hash({})
    resource_plan = root / KIS_RESOURCE_PLAN_FILE
    hashes["kis_resource_plan"] = json_file_hash(resource_plan) if resource_plan.exists() else canonical_json_hash(None)
    bars = root / MARKET_BARS_FILE
    hashes["market_bars_5m"] = file_sha256(bars) if bars.exists() else canonical_json_hash(None)
    return hashes


def runtime_event_rows(session_root: str | Path) -> list[dict[str, Any]]:
    rows = [
        row
        for row in _read_jsonl(Path(session_root) / "decision_stream.jsonl")
        if str(row.get("record_type") or "") == "runtime_event_input"
    ]
    return sorted(rows, key=lambda row: (int(row.get("event_sequence") or 0), str(row.get("event_ref") or "")))


def market_bar_hash(bar: Any) -> str:
    return canonical_json_hash(_market_bar_row(bar))


def snapshot_directory_manifest(directory: str | Path, bucket: str) -> dict[str, Any]:
    root = Path(directory)
    files: list[dict[str, Any]] = []
    if root.is_dir():
        for path in sorted(item for item in root.iterdir() if item.is_file()):
            row = {
                "name": path.name,
                "sha256": file_sha256(path),
                "artifact_hash": "",
                "strategy_id": "",
                "artifact_stage": "",
                "source_fingerprint": "",
                "candidate_count": 0,
            }
            if path.suffix.lower() == ".json":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8") or "{}")
                except json.JSONDecodeError:
                    payload = {}
                row["artifact_hash"] = str(payload.get("artifact_hash") or "")
                row["strategy_id"] = str(payload.get("strategy_id") or "")
                row["artifact_stage"] = str((payload.get("metadata") or {}).get("artifact_stage") or "")
                row["source_fingerprint"] = str(payload.get("source_fingerprint") or "")
                row["candidate_count"] = len(payload.get("candidates") or ())
            files.append(row)
    return {"bucket": bucket, "files": files}


def missing_artifact_evidence(session_root: str | Path, strategy_ids: Any = None) -> list[str]:
    root = Path(session_root)
    sids = _strategy_ids(strategy_ids)
    if not sids:
        return ["session_manifest:strategy_ids"]
    artifact_rows = _read_jsonl(root / "artifact_generation.jsonl")
    missing: list[str] = []
    for sid in sids:
        for bucket, stage in ARTIFACT_EVIDENCE_REQUIREMENTS.get(sid, ()):
            snapshot_row = _snapshot_bucket_row(root / bucket, sid, stage)
            if snapshot_row is None:
                missing.append(f"{bucket}:{sid}:{stage}")
            if not any(_artifact_row_matches(row, sid, bucket, stage, snapshot_row) for row in artifact_rows):
                missing.append(f"artifact_generation:{sid}:{stage}")
    return sorted(dict.fromkeys(missing))


def _read_market_bar_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to read canonical market-bar replay inputs") from exc
    return [_market_bar_row(row) for row in pd.read_parquet(path).to_dict("records")]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        return {}
    return json.loads(path.read_text(encoding="utf-8") or "{}")


def _missing_required_files(root: Path) -> list[str]:
    files = [*REQUIRED_JSONL, MARKET_BARS_FILE, "end_of_day_positions.json", "session_manifest.json"]
    return sorted(name for name in files if not (root / name).is_file())


def _strategy_ids(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        values = (raw,)
    else:
        try:
            values = tuple(raw)
        except TypeError:
            values = (raw,)
    normalized = []
    for value in values:
        sid = str(value or "").upper().strip()
        if sid in ARTIFACT_EVIDENCE_REQUIREMENTS:
            normalized.append(sid)
    return tuple(dict.fromkeys(normalized))


def _snapshot_bucket_row(directory: Path, strategy_id: str, stage: str) -> dict[str, Any] | None:
    manifest = snapshot_directory_manifest(directory, directory.name)
    for row in manifest["files"]:
        if row.get("strategy_id") == strategy_id and row.get("artifact_stage") == stage and row.get("artifact_hash"):
            return row
    return None


def _artifact_row_matches(row: Mapping[str, Any], strategy_id: str, bucket: str, stage: str, snapshot_row: Mapping[str, Any] | None) -> bool:
    if str(row.get("strategy_id") or "").upper().strip() != strategy_id:
        return False
    if str(row.get("stage") or row.get("artifact_stage") or "") != stage:
        return False
    if str(row.get("bucket") or "") != bucket:
        return False
    if not bool(row.get("artifact_hash")) or not bool(row.get("source_fingerprint")) or "candidate_count" not in row:
        return False
    if snapshot_row is None:
        return True
    return (
        str(row.get("artifact_hash") or "") == str(snapshot_row.get("artifact_hash") or "")
        and str(row.get("source_fingerprint") or "") == str(snapshot_row.get("source_fingerprint") or "")
        and int(row.get("candidate_count") or 0) == int(snapshot_row.get("candidate_count") or 0)
    )


def _write_market_bar_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to write canonical market-bar replay inputs") from exc
    pd.DataFrame(rows).to_parquet(path, index=False)


def _market_bar_row(bar: Any) -> dict[str, Any]:
    payload = bar.to_json_dict() if callable(getattr(bar, "to_json_dict", None)) else dict(bar or {})
    timestamp = _json_value(payload.get("timestamp"))
    if isinstance(timestamp, str):
        timestamp_value = timestamp
    else:
        isoformat = getattr(timestamp, "isoformat", None)
        timestamp_value = isoformat() if callable(isoformat) else str(timestamp or "")
    metadata = payload.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata) if metadata.strip() else {}
        except json.JSONDecodeError:
            metadata = {}
    if isinstance(metadata, str):
        metadata_value = "{}"
    else:
        metadata_value = json.dumps(_json_value(metadata), sort_keys=True, separators=(",", ":"))
    return {
        "symbol": str(payload.get("symbol") or payload.get("ticker") or "").zfill(6),
        "timestamp": timestamp_value,
        "timeframe": str(payload.get("timeframe") or "5m"),
        "open": float(payload.get("open")),
        "high": float(payload.get("high")),
        "low": float(payload.get("low")),
        "close": float(payload.get("close")),
        "volume": float(payload.get("volume") or 0.0),
        "is_completed": bool(payload.get("is_completed", True)),
        "source": "" if payload.get("source") is None else str(payload.get("source") or ""),
        "source_fingerprint": str(payload.get("source_fingerprint") or ""),
        "metadata": metadata_value,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return value

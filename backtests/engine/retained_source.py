"""Content-bound acceptance of retained Parquet inputs for offline research.

This checks the declared archive convention and cross-frequency price scale.
It does not certify broker share units, invent missing prints, or equate daily
and regular-session volume. Unreconciled sessions retain their own disposition.
"""
from collections import defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path

from kis_core.input_manifest import signature

VERSION = "retained-parquet-acceptance-v1"


def build_evidence(root, contract, destination):
    import pandas as pd
    import numpy as np
    from strategy_common.daily_lrs_parquet import load_daily_ohlcv
    root, destination = Path(root).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    daily_root = root / contract["daily_data_root"]
    daily_rows, basis_failures, price_rounding = [], [], []
    dates, checked = set(), 0
    files = {name: digest for name, digest in contract["files"].items()
             if name.endswith(".parquet")}
    for symbol, names in sorted(contract["intraday_files"].items()):
        daily = load_daily_ohlcv(daily_root, symbol, end=date.fromisoformat(contract["data_end"]))
        daily_by_date = {}
        for row in daily.to_dict("records"):
            day = str(row["date"])[:10]
            value = {"symbol": symbol, "date": day, **{k: float(row[k]) for k in ("open", "high", "low", "close", "volume")}}
            daily_by_date[day] = value
            daily_rows.append(value)
        seen = set()
        for name in names:
            frame = pd.read_parquet(root / name)
            if frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing:
                raise ValueError("Unordered or duplicate retained bars: " + name)
            for day, group in frame.groupby(frame.timestamp.dt.date):
                key = (symbol, day.isoformat())
                stamps = set(group.timestamp)
                if seen & stamps:
                    raise ValueError("Overlapping retained input files: " + name)
                seen.update(stamps)
                dates.add(key)
                checked += len(group)
                if day.isoformat() > contract["data_end"]:
                    continue
                source = daily_by_date.get(day.isoformat())
                reason = None
                numeric = group[["open", "high", "low", "close", "volume"]]
                if (not np.isfinite(numeric.to_numpy()).all() or (numeric[["open", "high", "low", "close"]] <= 0).any().any()
                        or (group.volume < 0).any() or (group.volume % 1 != 0).any()
                        or (group.low > group[["open", "close", "high"]].min(axis=1)).any()
                        or (group.high < group[["open", "close", "low"]].max(axis=1)).any()):
                    reason = "invalid_price_or_interval_shares"
                elif source is None:
                    reason = "daily_price_control_unavailable"
                else:
                    excess = max(source["low"] - float(group.low.min()), float(group.high.max()) - source["high"], 0)
                    if excess > 2:
                        reason = "intraday_outside_daily_price_range"
                    elif excess:
                        price_rounding.append({"symbol": symbol, "date": day.isoformat(), "excess_krw": excess,
                            "relative_to_daily_low": excess / source["low"]})
                if reason:
                    basis_failures.append({"symbol": symbol, "date": day.isoformat(), "reason": reason})
    def write(name, payload):
        path = destination / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path.relative_to(root).as_posix()
    parser_paths = ["scripts/update_kis_intraday.py", "strategy_common/daily_lrs_parquet.py",
                    "backtests/core/anchored_recovery.py", "backtests/core/session_replacement.py"]
    report = {"version": VERSION, "scope": "retained_archive_offline_research",
        "market_input_hash": contract["market_input_hash"], "files": files,
        "parser_files": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in parser_paths},
        "checked_bars": checked, "checked_symbol_sessions": len(dates),
        "basis_failures": basis_failures,
        "retained_price_discrepancies": price_rounding,
        "archive_acceptance_policy": "Preserve both archived series; allow at most 2 KRW range disagreement for integer price quantization. This is a declared offline acceptance tolerance, not proof of its vendor cause. Never synthesize or alter observations.",
        "clock": "archive Asia/Seoul interval-start labels; delayed auctions retain source match time",
        "volume": "recorded interval shares; no allocation of daily residuals",
        "adjustments": "preserved archive basis; source-bound recovered observations checked during composition",
        "limitations": ["not vendor-wide convention certification", "not live share-unit certification",
                        "daily range agreement does not establish session completeness"]}
    report_path = write("acceptance.json", report)
    daily_path = write("daily_references.json", daily_rows)
    snapshot_end = contract.get('daily_snapshot_end') or contract['data_end'].replace('-','')
    as_of = date.fromisoformat(snapshot_end).isoformat()
    normalized = {"id": "retained-archive-original-price-share-basis-"+snapshot_end.replace('-',''),
                  "price_unit": "KRW", "volume_unit": "shares", "adjustment_as_of": as_of}
    common = {"provider": "retained_KIS_and_KRX_archive", "source_url": "repository:retained-source-receipts",
        "venue": "KRX", "parser_version": VERSION, "price_unit": "KRW",
        "normalization": {"source_basis_id": normalized["id"], "target_basis": normalized, "transforms": []}}
    market_doc = write("market_contract.json", {**common, "kind": "market_basis_contract_v1",
        "raw_paths": [n for names in contract["intraday_files"].values() for n in names],
        "volume_unit": "shares", "volume_kind": "interval", "timezone": "Asia/Seoul",
        "timestamp_convention": "completed_interval_start", "acceptance_report": report_path})
    daily_doc = write("daily_contract.json", {**common, "kind": "daily_reference_contract_v1",
        "raw_paths": [daily_path], "date_convention": "KRX_session_date", "acceptance_report": report_path})
    sources = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in destination.glob("*.json")}
    sources.update({p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in parser_paths})
    evidence = {"version": "qualification-observations-v2", "source_files": sources,
        "basis": {"representation": VERSION, "source_documents": [market_doc], "acceptance_report": report_path},
        "references": {"representation": VERSION, "source_documents": [daily_doc], "daily_rows": daily_path,
                       "acceptance_report": report_path}}
    return root / write("qualification.json", evidence)


def verify_acceptance(read, evidence, *, generation=None):
    report = read(evidence["basis"]["acceptance_report"])
    if report.get("version") != VERSION or report.get("scope") != "retained_archive_offline_research":
        raise ValueError("invalid_retained_source_acceptance")
    if generation is None:
        from kis_core.input_manifest import active_manifest
        generation = active_manifest()
    if generation is None:
        raise ValueError("retained_acceptance_requires_frozen_inputs")
    from backtests.core.shared_market_inputs import market_input_contract
    contract = market_input_contract(generation)
    if not contract or report["market_input_hash"] != contract["market_input_hash"]:
        raise ValueError("retained_acceptance_market_identity_mismatch")
    if report["files"] != {p:digest for p,digest in contract["files"].items() if p.endswith(".parquet")}:
        raise ValueError("retained_acceptance_dependency_scope_mismatch")
    for name, digest in {**report["files"], **report["parser_files"]}.items():
        if generation.payload["files"].get(name) != digest:
            raise ValueError("retained_acceptance_dependency_mismatch")
    return report


def sparse_dispositions(read, evidence):
    path = evidence.get("session_resolution_report")
    if not path:
        return {}
    report = read(path)
    if report.get("version") != "session-recovery-dispositions-v1" or not report.get("all_original_flags_accounted"):
        raise ValueError("invalid_session_dispositions")
    # A provider-unavailable or incompatible session is not made complete here.
    allowed = {"archived_sparse_daily_reconciled", "archived_sparse_daily_reconciled_provider_unavailable",
               "archived_sparse_daily_close_residual_reconciled"}
    def accepted(row):
        if row.get("disposition") in allowed:
            return True
        comparison = row.get("recovered_daily_comparison") or {}
        return (row.get("disposition") == "recovered_for_shared_input"
            and row.get("coverage_basis") == "daily_ohlcv_conservation"
            and comparison.get("status") == "exact_daily_ohlcv_match"
            and bool(comparison.get("intraday"))
            and comparison.get("intraday") == comparison.get("daily")
            and comparison.get("mismatched_fields") == [])
    return {(r["symbol"], date.fromisoformat(r["date"])): r["disposition"] for r in report["sessions"] if accepted(r)}

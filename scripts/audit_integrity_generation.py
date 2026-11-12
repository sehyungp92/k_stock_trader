"""Verify a frozen generation and inventory every required symbol/session."""
import argparse
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml
from backtests.auto.shared.cache_keys import source_generation_hash
from backtests.core.input_manifest import InputManifest
from backtests.core.shared_market_inputs import market_input_contract, selected_intraday_paths
from kis_core.trading_calendar import get_trading_calendar
from strategy_common.session_coverage import session_bar_coverage


def audit_generation(path):
    generation = InputManifest(path)
    changed = []
    for name, expected in generation.payload["files"].items():
        actual = hashlib.sha256((generation.root / name).read_bytes()).hexdigest()
        if actual != expected:
            changed.append(name)
    if changed:
        raise ValueError(f"Frozen files changed: {changed}")
    if generation.payload["source_generation_hash"] != source_generation_hash():
        raise ValueError("Executing source differs from frozen generation")
    with generation.activate():
        contract = market_input_contract(generation)
        calendar = get_trading_calendar()
        start, end = date.fromisoformat(contract["train_start"]), date.fromisoformat(contract["data_end"])
        days = [start + timedelta(days=i) for i in range((end-start).days + 1)
                if calendar.is_trading_day(start + timedelta(days=i))]
        symbols = [str(s).zfill(6) for s in yaml.safe_load(generation.read(generation.root / contract["universe"]))["symbols"]]
        gaps, counts, bars, exceptions = [], Counter(), 0, []
        for symbol in symbols:
            stamps = []
            for selected in selected_intraday_paths(generation.root / contract["data_root"], symbol, contract["timeframe"]):
                with generation.open_verified(selected) as stream:
                    frame = pd.read_parquet(stream, columns=["timestamp"])
                stamps.extend(frame.timestamp)
            if len(stamps) != len(set(stamps)):
                raise ValueError(f"Duplicate selected timestamps: {symbol}")
            grouped = {}
            for stamp in stamps:
                if start <= stamp.date() <= end:
                    grouped.setdefault(stamp.date(), []).append(stamp)
                    bars += 1
            for day in days:
                coverage = session_bar_coverage(grouped.get(day, []), day, 5, symbol=symbol)
                if coverage.get("reason") == "sourced_symbol_suspension":
                    exceptions.append({"symbol": symbol, **coverage})
                if coverage["status"] != "pass":
                    counts[day.isoformat()] += 1
                    gaps.append({"symbol": symbol, "date": day.isoformat(), "observed_rows": len(grouped.get(day, [])), **coverage})
        proof = json.loads(generation.read(generation.root / generation.payload["evidence"]["qualification_path"]))
        resolution = json.loads(generation.read(generation.root / proof["session_resolution_report"])) if proof.get("session_resolution_report") else {}
        by_session = {(row["symbol"], row["date"]): row for row in resolution.get("sessions", [])}
        for gap in gaps:
            row = by_session.get((gap["symbol"], gap["date"]), {})
            gap["data_disposition"] = row.get("disposition", "not_resolved")
            gap["provider_availability"] = row.get("provider_availability", "not_queried")
        missing_evidence = {name: proof[name].get("unavailable_reason") for name in ("references", "basis", "capacity")
                            if proof[name].get("unavailable_reason")}
    return {"manifest_hash": generation.identity, "source_generation_hash": generation.payload["source_generation_hash"],
        "market_input_hash": contract["market_input_hash"], "verified_files": len(generation.files),
        "source_matches": True, "symbol_sessions_checked": len(symbols) * len(days), "selected_bars": bars,
        "symbols": len(symbols), "sessions": len(days), "window": [start.isoformat(), end.isoformat()],
        "applied_replacement_sessions": len(contract.get("session_replacements", [])),
        "unverified_symbol_sessions": len(gaps), "zero_row_sessions": sum(r["observed_rows"] == 0 for r in gaps),
        "recovery_dispositions": resolution.get("counts", {}),
        "remaining_inventory_classification": dict(Counter(r["data_disposition"] for r in gaps)),
        "affected_dates": dict(sorted(counts.items())), "gaps": gaps, "unavailable_evidence": missing_evidence,
        "sourced_no_trading_sessions": exceptions,
        "source_coverage_complete": not gaps,
        "historical_qualification": False,
        "qualification_scope": "file_identity_and_independent_full_window_census_only",
        "interpretation": "Nominal grid gaps are classified separately as provider-unavailable, sparse corroborated, or source-incompatible. None alone proves corrupt data or invalid trades."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path to preserve verification evidence")
    report = audit_generation(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"gaps", "affected_dates"}}, indent=2))


if __name__ == "__main__":
    main()

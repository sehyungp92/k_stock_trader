"""Prepare verified, archive-basis recoveries from retained KIS basis evidence."""
import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import yaml

from backtests.core.adjusted_recovery import reconstruct_adjusted_session
from backtests.core.anchored_recovery import verified_reconstruction
from scripts.reconcile_integrity_sessions import compare_session
from kis_core.trading_calendar import get_trading_calendar


def resolve(evidence, previous_report, previous_bundle, destination):
    if destination.exists():
        raise ValueError("Use a new destination; preserve earlier recovery evidence")
    destination.mkdir(parents=True)
    report = json.loads(previous_report.read_text(encoding="utf-8"))
    bundle = json.loads(previous_bundle.read_text(encoding="utf-8"))
    calendar_path = ROOT / "kis_core/data/krx_intraday_halts.yaml"
    regimes = yaml.safe_load(calendar_path.read_text(encoding="utf-8"))
    closing_events = {(r["symbol"], r["date"]): r for r in regimes.get("closing_auctions", [])}
    calendar = get_trading_calendar()
    recovered, failures, cache = [], [], {}

    def relative(path):
        return path.resolve().relative_to(ROOT).as_posix()

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    for item in report["sessions"]:
        if item["disposition"] != "returned_source_incompatible_with_archive":
            continue
        symbol, day = item["symbol"], date.fromisoformat(item["date"])
        if symbol not in cache:
            anchor_paths = list((ROOT / "data/kis_intraday_parquet" / symbol).glob(f"{symbol}_1m_*_20260512.parquet"))
            daily_paths = list((ROOT / "data/krx_daily_parquet/daily_ohlcv" / symbol).glob("*.parquet"))
            if len(anchor_paths) != 1 or len(daily_paths) != 1:
                raise ValueError("Ambiguous preserved input source")
            cache[symbol] = anchor_paths[0], pd.read_parquet(anchor_paths[0]), daily_paths[0], pd.read_parquet(daily_paths[0])
        anchor_path, anchor, daily_path, daily = cache[symbol]
        raw_path = ROOT / item["raw_source"]
        pages = sorted((raw_path.parent / "provider_responses").rglob("*.json"))
        original_path = evidence / "kis_daily" / f"{symbol}_{day.isoformat()}_original.json"
        adjusted_path = evidence / "kis_daily" / f"{symbol}_{day.isoformat()}_adjusted.json"
        row = {"symbol": symbol, "date": day.isoformat(), "method": "adjusted_provider_reconstruction",
               "anchor": relative(anchor_path), "raw": relative(raw_path), "daily_source": relative(daily_path),
               "original_daily_receipt": relative(original_path), "adjusted_daily_receipt": relative(adjusted_path),
               "provider_responses": [relative(p) for p in pages]}
        row["files"] = {relative(p): digest(p) for p in [anchor_path, raw_path, daily_path, original_path, adjusted_path, *pages]}
        try:
            reference = daily[daily.date.astype(str).str[:10] == day.isoformat()]
            if len(reference) != 1:
                raise ValueError("Absent or ambiguous frozen daily reference")
            combined, proof = reconstruct_adjusted_session(pd.read_parquet(raw_path), anchor, day, symbol,
                json.loads(original_path.read_text(encoding="utf-8")), json.loads(adjusted_path.read_text(encoding="utf-8")),
                reference.iloc[0].to_dict(), [v for p in pages for v in json.loads(p.read_text(encoding="utf-8"))["rows"]])
            closing = pd.Timestamp(proof["closing_auction"]["timestamp"]).time()
            event_path = destination / "closing_events" / f"{symbol}_{day.isoformat()}.json"
            event_path.parent.mkdir(parents=True, exist_ok=True)
            event = {"symbol": symbol, "date": day.isoformat(), "matching_time": closing.isoformat()}
            event_path.write_text(json.dumps({"closing_event": event, "observation": proof["closing_auction"],
                "reconstruction": proof, "source_files": row["files"]}, indent=2), encoding="utf-8")
            # The final raw print and exact turnover determine the sourced event.
            if closing > calendar.session_hours(day)[1]:
                calendar.closing_auctions[(symbol, day)] = closing
            _, replacement, verified, _ = verified_reconstruction(ROOT, row)
            if verified["coverage_basis"] != "complete_matching_grid":
                raise ValueError("Adjusted recovery has an unresolved regular-session grid")
            if closing > calendar.session_hours(day)[1]:
                regimes["evidence_files"][relative(event_path)] = digest(event_path)
                closing_events[(symbol, day.isoformat())] = {**event, "sources": [relative(event_path)]}
            bundle["sessions"].append(row)
            item.update(disposition="recovered_for_shared_input", previous_disposition="returned_source_incompatible_with_archive",
                        reconstruction=proof, coverage_basis=verified["coverage_basis"],
                        recovered_daily_comparison=compare_session(combined, reference.iloc[0].to_dict()))
            item.pop("reconstruction_error", None)
            recovered.append({"symbol": symbol, "date": day.isoformat(), "proof": proof,
                              "five_minute_rows": len(replacement), "sources": row["files"]})
            print(json.dumps({"symbol": symbol, "date": day.isoformat(), "added_minutes": proof["added_rows"],
                              "status": "verified"}), flush=True)
        except (ValueError, AssertionError) as exc:
            failures.append({"symbol": symbol, "date": day.isoformat(), "reason": str(exc)})
            print(json.dumps(failures[-1]), flush=True)
    regimes["closing_auctions"] = [closing_events[k] for k in sorted(closing_events)]
    calendar_path.write_text(yaml.safe_dump(regimes, sort_keys=False, allow_unicode=True), encoding="utf-8")
    report["counts"] = dict(Counter(r["disposition"] for r in report["sessions"]))
    report["source_files"].update({relative(previous_report): digest(previous_report),
                                  relative(evidence / "sources.json"): digest(evidence / "sources.json")})
    source_inventory = json.loads((evidence / "sources.json").read_text(encoding="utf-8"))
    for name, expected in source_inventory["files"].items():
        source = (evidence / name).resolve()
        if not source.is_relative_to(evidence.resolve()) or digest(source) != expected:
            raise ValueError("Basis evidence source identity mismatch")
        report["source_files"][relative(source)] = expected
    result = {"version": "adjusted-session-recovery-v1", "historical_qualification": False,
              "resolved": len(recovered), "unresolved": len(failures), "sessions": recovered, "failures": failures,
              "prior_sessions_preserved": len(bundle["sessions"]) - len(recovered)}
    for name, value in (("resolution.json", result), ("report.json", report), ("bundle.json", bundle)):
        (destination / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in {"sessions", "failures"}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--previous-report", type=Path, default=ROOT / "data/backtests/integrity/session-recoveries-20261004-v10/report.json")
    parser.add_argument("--previous-bundle", type=Path, default=ROOT / "data/backtests/integrity/session-recoveries-20261004-v10/bundle.json")
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    resolve(args.evidence, args.previous_report, args.previous_bundle, args.destination)

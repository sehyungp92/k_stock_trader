"""Recheck retained residual evidence and prepare the single sourced auction repair."""
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
from backtests.core.daily_volume_residual import DAILY_FIELDS, integer, verify_daily_close_residual
from backtests.core.anchored_recovery import verified_reconstruction
from scripts.recover_flagged_sessions import reconstruct_session


def resolve(evidence, prior):
    evidence, prior = evidence.resolve(), prior.resolve()
    out = evidence / "recovery"
    if out.exists():
        raise ValueError("Use a new recovery destination; preserve earlier evidence")
    out.mkdir()
    report = json.loads((prior / "report.json").read_text(encoding="utf-8"))
    bundle = json.loads((prior / "bundle.json").read_text(encoding="utf-8"))
    inventory = json.loads((evidence / "inventory.json").read_text(encoding="utf-8"))["sessions"]
    inventory_by_key = {(r["symbol"], r["date"]): r for r in report["sessions"]}
    sources, results, frames = {}, [], {}

    def bind(path):
        path = path.resolve()
        name = path.relative_to(ROOT).as_posix()
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return name

    bind(prior / "report.json")
    bind(prior / "bundle.json")
    for item in inventory:
        symbol, day = item["symbol"], date.fromisoformat(item["date"])
        if symbol not in frames:
            frames.clear()
            anchor_paths = list((ROOT / "data/kis_intraday_parquet" / symbol).glob("*_1m_*_20260512.parquet"))
            daily_paths = list((ROOT / "data/krx_daily_parquet/daily_ohlcv" / symbol).glob("*.parquet"))
            if len(anchor_paths) != 1 or len(daily_paths) != 1:
                raise ValueError("Ambiguous archive or daily reference")
            frames[symbol] = anchor_paths[0], pd.read_parquet(anchor_paths[0]), daily_paths[0], pd.read_parquet(daily_paths[0])
        anchor_path, anchor, daily_path, daily = frames[symbol]
        bind(anchor_path)
        bind(daily_path)
        reference = daily[daily.date.astype(str).str[:10] == str(day)].iloc[0].to_dict()
        daily_receipt = evidence / "kis_daily" / f"{symbol}_{day}_original.json"
        receipt = json.loads(daily_receipt.read_text(encoding="utf-8"))
        control = [r for r in receipt["output2"] if r.get("stck_bsop_date") == day.strftime("%Y%m%d")]
        if receipt["status"] != "ok" or len(control) != 1:
            raise ValueError("Daily quotation control unavailable")
        original_match = all(integer(reference[k]) == integer(control[0][v]) for k, v in DAILY_FIELDS.items())
        nav = json.loads((evidence / "pykrx" / f"{symbol}_adjusted.json").read_text(encoding="utf-8"))
        nav_rows = [r for r in nav["rows"]["data"] if str(r["\ub0a0\uc9dc"])[:10] == str(day)]
        names = dict(open="\uc2dc\uac00", high="\uace0\uac00", low="\uc800\uac00", close="\uc885\uac00", volume="\uac70\ub798\ub7c9")
        naver_match = len(nav_rows) == 1 and all(integer(reference[k]) == integer(nav_rows[0][v]) for k, v in names.items())
        outcome = {"symbol": symbol, "date": str(day), "original_kis_daily_matches_reference": original_match,
            "current_pykrx_naver_matches_reference": naver_match, "historical_qualification": False,
            "status": "daily_control_confirmed_regular_phase_unresolved" if original_match else "changed_adjustment_basis_unresolved"}
        folder = evidence / "full_minutes" / symbol / str(day)
        if folder.exists():
            query = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
            if query["status"] != "ok" or query["pages"] >= 8:
                raise ValueError("Full-session query did not complete")
            for name, digest in query["files"].items():
                if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                    raise ValueError("Minute collection receipt changed")
            raw = pd.read_parquet(folder / "raw_1m.parquet")
            combined, reconstruction = reconstruct_session(raw, anchor, day)
            pages = sorted((folder / "provider_responses").rglob("*.json"))
            provider_rows = []
            for page in pages:
                payload = json.loads(page.read_text(encoding="utf-8"))
                if payload["symbol"] != symbol or payload["market"] != "J" or payload["endpoint"] != "inquire-time-dailychartprice":
                    raise ValueError("Unexpected minute source")
                provider_rows.extend(payload["rows"])
            outcome["reconstruction"] = reconstruction
            last = max((r for r in provider_rows if r["stck_bsop_date"] == day.strftime("%Y%m%d")), key=lambda r:r["stck_cntg_hour"])
            outcome["daily_excess_shares"] = integer(reference["volume"]) - integer(combined.volume.sum())
            outcome["daily_excess_turnover"] = integer(control[0]["acml_tr_pbmn"]) - integer(last["acml_tr_pbmn"])
            try:
                proof = verify_daily_close_residual(combined, reference, receipt, provider_rows, day, symbol)
            except (ValueError, AssertionError) as exc:
                outcome.update(status="repeated_source_residual_unresolved", reason=str(exc))
            else:
                outcome.update(status=proof["status"], proof=proof)
                current = inventory_by_key[(symbol, str(day))]
                current["daily_residual_resolution"] = proof
                if reconstruction["added_rows"]:
                    paths = [anchor_path, daily_path, daily_receipt, folder / "raw_1m.parquet", *pages]
                    row = dict(symbol=symbol, date=str(day), method="verified_sparse_closing_auction",
                        anchor=bind(anchor_path), raw=bind(folder / "raw_1m.parquet"), daily_source=bind(daily_path),
                        original_daily_receipt=bind(daily_receipt), provider_responses=[bind(p) for p in pages],
                        files={bind(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
                    old, new, checked, _ = verified_reconstruction(ROOT, row)
                    if any((r["symbol"], r["date"]) == (symbol, str(day)) for r in bundle["sessions"]):
                        raise ValueError("Recovery already included")
                    bundle["sessions"].append(row)
                    outcome["admitted_added_volume"] = integer(new.volume.sum()) - integer(old.volume.sum())
                    current.update(disposition="recovered_for_shared_input", coverage_basis=checked["coverage_basis"])
                else:
                    current["disposition"] = "archived_sparse_daily_close_residual_reconciled"
        inventory_by_key[(symbol, str(day))]["residual_investigation"] = outcome
        results.append(outcome)
    for path in evidence.rglob("*"):
        if path.is_file() and not path.is_relative_to(out) and not path.is_relative_to(evidence / "test-temp"):
            bind(path)
    report["source_files"].update(sources)
    report["counts"] = dict(Counter(r["disposition"] for r in report["sessions"]))
    report["all_original_flags_accounted"] = len(report["sessions"]) == 365
    summary = dict(version="daily-volume-residual-resolution-v1", historical_qualification=False,
        counts=dict(Counter(r["status"] for r in results)), sessions=results, source_files=sources,
        admitted_recovery_sessions=len(bundle["sessions"]),
        interpretation="Exact residual arithmetic is distinct from independently observed after-hours prints or complete intraday coverage.")
    for name, value in [("report.json", report), ("bundle.json", bundle), ("resolution.json", summary)]:
        (out / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps({"outcomes": summary["counts"], "recoveries": len(bundle["sessions"]), "inventory": report["counts"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--prior-recovery", required=True, type=Path)
    args = parser.parse_args()
    resolve(args.evidence, args.prior_recovery)

"""Resume the sealed portfolio validation using an immutable repaired generation."""
from __future__ import annotations
import argparse
import copy
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.portfolio_source_repair import ROOT, PARENT
from scripts.portfolio_targeted_validation import read, bound
from backtests.strategies.kalcb.executed_value import digest, signature, write_json


def verify(item):
    if digest(item["path"]) != item["sha256"]:
        raise ValueError("Changed source dependency: " + item["path"])


def flows(root):
    import pandas as pd
    from scripts.prepare_olr_forward_daily import fetch_flow
    from scripts.update_kis_intraday import _load_dotenv
    from kis_core import KoreaInvestAPI, KoreaInvestEnv, build_kis_config_from_env
    from loguru import logger
    _load_dotenv(REPO / ".env"); logger.remove()
    api = KoreaInvestAPI(KoreaInvestEnv(build_kis_config_from_env()))
    path = next((PARENT / "forward_inputs/files/data/krx_daily_parquet/daily_flow/068270").glob("*.parquet"))
    old = pd.read_parquet(path); old = old[(old.date >= "2026-05-21") & (old.date <= "2026-06-04")]
    folder = root / "flow_sources"; folder.mkdir(exist_ok=False)
    rows = fetch_flow(api, "068270", old.date.tolist(), date(2026, 6, 4), folder)
    current = {x["date"]: x for x in rows}; errors = []
    for row in old.to_dict("records"):
        for field in ("foreign_net", "inst_net"):
            if row[field] != current[row["date"]][field]: errors.append({"date": row["date"], "field": field})
    report = {"pass": not errors, "comparisons": len(old) * 2, "errors": errors,
        "quantity_basis": "original net shares; no chart-factor scaling of investor quantities",
        "inputs": [bound(path), *[bound(p) for p in folder.glob("*.json")]]}
    write_json(root / "flow_basis_audit.json", report)
    print(json.dumps({k: report[k] for k in ("pass", "comparisons", "errors")}), flush=True)
    if errors: raise ValueError("Affected investor flows differ from original quantity controls")


def volumes(root):
    """Keep interval capacity; qualify original bars against fresh monetary totals.

    July 10 daily totals are revised from current original-price controls. They
    enter no decision on/before July 10 (strict prior-day daily features). Their
    revision cause is not inferred; they are never assigned to interval volume.
    """
    import pandas as pd
    from backtests.core.adjusted_recovery import _bounds, DAILY_FIELDS
    from backtests.core.daily_volume_residual import FIELDS
    review = read(PARENT / "source_basis_review.json")
    results = []
    for issue in review["exceptions"]:
        if issue["classification"] != "volume_only_revision": continue
        symbol, day = issue["symbol"], issue["date"]
        folder = root / "minute_sources" / symbol / day
        completion = read(folder / "completion.json")
        for item in [completion["source"], *completion["receipts"]]: verify(item)
        current = pd.read_parquet(folder / "current_1m.parquet")
        archive_path = next((REPO / "data/kis_intraday_parquet" / symbol).glob(symbol + "_1m_*_20260710.parquet"))
        archive = pd.read_parquet(archive_path)
        archive = archive[archive.timestamp.dt.strftime("%Y-%m-%d") == day].reset_index(drop=True)
        if list(archive.timestamp) != list(current.timestamp): raise ValueError("Changed interval population " + symbol)
        if archive[FIELDS].equals(current[FIELDS]):
            convention = {"type": "exact_original_interval_identity", "rows": len(archive)}
        else:
            lower, upper = _bounds(archive.merge(current, on="timestamp", suffixes=("_old", "_new")))
            convention = {"type": "original_archive_with_later_adjusted_provider_vintage", "rows": len(archive),
                          "common_factor_interval": list(map(str, (lower, upper))), "original_archive_preserved": True}
        daily_path = PARENT / "provider_basis_audit" / (symbol + "_original.json")
        daily = next(x for x in read(daily_path)["rows"] if x["stck_bsop_date"] == day.replace("-", ""))
        controls = {k: int(daily[v]) for k, v in DAILY_FIELDS.items()}
        actual = dict(open=archive.open.iloc[0], high=archive.high.max(), low=archive.low.min(), close=archive.close.iloc[-1])
        if any(actual[k] != controls[k] for k in actual): raise ValueError("Daily OHLC mismatch " + symbol)
        money = {}
        for item in completion["receipts"]:
            for row in read(item["path"])["rows"]:
                if row["stck_bsop_date"] != day.replace("-", ""): continue
                clock, total = row["stck_cntg_hour"], int(row["acml_tr_pbmn"])
                if clock in money and money[clock] != total: raise ValueError("Conflicting cumulative turnover")
                money[clock] = total
        previous = 0; opening_baseline = None
        for index, row in enumerate(archive.itertuples()):
            total = money[row.timestamp.strftime("%H%M%S")]; change = total - previous; previous = total
            if not row.low * row.volume <= change <= row.high * row.volume:
                # The first cumulative observation has no observed predecessor.
                # Direct original OHLCV identity remains valid; do not invent a
                # zero pre-session balance or alter interval shares.
                if index != 0 or convention["type"] != "exact_original_interval_identity" or change < row.low * row.volume:
                    raise ValueError("Original minute turnover mismatch " + symbol)
                opening_baseline = {"status": "unobserved_before_first_regular_minute",
                    "feasible_prior_turnover_range": [max(0, int(change-row.high*row.volume)), int(change-row.low*row.volume)],
                    "cause": "not inferred", "interval_shares_from_direct_provider_identity": True}
        residual = controls["volume"] - int(archive.volume.sum())
        residual_money = int(daily["acml_tr_pbmn"]) - previous
        if residual < 0 or residual_money < 0: raise ValueError("Negative daily/session residual")
        results.append({"symbol": symbol, "date": day, "pass": True, "interval_convention": convention,
            "archive_daily_volume": issue["archive"]["volume"], "current_original_daily_volume": controls["volume"],
            "regular_interval_shares": int(archive.volume.sum()), "daily_residual_shares": residual,
            "daily_residual_turnover": residual_money, "residual_added_to_bars": False,
            "opening_cumulative_baseline": opening_baseline,
            "turnover_bounds_verified_intervals": len(archive) - int(opening_baseline is not None),
            "revision_cause": "unverified; observation-vintage difference retained explicitly",
            "decision_use": "Daily observations are strictly prior-day; July 10 has no following decision in this window",
            "inputs": [bound(archive_path), bound(daily_path), bound(folder / "completion.json")]})
    if len(results) != 24: raise ValueError("Incomplete daily volume revision audit")
    write_json(root / "volume_revision_audit.json", {"pass": True, "sessions": results,
        "execution_volume_replaced": False, "future_daily_total_used_intraday": False})
    print(json.dumps({"volume_revisions_qualified_for_scope": len(results)}), flush=True)


def freeze(root):
    """Copy-on-write of the quarantined generation; preserve every old artifact."""
    import pandas as pd
    from backtests.engine.retained_source import build_evidence
    proof = read(root / "repaired_sessions/proof.json")
    if not proof["pass"] or not read(root / "flow_basis_audit.json")["pass"] or not read(root / "volume_revision_audit.json")["pass"]:
        raise ValueError("Incomplete repair verification")
    verify(proof["program"]); verify(proof["verifier"])
    for session in proof["sessions"]:
        for item in [*session["inputs"], *session["outputs"].values()]: verify(item)
    # Reuse the exact registered selection and bridges, never a new search.
    for name in ("protocol.json", "screen.json", "bridge.json", "source_audit.json", "development_selection.json",
                 "frozen_inference_bridge.json", "olr_inference_bridge.json", "forward_program_dependency_review.json",
                 "development/study.json", "battery/study.json"):
        destination = root / name; destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists(): raise FileExistsError(destination)
        shutil.copyfile(PARENT / name, destination)
    parent = read(PARENT / "forward_inputs/manifest.json")
    old = PARENT / "forward_inputs/files"; destination = root / "forward_inputs"
    target = destination / "files"; target.mkdir(parents=True, exist_ok=False)
    files = {}; changed = []; revisions = read(PARENT / "source_basis_review.json")["exceptions"]
    def bind(path, name, copying=False):
        out = target / name; out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists(): raise FileExistsError(name)
        if copying: shutil.copyfile(path, out)
        else:
            try: os.link(path, out)
            except OSError: shutil.copyfile(path, out)
        files[name] = digest(out)
    overlays = {"scripts/update_kis_intraday.py", "backtests/core/anchored_basis_repair.py", "scripts/kalcb_executed_value_study.py",
                "scripts/portfolio_synergy_study.py", "backtests/engine/corporate_action_guard.py"}
    for name, sha in parent["files"].items():
        if digest(old / name) != sha: raise ValueError("Changed prior frozen source: " + name)
        if name.startswith("research/forward_acceptance/") or name in overlays: continue
        altered = False
        if name.startswith("data/kis_intraday_parquet/068270/"):
            frame = pd.read_parquet(old / name); days = {x["date"] for x in proof["sessions"]}
            frame = frame[~frame.timestamp.dt.strftime("%Y-%m-%d").isin(days)]
            frame = pd.concat([frame, *[pd.read_parquet(x["outputs"]["5m"]["path"]) for x in proof["sessions"]]], ignore_index=True)
            frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True); altered = True
            if frame.timestamp.duplicated().any(): raise ValueError("Duplicated repaired minute aggregation")
        elif name.startswith("data/krx_daily_parquet/daily_ohlcv/"):
            symbol = Path(name).parent.name; relevant = [x for x in revisions if x["symbol"] == symbol]
            if relevant:
                frame = pd.read_parquet(old / name)
                for row in relevant:
                    mask = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d") == row["date"]
                    if int(mask.sum()) != 1: raise ValueError("Missing corrected daily observation")
                    for field, value in row["original"].items(): frame.loc[mask, field] = value
                altered = True
        if altered:
            out = target / name; out.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(out, index=False); files[name] = digest(out)
            changed.append({"path": name, "before": sha, "after": files[name]})
        else: bind(old / name, name)
    for name in overlays: bind(REPO / name, name, True)
    # freeze_market formerly omitted this daily metadata table while retaining
    # the YAML map used by OLR. Restore the exact development dependency.
    sector_name = "data/krx_daily_parquet/tables/sector_map.parquet"
    development = read(root / "development/study.json")
    development_manifest = read(Path(development["manifest"]))
    sector_source = Path(development["code"]) / sector_name
    if digest(sector_source) != development_manifest["files"][sector_name]:
        raise ValueError("Changed development sector table")
    bind(sector_source, sector_name)
    import yaml
    canonical = yaml.safe_load((target / "config/olr/sector_map.yaml").read_text())["sector_map"]
    sector_rows = pd.read_parquet(target / sector_name)
    for row in sector_rows.itertuples():
        symbol = str(row.ticker).zfill(6)
        if str(row.sector).strip().upper() != str(canonical.get(symbol, "UNKNOWN")).strip().upper():
            raise ValueError("Development KALCB/OLR sector mismatch " + symbol)
    write_json(root / "sector_dependency_repair.json", {"pass": True, "source": bound(sector_source),
        "restored": bound(target / sector_name), "symbols": len(sector_rows),
        "canonical_agreement": True, "new_classifications_introduced": False,
        "prior_attempt": bound(ROOT / "validation/logs/qualified_control_baseline.log"),
        "prior_attempt_performance_outputs": 0,
        "consequence": "Rebuild all features and fixed-model predictions; do not relabel already scored candidates"})
    for child in ("repaired_sessions", "minute_sources", "flow_sources"):
        for p in (root / child).rglob("*"):
            if p.is_file(): bind(p, "research/source_repair/" + p.relative_to(root).as_posix(), True)
    for name in ("repair_registration.json", "flow_basis_audit.json", "volume_revision_audit.json", "lookback_overlap_probe.json"):
        bind(root / name, "research/source_repair/" + name, True)
    bind(REPO / "scripts/portfolio_source_repair.py", "research/source_repair/repair_program.py", True)
    bind(REPO / "scripts/portfolio_repaired_validation.py", "research/source_repair/validation_program.py", True)
    schedule = {"version": "corporate-action-exposure-guard-v1", "mode": "assert_zero_exposure_or_abort_replay",
        "events": [
            {"symbol": "068270", "ex_date": "2026-06-04", "type": "bonus_issue", "ratio": "1/20",
             "record_date": "2026-06-05", "scheduled_listing": "2026-06-30",
             "fractional_shares": "cash valued at listing-day close; no immediate tradability assumed",
             "announced": "2026-05-21", "sources": [
                "https://www.celltrion.com/ko-kr/investment/notice/announce/4775",
                "https://www.celltrion.com/en-us/company/notice/4785"]},
            {"symbol": "214370", "ex_date": "2026-06-29", "type": "interim_dividend",
             "provider_code": "05", "terms": "not required only if no account exposure; otherwise replay must abort",
             "provider_receipt": bound(PARENT / "provider_basis_audit/214370_original.json")},
        ]}
    action_name = "research/source_repair/corporate_actions.json"
    write_json(target / action_name, schedule); files[action_name] = digest(target / action_name)
    contract = copy.deepcopy(parent["evidence"]["shared_market_inputs"])
    contract.pop("market_input_hash")
    for name in contract["files"]: contract["files"][name] = files[name]
    contract["files"][sector_name] = files[sector_name]
    contract["kalcb_sector_table"] = sector_name
    contract["source_repair_proof"] = "research/source_repair/repaired_sessions/proof.json"
    for name, sha in files.items():
        if name.startswith("research/source_repair/"): contract["files"][name] = sha
    contract["market_input_hash"] = signature(contract)
    qp = build_evidence(target, contract, target / "research/forward_acceptance")
    qualified = read(qp); oldq = read(old / parent["evidence"]["qualification_path"])
    qualified["market_files"] = oldq["market_files"]
    qualified["capacity"] = oldq["capacity"]; qualified["session_resolution_report"] = oldq["session_resolution_report"]
    qualified["capacity"]["market_basis"] = read(qp.parent / "market_contract.json")["normalization"]["target_basis"]
    write_json(qp, qualified)
    acceptance = read(qp.parent / "acceptance.json")
    if acceptance["basis_failures"]: raise ValueError("Repaired lookback still has incompatible bases")
    for p in qp.parent.iterdir(): files[p.relative_to(target).as_posix()] = digest(p)
    evidence = copy.deepcopy(parent["evidence"]); evidence["shared_market_inputs"] = contract
    evidence["corporate_actions"] = action_name
    evidence["reserved_validation"]["source_repair"] = bound(root / "repaired_sessions/proof.json")
    command = "from kis_core.trading_calendar import get_trading_calendar;get_trading_calendar();from backtests.auto.shared.cache_keys import source_generation_hash;print(source_generation_hash())"
    generation = subprocess.check_output([sys.executable, "-B", "-c", command], cwd=target, text=True).strip()
    payload = {**{k: v for k, v in parent.items() if k != "manifest_hash"}, "files": files,
               "evidence": evidence, "source_generation_hash": generation}
    write_json(destination / "manifest.json", {**payload, "manifest_hash": signature(payload)})
    write_json(root / "repair_generation.json", {"parent_manifest": bound(PARENT / "forward_inputs/manifest.json"),
        "new_manifest": bound(destination / "manifest.json"), "changed_market_files": changed,
        "strategy_parameters_changed": False, "source_gate_pending_inference_and_action_exposure": True})
    print(json.dumps({"repaired_generation": generation, "changed_market_files": len(changed),
                      "basis_failures": len(acceptance["basis_failures"])}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["flows", "volumes", "freeze"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    {"flows": flows, "volumes": volumes, "freeze": freeze}[args.command](args.root)

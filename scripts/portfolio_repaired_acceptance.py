"""Source qualification, frozen inference and the six preregistered OOS cases."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
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
from scripts.portfolio_repaired_validation import verify
from backtests.strategies.kalcb.executed_value import digest, write_json


def inference(root):
    """Use frozen strategy helpers even if other research changes the checkout."""
    market = read(root / "forward_inputs/manifest.json")
    source = root / "forward_inputs/files"
    runtime = root / "inference_runtime"; runtime.mkdir(exist_ok=False)
    programs = ("scripts/portfolio_forward_validation.py", "scripts/portfolio_frozen_inference.py", "scripts/portfolio_forward_olr.py")
    files = {}
    for name, sha in market["files"].items():
        if name in programs: continue
        if digest(source / name) != sha: raise ValueError("Changed repaired generation")
        target = runtime / name; target.parent.mkdir(parents=True, exist_ok=True)
        try: os.link(source / name, target)
        except OSError: shutil.copyfile(source / name, target)
        files[name] = sha
    for name in programs:
        target = runtime / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target); files[name] = digest(target)
    receipt = {"pass": True, "root": str(runtime), "files": files,
        "market_manifest": bound(root / "forward_inputs/manifest.json"),
        "strategy_helpers": "Byte-identical to repaired frozen generation; no mutable checkout strategy imports",
        "orchestration_programs": {n: bound(runtime / n) for n in programs}}
    write_json(root / "inference_runtime.json", receipt)
    shutil.copyfile(root / "forward_program_dependency_review.json", root / "prior_forward_program_dependency_review.json")
    write_json(root / "forward_program_dependency_review.json", {"inference_source_parity_pass": True,
        "runtime": bound(root / "inference_runtime.json"), "frozen_helper_files": len(files) - len(programs),
        "working_import_probe": bound(root / "inference_import_audit.json"),
        "resolution": "Working checkout drift avoided by running all inference in its own frozen runtime"})
    inference_stages(root, runtime, files)


def inference_stages(root, runtime, files):
    for stage in ("prepare-features", "feature-bridge", "predict", "audit-split"):
        print(json.dumps({"inference_stage_started": stage}), flush=True)
        log = root / ("inference_" + stage + ".log")
        with log.open("x", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable, "-B", "-u", str(runtime / "scripts/portfolio_forward_validation.py"), stage, "--root", str(root)],
                cwd=runtime, stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        if result.returncode: raise ValueError("Frozen inference failed; inspect " + str(log))
        print(json.dumps({"inference_stage_complete": stage}), flush=True)
    for name, sha in files.items():
        if digest(runtime / name) != sha: raise ValueError("Inference runtime changed during execution")
    write_json(root / "repaired_inference_completion.json", {"complete": True,
        "runtime": bound(root / "inference_runtime.json"), "predictions": bound(root / "forward_prediction_completion.json"),
        "split": bound(root / "split_audit_verified.json"), "fit_performed": False})


def resume_inference(root):
    """Preserve the failed source-only attempt and add omitted frozen evidence."""
    if (root / "validation").exists(): raise ValueError("Cannot repair inference after opening validation")
    receipt = read(root / "inference_runtime.json"); runtime = Path(receipt["root"])
    for name, sha in receipt["files"].items():
        if digest(runtime / name) != sha: raise ValueError("Changed inference runtime before resume")
    old_receipt = root / "inference_runtime_initial.json"
    if old_receipt.exists(): raise FileExistsError("Preserve previous repair attempt")
    shutil.copyfile(root / "inference_runtime.json", old_receipt)
    manifest = read(root / "forward_inputs/manifest.json"); source = root / "forward_inputs/files"
    for name, sha in manifest["files"].items():
        if name in receipt["files"]: continue
        if digest(source / name) != sha: raise ValueError("Changed frozen evidence before runtime completion")
        target = runtime / name; target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists(): raise FileExistsError(target)
        try: os.link(source / name, target)
        except OSError: shutil.copyfile(source / name, target)
        receipt["files"][name] = sha
    receipt["initial_runtime"] = bound(old_receipt)
    receipt["source_only_retry_reason"] = "Calendar loader requires its content-bound integrity reference files; include the complete frozen generation"
    write_json(root / "inference_runtime.json", receipt)
    review = read(root / "forward_program_dependency_review.json"); review["runtime"] = bound(root / "inference_runtime.json")
    write_json(root / "forward_program_dependency_review.json", review)
    for name, preserved in (("forward_features", "failed_feature_preparation_1"),
                            ("inference_prepare-features.log", "failed_feature_preparation_1.log")):
        old = (root / name).resolve(); new = (root / preserved).resolve()
        if old.parent != root.resolve() or new.parent != root.resolve() or new.exists():
            raise ValueError("Unexpected source-only retry location")
        old.rename(new)
    inference_stages(root, runtime, receipt["files"])


def development_overlap(root):
    """Verify retained model-selection inputs without reading OOS performance."""
    import pandas as pd
    study = read(PARENT / "development/study.json")
    original = read(Path(study["manifest"]))
    repaired = read(root / "forward_inputs/manifest.json")
    old_root = Path(study["code"])
    new_root = root / "forward_inputs/files"
    inputs = []; changes = []; totals = {}; coverage_errors = []
    columns = ["open", "high", "low", "close", "volume"]
    for kind, prefix, marker, key in (
        ("intraday", "data/kis_intraday_parquet/", "_5m_", "timestamp"),
        ("daily", "data/krx_daily_parquet/daily_ohlcv/", "_daily_ohlcv_", "date"),
    ):
        old_files = {Path(n).parent.name: n for n in original["files"]
                     if n.startswith(prefix) and marker in n}
        new_files = {Path(n).parent.name: n for n in repaired["files"]
                     if n.startswith(prefix) and marker in n}
        count = 0; development_count = 0
        for symbol, new_name in sorted(new_files.items()):
            old_name = old_files[symbol]
            old_path = old_root / old_name; new_path = new_root / new_name
            if digest(old_path) != original["files"][old_name] or digest(new_path) != repaired["files"][new_name]:
                raise ValueError("Changed overlap audit input")
            a = pd.read_parquet(old_path); b = pd.read_parquet(new_path)
            for frame in (a, b):
                frame[key] = pd.to_datetime(frame[key])
                if frame[key].duplicated().any(): raise ValueError("Duplicate overlap timestamps")
            a = a.set_index(key); b = b.set_index(key)
            shared = a.index.intersection(b.index)
            shared = shared[shared.strftime("%Y-%m-%d") < "2026-06-01"]
            dates = shared.strftime("%Y-%m-%d")
            dev = (dates >= "2025-08-07") & (dates <= "2026-03-31")
            expected = a.index[(a.index.strftime("%Y-%m-%d") >= "2025-08-07") &
                               (a.index.strftime("%Y-%m-%d") <= "2026-03-31")]
            missing = expected.difference(b.index)
            if len(missing): coverage_errors.append({"kind": kind, "symbol": symbol, "missing_rows": len(missing)})
            different = (a.loc[shared, columns] != b.loc[shared, columns]).any(axis=1)
            if different.any():
                changed = shared[different]
                changes.append({"kind": kind, "symbol": symbol, "rows": int(different.sum()),
                    "min": str(changed.min()), "max": str(changed.max()),
                    "development_changes": int((different.to_numpy() & dev).sum()),
                    "changes_through_development_end": int((changed.strftime("%Y-%m-%d") <= "2026-03-31").sum())})
            count += len(shared); development_count += int(dev.sum())
            inputs.append({"kind": kind, "symbol": symbol, "old": bound(old_path), "repaired": bound(new_path), "rows": len(shared)})
        totals[kind] = {"symbols": len(new_files), "compared_rows": count, "development_rows": development_count}
    passed = (not coverage_errors and all(t["symbols"] == 103 for t in totals.values()) and
              not any(c["changes_through_development_end"] for c in changes))
    result = {"pass": passed, "totals": totals, "changes": changes, "coverage_errors": coverage_errors,
        "inputs": inputs, "development_manifest": bound(Path(study["manifest"])),
        "repaired_manifest": bound(root / "forward_inputs/manifest.json"),
        "scope": "All overlapping pre-OOS OHLCV; complete development interval coverage; no changed observations through March 31",
        "prior_incomplete_daily_path_probe": bound(root / "lookback_overlap_verified.json")}
    target = root / "lookback_development_verified.json"
    if target.exists(): raise FileExistsError(target)
    write_json(target, result)
    print(json.dumps({k: v for k, v in result.items() if k != "inputs"}), flush=True)
    if not passed: raise ValueError("Development source overlap not qualified")


def source_gate(root):
    import pandas as pd
    manifest = read(root / "forward_inputs/manifest.json")
    for name, sha in manifest["files"].items():
        if digest(root / "forward_inputs/files" / name) != sha: raise ValueError("Changed repaired source " + name)
    old = read(PARENT / "provider_basis_audit/audit.json")
    for item in old["original_and_adjusted_controls"].values(): verify(item)
    acceptance = read(root / "forward_inputs/files/research/forward_acceptance/acceptance.json")
    if acceptance["basis_failures"]: raise ValueError("Unresolved source-basis failure")
    for name in ("flow_basis_audit.json", "volume_revision_audit.json", "repaired_sessions/proof.json", "split_audit_verified.json", "lookback_development_verified.json"):
        if not read(root / name)["pass"]: raise ValueError("Failed required source/split audit")
    source = root / "forward_inputs/files"; rows = []; errors = []; symbols = set()
    sector_repair = read(root / "sector_dependency_repair.json")
    if not sector_repair["pass"] or sector_repair["symbols"] != 103:
        raise ValueError("Missing frozen sector dependency qualification")
    verify(sector_repair["source"]); verify(sector_repair["restored"])
    sectors = {str(r.ticker).zfill(6): str(r.sector).strip().upper()
               for r in pd.read_parquet(source / "data/krx_daily_parquet/tables/sector_map.parquet").itertuples()}
    sector_checks = 0
    with (root / "forward_features/opening/opportunities.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (str(row["sector"]).strip().upper() != sectors[row["symbol"]]
                    or str(row["candidate"]["sector"]).strip().upper() != sectors[row["symbol"]]):
                raise ValueError("Incorrect sector in prepared model inputs")
            sector_checks += 1
    if not sector_checks: raise ValueError("Missing sector feature checks")
    snapshots = read(root / "forward_policies/interaction_cost_margin.snapshots.json") + read(root / "forward_olr_snapshots.json")
    snapshots += [json.loads(line) for line in (root / "forward_policies/timed.snapshots.jsonl").read_text(encoding="utf-8").splitlines()]
    for snap in snapshots:
        for candidate in snap["candidates"]:
            if str(candidate["sector"]).strip().upper() != sectors[candidate["symbol"]]:
                raise ValueError("Inconsistent portfolio sector input")
    controls = {(x["symbol"], x["date"]): x for x in old["sessions"]}
    for name in manifest["files"]:
        if not name.startswith("data/krx_daily_parquet/daily_ohlcv/"): continue
        symbol = Path(name).parent.name; symbols.add(symbol)
        data = pd.read_parquet(source / name)
        data = data.assign(day=pd.to_datetime(data.date).dt.strftime("%Y-%m-%d"))
        for row in data[data.day.between("2026-05-29", "2026-07-10")].to_dict("records"):
            control = controls[symbol, row["day"]]
            actual = {k: float(row[k]) for k in ("open", "high", "low", "close", "volume")}
            checked = {**control, "archive": actual, "matches_original": actual == control["original"],
                       "matches_adjusted": actual == control["adjusted"]}
            rows.append(checked)
            if not checked["matches_original"]: errors.append(checked)
    if len(symbols) != 103 or len(rows) != len(old["sessions"]) or errors:
        raise ValueError("Incomplete or inconsistent original-price daily control coverage")
    actions = read(source / manifest["evidence"]["corporate_actions"])
    olr = read(root / "forward_olr_snapshots.json")
    exposures = []
    for event in actions["events"]:
        eligible = [snap["trade_date"] for snap in olr if snap["trade_date"] < event["ex_date"]
                    and any(c["symbol"] == event["symbol"] and c.get("tradable", True) for c in snap["candidates"])]
        if eligible: raise ValueError("Potential overnight corporate-action exposure requires full entitlement accounting before opening OOS")
        exposures.append({"symbol": event["symbol"], "ex_date": event["ex_date"], "prior_eligible_olr_sessions": eligible,
            "kalcb": "Same-day strategy; actual positions/pending orders must pass the runtime action guard",
            "issuers_excluded": False})
    (root / "provider_basis_audit").mkdir(exist_ok=False)
    audit = {"symbols": len(symbols), "sessions": rows, "non_original_sessions": [], "all_original": True,
        "original_and_adjusted_controls": old["original_and_adjusted_controls"],
        "source_gate_pass": True, "strategy_performance_evaluated": False,
        "prior_failed_audit": bound(PARENT / "provider_basis_audit/audit.json"),
        "frozen_manifest": bound(root / "forward_inputs/manifest.json")}
    write_json(root / "provider_basis_audit/audit.json", audit)
    review = {"version": "repaired-reserved-source-review-v1", "source_gate_pass": True,
        "provider_audit": bound(root / "provider_basis_audit/audit.json"),
        "dependencies": {"frozen_manifest": bound(root / "forward_inputs/manifest.json"),
            **{name: bound(root / name) for name in ("repaired_sessions/proof.json", "flow_basis_audit.json", "volume_revision_audit.json",
                "lookback_overlap_probe.json", "lookback_development_verified.json", "sector_dependency_repair.json", "inference_runtime.json", "repaired_inference_completion.json", "split_audit_verified.json")}},
        "sector_feature_checks": sector_checks, "candidate_sector_consistency_pass": True,
        "corporate_action_preflight": exposures, "corporate_action_runtime_guard_required": True,
        "price_share_convention": "Original KRW/share observations; dated exact source reconstruction for 068270 May29, June1 and June2",
        "feature_convention": "Frozen strategy formulas applied to original-price history; no OOS-designed total-return feature change",
        "volume_revision_cause_known": False, "volume_revision_scope": "Final day daily total, unused for window decisions; directly verified original interval capacity preserved",
        "strategy_performance_evaluated": False, "original_unqualified_generation_preserved": True}
    for name in ('metadata_annotation.json', 'metadata_frozen_calendar_verification.json', 'qualification_repair.json'):
        if (root / name).exists():
            if not read(root / name)['pass']: raise ValueError('Failed audit annotation repair')
            review['dependencies'][name] = bound(root / name)
    if (root / 'metadata_annotation.json').exists():
        annotation = read(root / 'metadata_annotation.json')
        if annotation['candidates_changed'] or annotation['original_candidate_digest'] != annotation['corrected_candidate_digest']:
            raise ValueError('Audit annotation altered strategy choices')
        for key in ('source', 'output', 'database', 'frozen_runtime', 'program'): verify(annotation[key])
    write_json(root / "source_basis_review.json", review)
    print(json.dumps({"source_gate_pass": True, "symbols": 103, "daily_controls": len(rows),
        "corporate_action_preflight": exposures}), flush=True)


def run(root):
    from scripts.portfolio_forward_validation import freeze_replay
    from scripts.portfolio_native_budget_study import one
    review = read(root / "source_basis_review.json")
    if not review["source_gate_pass"]: raise ValueError("Source review failed")
    for item in review["dependencies"].values(): verify(item)
    freeze_replay(root)
    registration = read(root / "validation/registration.json")
    write_json(root / "oos_opening.json", {"opened_at": datetime.now(timezone.utc).isoformat(),
        "registration": bound(root / "validation/registration.json"), "source_review": bound(root / "source_basis_review.json"),
        "no_reselection": True, "return_thresholds_unchanged": True})
    def execute_job(job):
        one(root / "validation", job, full=True, folder="qualified")
        audit = read(root / "validation/qualified" / (job["recipe"] + "_" + job["scenario"]) / "corporate_action_audit.json")
        if not audit["pass"] or len(audit["events"]) != 2: raise ValueError("Incomplete actual corporate-action exposure proof")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute_job, job) for job in registration['jobs']]
        for future in futures: future.result()
    if (root / 'qualification_repair.json').exists():
        old = Path(read(root / 'qualification_repair.json')['old_audit']['path']).parent
        current = root / 'validation/qualified/control_baseline'
        checks = {n: read(old / n) == read(current / n) for n in ('daily.json','equity.json')}
        excluded = {'source_artifact_hash','route_metadata','cohort_metadata'}
        def trades(folder):
            return [{k:v for k,v in row.items() if k not in excluded} for row in read(folder / 'trades.json')]
        def fills(folder):
            return [{**{k:row[k] for k in ('strategy_id','symbol','side','qty','price','timestamp','reason')},
                     'fee':row['metadata']['fee']} for row in read(folder / 'fills.json')]
        checks.update(trade_economics=trades(old)==trades(current), fill_economics=fills(old)==fills(current))
        write_json(root / 'economic_evidence_repair_bridge.json', {'pass':all(checks.values()), 'checks':checks,
            'old_completion':bound(old / 'completion.json'), 'qualified_completion':bound(current / 'completion.json'),
            'scope':'Exact account path, trade economics and fill economics; provenance metadata may differ'})
        if not all(checks.values()): raise ValueError('Audit evidence repair changed economic results')
    from scripts.portfolio_targeted_result import assess
    result = assess(root)
    write_json(root / "acceptance.json", result)
    print(json.dumps({"oos_complete": True, "acceptance": result["acceptance"],
                      "promotion_eligible": result["promotion_eligible"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inference", "resume-inference", "development-overlap", "source-gate", "run"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    {"inference": inference, "resume-inference": resume_inference, "development-overlap": development_overlap, "source-gate": source_gate, "run": run}[args.command](args.root)

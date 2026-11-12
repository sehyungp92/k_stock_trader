"""Resolve registered OLR cases without accepting or scoring audit rejections."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import subprocess
import sys

try:
    from scripts.olr_rule_study import digest, load_study, read, signature, write
    from scripts.olr_rule_analysis import load_case
except ModuleNotFoundError:
    from olr_rule_study import digest, load_study, read, signature, write
    from olr_rule_analysis import load_case


def reject_receipt(spec, directory):
    job = read(directory / "job.json")
    metrics = read(directory / "forensic_metrics.json")
    if metrics.get("audit_pass") is True and metrics.get("daily_equity_coverage", {}).get("status") == "complete":
        raise ValueError("Cannot reject qualified economics through the observer")
    write(directory / "rejected.json", {"status": "unqualified_native_result", "study_hash": spec["study_hash"],
        "job_hash": signature(job), "source_hash": spec["source_generation_hash"],
        "resolved_at_utc": datetime.now(timezone.utc).isoformat(), "observer_sha256": digest(__file__),
        "outputs": {p.name: digest(p) for p in directory.iterdir()
                    if p.is_file() and p.name not in {"completed.json", "rejected.json", "worker.log"}},
        "performance": "not_reported_or_scored"})
    load_case(spec, job["case"], job["mode"])


def worker(spec, name, mode, mutation_file=None):
    module_spec = importlib.util.spec_from_file_location("original_olr_rule_worker", spec["harness"])
    original = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(original)
    sys.path.insert(0, spec["engine_root"])
    from backtests.strategies.olr import final_diagnostics
    original_export = final_diagnostics.export_case
    rejected = False

    def observed_export(result, config, mutations, directory):
        nonlocal rejected
        try:
            return original_export(result, config, mutations, directory)
        except ValueError:
            if result.metrics.get("audit_pass") is True and result.metrics.get("daily_equity_coverage", {}).get("status") == "complete":
                raise
            write(directory / "forensic_metrics.json", final_diagnostics.json_value(result.metrics))
            write(directory / "forensic_request.json", {"config": config, "mutations": mutations,
                  "scope": "Forensic evidence only; rejected native result"})
            replay = result.replay_result
            for label, records in (("fills", replay.broker.fills), ("trades", result.trades),
                                   ("ledger", replay.broker.journal), ("decisions", result.decisions)):
                final_diagnostics.write_rows(directory / ("forensic_" + label + ".jsonl.gz"), records)
            reject_receipt(spec, directory)
            rejected = True
            raise

    final_diagnostics.export_case = observed_export
    try:
        original.replay_worker(spec, name, mode, mutation_file)
    except ValueError:
        if not rejected:
            raise
        print("REJECTED", name, mode, "native audit/evidence gate; performance excluded", flush=True)


def core(spec, study):
    names = ["control"] + sorted(name for name in spec["cases"] if name != "control")
    for mode in ("nominal", "sizing_cap"):
        for name in names:
            directory = Path(spec["output"]) / "cases" / name / mode
            if not any((directory / label).exists() for label in ("completed.json", "rejected.json")):
                directory.mkdir(parents=True, exist_ok=True)
                print("START", name, mode, flush=True)
                with (directory / "worker.log").open("w", encoding="utf-8") as stream:
                    subprocess.run([sys.executable, "-B", "-u", __file__, "worker", "--study", str(study),
                                    "--case", name, "--mode", mode], cwd=spec["engine_root"],
                                   stdout=stream, stderr=subprocess.STDOUT, check=True)
            loaded = load_case(spec, name, mode)
            m = loaded["metrics"]
            headline = "unqualified; performance excluded" if loaded.get("qualification_rejected") else {
                key: m[key] for key in ("official_mtm_net_return_pct", "max_drawdown_pct", "total_trades")}
            print("RESOLVED", name, mode, headline, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("core", "worker"))
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--mode", default="nominal")
    parser.add_argument("--mutation-file", type=Path)
    args = parser.parse_args()
    spec = load_study(args.study)
    if args.command == "worker":
        worker(spec, args.case, args.mode, args.mutation_file)
    else:
        core(spec, args.study)


if __name__ == "__main__":
    main()

"""Preserve a rejected native result for diagnosis without accepting or scoring it."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--mode", default="nominal")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh forensic directory; preserve earlier attempts")
    import json
    registered = json.loads(args.study.read_text(encoding="utf-8"))
    module_spec = importlib.util.spec_from_file_location("frozen_olr_rule_worker", registered["harness"])
    harness = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(harness)
    spec = harness.load_study(args.study)
    args.output.mkdir(parents=True)
    spec["output"] = str(args.output.resolve())
    sys.path.insert(0, spec["engine_root"])
    from backtests.strategies.olr import final_diagnostics
    original_export = final_diagnostics.export_case

    def observed_export(result, config, mutations, directory):
        # Called only after the unchanged native evaluator and its audit finish.
        # Save forensic evidence first; the original acceptance gate still runs.
        harness.write(args.output / "forensic_metrics.json", final_diagnostics.json_value(result.metrics))
        harness.write(args.output / "forensic_request.json", {
            "study_hash": spec["study_hash"], "source_generation_hash": spec["source_generation_hash"],
            "case": args.case, "mode": args.mode, "config": config, "mutations": mutations,
            "scope": "Forensic evidence only; failed audit output cannot qualify or supply a return claim"})
        replay = result.replay_result
        for name, rows in (("fills", replay.broker.fills), ("trades", result.trades),
                           ("ledger", replay.broker.journal), ("decisions", result.decisions)):
            final_diagnostics.write_rows(args.output / ("forensic_" + name + ".jsonl.gz"), rows)
        return original_export(result, config, mutations, directory)

    final_diagnostics.export_case = observed_export
    harness.replay_worker(spec, args.case, args.mode)


if __name__ == "__main__":
    main()

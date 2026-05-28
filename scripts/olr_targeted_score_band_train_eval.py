from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEARCH_PATH = ROOT / "scripts" / "olr_nonmonotone_score_band_search.py"
ROUND3_CONFIG = ROOT / "data" / "backtests" / "output" / "olr" / "round_3" / "optimized_config.json"
DEFAULT_OUTPUT = ROOT / "tmp" / "olr_nonmonotone_score_band_search"


def main() -> int:
    parser = argparse.ArgumentParser(description="Targeted train validation for selected OLR score-band rules.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--holdout-days", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    started = time.monotonic()
    output_dir = Path(args.output_dir)
    eval_path = output_dir / "evaluations.jsonl"
    progress_path = output_dir / "progress.jsonl"
    search = load_module("olr_nonmonotone_score_band_search", SEARCH_PATH)
    helper = search.load_helper()
    config = helper.normalize_runtime_config("olr", helper.load_yaml_config(str(ROOT / "config" / "optimization" / "olr.yaml")))
    config["capability_level"] = "real_replay"
    config["holdout_days"] = int(args.holdout_days)
    config["use_full_available_window"] = True

    round3 = json.loads(ROUND3_CONFIG.read_text(encoding="utf-8"))
    base = copy.deepcopy(round3["mutations"])
    candidates = search.build_candidates(helper, base)
    candidate_by_label = {candidate.label: candidate for candidate in candidates}
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    missing = [label for label in labels if label not in candidate_by_label]
    if missing:
        raise SystemExit(f"Unknown labels: {', '.join(missing)}")
    requested = [candidate_by_label[label] for label in labels]

    cached = helper.load_cached_evaluations(eval_path)
    helper.evaluate_candidates(
        config,
        requested,
        "train",
        output_dir,
        eval_path,
        progress_path,
        cached,
        holdout_days=int(args.holdout_days),
        batch_size=max(1, int(args.batch_size)),
    )
    cached = helper.load_cached_evaluations(eval_path)
    oos_by_label = {label: row for (window, label), row in cached.items() if window == "oos"}
    train_by_label = {label: row for (window, label), row in cached.items() if window == "train"}
    payload = search.summarize(helper, candidates, oos_by_label, train_by_label, started)
    search.write_json(output_dir / "nonmonotone_score_band_search.json", payload)
    search.write_text(output_dir / "nonmonotone_score_band_search.md", search.render_markdown(payload))
    if payload.get("best_balanced"):
        search.write_json(output_dir / "best_balanced_mutations.json", payload["best_balanced"].get("mutations", {}))
    if payload.get("best_oos_first"):
        search.write_json(output_dir / "best_oos_first_mutations.json", payload["best_oos_first"].get("mutations", {}))
    search.status(
        progress_path,
        "targeted_train_complete",
        labels=labels,
        elapsed_seconds=round(time.monotonic() - started, 3),
        summary_path=str(output_dir / "nonmonotone_score_band_search.md"),
    )
    print(json.dumps({"status": "complete", "labels": labels}, sort_keys=True), flush=True)
    return 0


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())

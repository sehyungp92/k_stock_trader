from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMOTE_PATH = ROOT / "scripts" / "promote_olr_round3_reject_score_min300.py"
ROUND3_DIR = ROOT / "data" / "backtests" / "output" / "olr" / "round_3"


def main() -> int:
    promote = load_promote()
    payload = promote.read_json(ROUND3_DIR / "round_final_full_diagnostics.json")
    optimized = promote.read_json(ROUND3_DIR / "optimized_config.json")
    train = {"metrics": payload["train"]["metrics"], "source": payload["train"].get("source", {})}
    oos = {"metrics": payload["oos"]["metrics"], "source": payload["oos"].get("source", {})}
    promote.write_text(ROUND3_DIR / "round_final_diagnostics.txt", promote.render_final_diagnostics(None, payload, train, oos))
    promote.write_text(ROUND3_DIR / "round_evaluation.txt", promote.render_round_evaluation(None, payload, optimized["mutations"]))
    status = promote.build_diagnostics_status(payload["generated_at_utc"])
    promote.write_json(ROUND3_DIR / "round_final_diagnostics_status.json", status)
    for name in ("optimized_config.json", "run_summary.json"):
        path = ROUND3_DIR / name
        data = promote.read_json(path)
        data["final_diagnostics"] = status
        promote.write_json(path, data)
    state_path = ROUND3_DIR / "phase_state.json"
    state = promote.read_json(state_path)
    promote.replace_mutation_value(state, "olr.afternoon.reject_score_min", 300.0)
    promote.write_json(state_path, state)
    return 0


def load_promote():
    spec = importlib.util.spec_from_file_location("promote_olr_round3_reject_score_min300", PROMOTE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {PROMOTE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())

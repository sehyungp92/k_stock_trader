from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .round_manager import RoundManager


def bootstrap_round_1(
    *,
    strategy: str,
    mutations: dict[str, Any],
    output_root: Path = Path("data/backtests/output"),
    diagnostics_text: str = "Round 1 bootstrapped from live config.",
    final_metrics: dict[str, Any] | None = None,
) -> Path:
    manager = RoundManager("stock", strategy, base_dir=output_root)
    round_dir = manager.get_round_dir(1)
    manager.diagnostics_path(round_dir).write_text(diagnostics_text, encoding="utf-8")
    manager.write_run_spec(round_dir, 1, strategy_name=strategy, description="Round 1 bootstrap", baseline_mutations=mutations, overwrite=True)
    manager.write_run_summary(round_dir, mutations, final_metrics or {}, [], round_num=1)
    manager.write_optimized_config(round_dir, mutations)
    manager.append_to_manifest(1, mutations, final_metrics or {})
    return round_dir


from __future__ import annotations

from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.auto.shared.round_manager import RoundManager
from backtests.strategies.registry import create_plugin


def test_two_phase_optimisation_writes_artifacts(tmp_path):
    manager = RoundManager("stock", "kalcb", base_dir=tmp_path)
    round_num, round_dir = manager.resolve_round(None, for_write=True, expected_phases=2)
    plugin = create_plugin("kalcb", {"capability_level": "synthetic"}, output_dir=round_dir, max_workers=1)
    plugin.num_phases = 2
    state = PhaseRunner(plugin, round_dir, round_name="smoke", round_manager=manager, round_num=round_num).run_all_phases()
    assert state.completed_phases == [1, 2]
    assert (round_dir / "phase_state.json").exists()
    assert (round_dir / "progress.json").exists()
    assert (round_dir / "phase_activity_log.jsonl").exists()
    assert (round_dir / "optimized_config.json").exists()
    assert (round_dir / "round_final_diagnostics.txt").exists()
    assert (round_dir / "round_evaluation.txt").exists()

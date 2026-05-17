from __future__ import annotations

from pathlib import Path

from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.auto.shared.round_manager import RoundManager
from backtests.auto.shared.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.auto.shared.phase_state import PhaseState
from backtests.auto.shared.types import EndOfRoundArtifacts


class TinyPlugin:
    name = "tiny"
    num_phases = 1
    ultimate_targets = {}
    initial_mutations = {"seed": "live"}

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        del phase, state
        return PhaseSpec("noop", [], lambda metrics: [], {}, {}, PhaseAnalysisPolicy())

    def create_evaluate_batch(self, *args, **kwargs):
        raise AssertionError("not needed")

    def compute_final_metrics(self, mutations: dict) -> dict[str, float]:
        del mutations
        return {}

    def run_phase_diagnostics(self, *args, **kwargs) -> str:
        return ""

    def run_enhanced_diagnostics(self, *args, **kwargs) -> str:
        return ""

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        del state
        return EndOfRoundArtifacts("", {}, "")


def test_round_two_loads_previous_optimized_config_as_baseline(tmp_path: Path):
    manager = RoundManager("stock", "tiny", base_dir=tmp_path)
    round_1 = manager.get_round_dir(1)
    manager.write_optimized_config(round_1, {"seed": "round_1", "phase_1": 7})
    round_2 = manager.get_round_dir(2)

    state = PhaseRunner(TinyPlugin(), round_2, round_manager=manager, round_num=2).load_state()

    assert state.cumulative_mutations == {"seed": "round_1", "phase_1": 7}


def test_phase_spec_exposes_redesign_scoring_weights_fn():
    def redesign(*args, **kwargs):
        del args, kwargs
        return None

    spec = PhaseSpec(
        "focus",
        [],
        lambda metrics: [],
        {},
        {},
        PhaseAnalysisPolicy(redesign_scoring_weights_fn=redesign),
    )

    assert spec.redesign_scoring_weights_fn is redesign

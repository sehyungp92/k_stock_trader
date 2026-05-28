from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.auto.shared.round_manager import RoundManager, canonicalize_metrics
from backtests.auto.shared.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.auto.shared.phase_state import PhaseState, save_phase_state
from backtests.auto.shared.types import EndOfRoundArtifacts
from backtests.strategies.common.plugin_base import build_execution_contract


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


def test_completed_round_summary_advances_even_with_baseline_phase_state(tmp_path: Path):
    manager = RoundManager("stock", "tiny", base_dir=tmp_path)
    round_1 = manager.get_round_dir(1)
    state = PhaseState(current_phase=0, cumulative_mutations={"seed": "round_1"}, round_name="round_1")
    save_phase_state(state, manager.phase_state_path(round_1))
    manager.write_optimized_config(round_1, {"seed": "round_1"})
    manager.write_run_summary(round_1, {"seed": "round_1"}, {"total_trades": 1.0}, [], round_num=1)
    manager.append_to_manifest(1, {"seed": "round_1"}, {"total_trades": 1.0})

    round_num, round_dir = manager.resolve_round(None, for_write=True, expected_phases=6)

    assert round_num == 2
    assert round_dir.name == "round_2"


def test_completed_round_rerun_refreshes_end_of_round_artifacts(tmp_path: Path):
    class RefreshPlugin(TinyPlugin):
        def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
            del state
            return EndOfRoundArtifacts("fresh diagnostics", {"signal_extraction": "fresh"}, "fresh verdict")

    manager = RoundManager("stock", "tiny", base_dir=tmp_path)
    round_1 = manager.get_round_dir(1)
    state = PhaseState(
        current_phase=1,
        completed_phases=[1],
        cumulative_mutations={"seed": "round_1"},
        phase_results={1: {"focus": "done", "base_score": 0.0, "final_score": 0.0, "kept_features": []}},
    )
    save_phase_state(state, manager.phase_state_path(round_1))
    (round_1 / "round_final_diagnostics.txt").write_text("stale diagnostics", encoding="utf-8")
    (round_1 / "round_evaluation.txt").write_text("stale evaluation", encoding="utf-8")

    PhaseRunner(RefreshPlugin(), round_1, round_manager=manager, round_num=1).run_all_phases()

    assert (round_1 / "round_final_diagnostics.txt").read_text(encoding="utf-8") == "fresh diagnostics"
    assert "fresh verdict" in (round_1 / "round_evaluation.txt").read_text(encoding="utf-8")
    status = json.loads((round_1 / "round_final_diagnostics_status.json").read_text(encoding="utf-8"))
    assert status["mode"] == "shared_end_of_round_artifacts"
    assert status["round_final_diagnostics_exists"] is True


def test_completed_round_runs_plugin_full_diagnostics_hook(tmp_path: Path):
    class FullDiagnosticsPlugin(TinyPlugin):
        name = "fulltiny"

        def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
            del state
            return EndOfRoundArtifacts("light diagnostics", {"signal_extraction": "fresh"}, "fresh verdict")

        def write_full_diagnostics(self, state: PhaseState, output_dir: Path, *, round_num: int | None = None, round_name: str = "") -> dict:
            del state, round_name
            (output_dir / "round_final_diagnostics.txt").write_text("full diagnostics", encoding="utf-8")
            return {"round": round_num, "full": True}

    manager = RoundManager("stock", "fulltiny", base_dir=tmp_path)
    round_1 = manager.get_round_dir(1)
    state = PhaseState(
        current_phase=1,
        completed_phases=[1],
        cumulative_mutations={"seed": "round_1"},
        phase_results={1: {"focus": "done", "base_score": 0.0, "final_score": 0.0, "kept_features": []}},
    )
    save_phase_state(state, manager.phase_state_path(round_1))

    PhaseRunner(FullDiagnosticsPlugin(), round_1, round_manager=manager, round_num=1).run_all_phases()

    assert (round_1 / "round_final_diagnostics.txt").read_text(encoding="utf-8") == "full diagnostics"
    status = json.loads((round_1 / "round_final_diagnostics_status.json").read_text(encoding="utf-8"))
    assert status["mode"] == "plugin_full_diagnostics"
    assert status["payload"] == {"round": 1, "full": True}
    run_summary = json.loads((round_1 / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["final_diagnostics"]["mode"] == "plugin_full_diagnostics"


def test_completed_round_fails_if_plugin_full_diagnostics_does_not_write_report(tmp_path: Path):
    class BrokenFullDiagnosticsPlugin(TinyPlugin):
        name = "brokenfulltiny"
        requires_full_diagnostics = True

        def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
            del state
            return EndOfRoundArtifacts("light diagnostics", {}, "fresh verdict")

        def write_full_diagnostics(self, state: PhaseState, output_dir: Path, *, round_num: int | None = None, round_name: str = "") -> dict:
            del state, output_dir, round_num, round_name
            return {"full": False}

    manager = RoundManager("stock", "brokenfulltiny", base_dir=tmp_path)
    round_1 = manager.get_round_dir(1)
    state = PhaseState(
        current_phase=1,
        completed_phases=[1],
        cumulative_mutations={"seed": "round_1"},
        phase_results={1: {"focus": "done", "base_score": 0.0, "final_score": 0.0, "kept_features": []}},
    )
    save_phase_state(state, manager.phase_state_path(round_1))

    with pytest.raises(RuntimeError, match="did not create round_final_diagnostics.txt"):
        PhaseRunner(BrokenFullDiagnosticsPlugin(), round_1, round_manager=manager, round_num=1).run_all_phases()


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


def test_phase_spec_exposes_metric_contract_fields():
    spec = PhaseSpec(
        "focus",
        [],
        lambda metrics: [],
        {},
        {},
        PhaseAnalysisPolicy(),
        phase_metric_basis="direct_official_replay",
        primary_promotion_metric="official_mtm_net_return_pct",
        proxy_metric_keys=("proxy_return",),
        official_metric_keys=("official_mtm_net_return_pct",),
        promotion_requires_audit_pass=True,
    )

    assert spec.phase_metric_basis == "direct_official_replay"
    assert spec.primary_promotion_metric == "official_mtm_net_return_pct"
    assert spec.proxy_metric_keys == ("proxy_return",)
    assert spec.official_metric_keys == ("official_mtm_net_return_pct",)
    assert spec.promotion_requires_audit_pass is True


def test_manifest_preserves_official_metric_contract(tmp_path: Path):
    manager = RoundManager("stock", "tiny", base_dir=tmp_path)
    metrics = {
        "total_trades": 4.0,
        "net_return_pct": 0.01,
        "net_return_pct_basis": "closed_trade_net_pnl_over_initial_equity",
        "official_mtm_net_return_pct": 0.012,
        "official_metric_basis": "SimBroker.equity_curve_bar_level_mtm",
        "primary_promotion_metric": "official_mtm_net_return_pct",
        "primary_promotion_value": 0.012,
        "primary_promotion_basis": "SimBroker.equity_curve_bar_level_mtm",
        "audit_status": "direct_official_replay",
        "official_replay_pass": True,
        "audit_pass": False,
        "same_bar_fill_count": 0.0,
        "metric_contract": {"primary_promotion_metric": "official_mtm_net_return_pct"},
        "execution_contract": {
            "strategy": "tiny",
            "source_fingerprint": "source-a",
            "feature_manifest_hash": "features-a",
            "candidate_snapshot_hash": "candidates-a",
            "cost_policy": {"commission_bps": 1.0},
            "fill_timing": "next_open",
        },
    }

    manager.append_to_manifest(1, {"seed": "x"}, metrics)
    entry = json.loads(manager.manifest_path.read_text(encoding="utf-8"))["rounds"][0]

    assert entry["official_mtm_net_return_pct"] == 0.012
    assert entry["primary_promotion_metric"] == "official_mtm_net_return_pct"
    assert entry["audit_status"] == "direct_official_replay"
    assert entry["execution_contract"]["source_fingerprint"] == "source-a"
    assert entry["source_fingerprint"] == "source-a"
    assert entry["feature_manifest_hash"] == "features-a"
    assert entry["candidate_snapshot_hash"] == "candidates-a"
    assert entry["cost_policy_hash"]
    assert entry["fill_timing"] == "next_open"


def test_previous_optimized_config_rejects_incompatible_execution_contract(tmp_path: Path):
    manager = RoundManager("stock", "tiny", base_dir=tmp_path)
    round_1 = manager.get_round_dir(1)
    manager.write_optimized_config(
        round_1,
        {"seed": "round_1"},
        artifact_metadata={
            "execution_contract": {
                "strategy": "tiny",
                "source_fingerprint": "old-source",
                "primary_promotion_metric": "official_mtm_net_return_pct",
            }
        },
    )

    with pytest.raises(ValueError, match="incompatible execution contract"):
        manager.get_previous_mutations(
            2,
            expected_execution_contract={
                "strategy": "tiny",
                "source_fingerprint": "new-source",
                "primary_promotion_metric": "official_mtm_net_return_pct",
            },
        )


def test_previous_optimized_config_can_explicitly_allow_incompatible_baseline(tmp_path: Path):
    manager = RoundManager("stock", "tiny", base_dir=tmp_path)
    round_1 = manager.get_round_dir(1)
    manager.write_optimized_config(
        round_1,
        {"seed": "round_1"},
        artifact_metadata={"execution_contract": {"strategy": "tiny", "source_fingerprint": "old-source"}},
    )

    mutations = manager.get_previous_mutations(
        2,
        expected_execution_contract={"strategy": "tiny", "source_fingerprint": "new-source"},
        allow_incompatible_baseline=True,
    )

    assert mutations == {"seed": "round_1"}


def test_previous_optimized_config_accepts_actual_execution_contract_superset(tmp_path: Path):
    manager = RoundManager("stock", "tiny", base_dir=tmp_path)
    round_1 = manager.get_round_dir(1)
    manager.write_optimized_config(
        round_1,
        {"seed": "round_1"},
        artifact_metadata={
            "execution_contract": {
                "strategy": "tiny",
                "source_fingerprint": "source-a",
                "date_window": {"start": "2026-01-01", "end": "2026-01-31", "sessions": 20},
            }
        },
    )

    mutations = manager.get_previous_mutations(
        2,
        expected_execution_contract={
            "strategy": "tiny",
            "source_fingerprint": "source-a",
            "date_window": {"start": "2026-01-01", "end": "2026-01-31"},
        },
    )

    assert mutations == {"seed": "round_1"}


def test_phase_runner_baseline_integrity_blocks_value_drift(tmp_path: Path):
    class GuardedPlugin(TinyPlugin):
        config = {"baseline_integrity_required": True, "baseline_integrity_tolerance_abs": 0.001}

        def __init__(self, value: float):
            self.value = value

        def compute_final_metrics(self, mutations: dict) -> dict[str, float]:
            del mutations
            return {
                "official_mtm_net_return_pct": self.value,
                "primary_promotion_metric": "official_mtm_net_return_pct",
                "primary_promotion_value": self.value,
                "strategy_core_version": "unit-core",
            }

    manager = RoundManager("stock", "tiny", base_dir=tmp_path)
    round_1 = manager.get_round_dir(1)
    manager.write_optimized_config(
        round_1,
        {"seed": "round_1"},
        artifact_metadata={
            "primary_promotion_metric": "official_mtm_net_return_pct",
            "primary_promotion_value": 0.442,
            "strategy_core_version": "previous-core",
        },
    )

    runner = PhaseRunner(GuardedPlugin(0.371), manager.get_round_dir(2), round_manager=manager, round_num=2)
    with pytest.raises(RuntimeError, match="initial baseline drifted"):
        runner._validate_initial_baseline({"seed": "round_1"})

    passed = PhaseRunner(GuardedPlugin(0.4425), manager.get_round_dir(2), round_manager=manager, round_num=2)._validate_initial_baseline({"seed": "round_1"})

    assert passed["passed"] is True
    assert passed["expected_value"] == 0.442


def test_phase_runner_baseline_integrity_reads_metric_contract_value(tmp_path: Path):
    class ContractPlugin(TinyPlugin):
        config = {"baseline_integrity_required": True, "baseline_integrity_tolerance_abs": 0.001}

        def compute_final_metrics(self, mutations: dict) -> dict[str, float]:
            del mutations
            return {
                "broker_net_return_pct": 1.2309,
                "primary_promotion_metric": "broker_net_return_pct",
                "primary_promotion_value": 1.2309,
            }

    manager = RoundManager("stock", "tiny", base_dir=tmp_path)
    round_1 = manager.get_round_dir(1)
    manager.write_optimized_config(
        round_1,
        {"seed": "round_1"},
        artifact_metadata={
            "metric_contract": {
                "primary_promotion_metric": "broker_net_return_pct",
                "primary_promotion_value": 1.2305,
            },
        },
    )

    passed = PhaseRunner(ContractPlugin(), manager.get_round_dir(2), round_manager=manager, round_num=2)._validate_initial_baseline({"seed": "round_1"})

    assert passed["passed"] is True
    assert passed["metric"] == "broker_net_return_pct"
    assert passed["expected_value"] == 1.2305


def test_canonicalize_metrics_keeps_headline_fields_compact():
    canonical = canonicalize_metrics(
        {
            "official_mtm_net_return_pct": 0.02,
            "official_metric_basis": "SimBroker.equity_curve_bar_level_mtm",
            "primary_promotion_metric": "official_mtm_net_return_pct",
            "primary_promotion_value": 0.02,
            "audit_status": "direct_official_replay",
            "metric_contract": {"primary_promotion_metric": "official_mtm_net_return_pct"},
        }
    )

    assert canonical["official_mtm_net_return_pct"] == 0.02
    assert canonical["primary_promotion_value"] == 0.02
    assert "metric_contract" not in canonical


def test_execution_contract_prefers_latest_replay_identity_over_baseline_context():
    class PluginWithBaselineContext:
        name = "tiny"
        execution_context = {
            "source_fingerprint": "baseline-source",
            "feature_manifest_hash": "baseline-features",
            "candidate_snapshot_hash": "baseline-candidates",
        }
        config = {}

    plugin = PluginWithBaselineContext()
    plugin._last_result = SimpleNamespace(
        source_fingerprint="latest-source",
        feature_bundle_hash="latest-features",
        candidate_snapshot_hash="latest-candidates",
    )

    contract = build_execution_contract(plugin)

    assert contract["source_fingerprint"] == "latest-source"
    assert contract["feature_manifest_hash"] == "latest-features"
    assert contract["candidate_snapshot_hash"] == "latest-candidates"

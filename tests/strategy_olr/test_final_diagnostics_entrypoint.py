"""CLI compatibility and shared replay/export contracts, without financial runs."""
from copy import deepcopy
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import refresh_olr_final_artifacts as cli
from scripts import refresh_olr_round5_final_artifacts as legacy
from backtests.strategies.olr import exported_diagnostics, final_diagnostics, plugin, replay_cache, runner


def test_legacy_delegates_arguments_and_retains_only_its_historical_default(monkeypatch):
    calls = []
    monkeypatch.setattr(legacy, "refresh_main", lambda argv, **kw: calls.append((argv, kw)))
    arguments = ["--round", "2", "--native-interval-cases", "plan.json"]
    legacy.main(arguments)
    assert calls == [(arguments, {"default_round": 5})]
    assert cli.parse_args([], default_round=5)[1].round == 5
    assert cli.parse_args(arguments, default_round=5)[1].round == 2


@pytest.mark.parametrize("arguments, message", [
    ([], "--round"),
    (["--round", "0"], "positive"),
    (["--round", "2", "--exported-cases", "a", "--native-interval-cases", "b"], "one native"),
    (["--round", "2", "--native-interval-cases", "a", "--execution-root", "b"], "per-window"),
    (["--round", "2", "--verified-interval-report", "a"], "explicit SHA-256"),
])
def test_invalid_request_stops_before_import_or_output(arguments, message, tmp_path, capsys):
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as error:
        cli.main([*arguments, "--output-root", str(output)])
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not output.exists()


def test_native_bound_profile_cannot_fall_back_to_generic_feed(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("native_case_plan_sha256: sealed\n", encoding="utf-8")
    output = tmp_path / "output"
    with pytest.raises(SystemExit):
        cli.main(["--round", "2", "--config", str(config), "--output-root", str(output)])
    assert "matching --native-capacity-cases or --native-interval-cases" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("filename", ["refresh_olr_final_artifacts.py", "refresh_olr_round5_final_artifacts.py"])
def test_frozen_execution_root_selected_before_validator_import(filename, tmp_path):
    frozen = tmp_path / "frozen"
    package = frozen / "backtests"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('raise RuntimeError("FROZEN_ENGINE_SELECTED")\n', encoding="utf-8")
    result = subprocess.run([sys.executable, "-B", str(cli.ROOT / "scripts" / filename),
                             "--round", "2", "--execution-root", str(frozen)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "FROZEN_ENGINE_SELECTED" in result.stderr


@pytest.mark.parametrize("entrypoint", [cli.main, legacy.main])
def test_interval_route_preserves_validation_and_shared_lifecycle(entrypoint, tmp_path, monkeypatch):
    from backtests.auto.shared import phase_runner, phase_state, round_manager
    from backtests.strategies.olr import interval_round_diagnostics
    events = []
    config = tmp_path / "config.yaml"
    config.write_text("native_case_plan_sha256: sealed\ninitial_mutations: {policy: retained}\n", encoding="utf-8")
    plan, parent = tmp_path / "plan.json", tmp_path / "accepted"
    class Adapter:
        execution_context = {}
        def __init__(self, config, path, **kwargs):
            assert path == plan
            events.append("interval")
        def reuse_verified_report(self, directory, digest):
            assert directory == parent and digest == "digest"
            events.append("reuse")
        def require_bound_cases(self, mutations):
            assert mutations == {"policy": "retained"}
            events.append("validate")
    class Manager:
        def __init__(self, *args, **kwargs): pass
        def get_round_dir(self, number):
            assert number == 2
            return tmp_path / "output" / "round_2"
        def write_run_spec(self, *args, **kwargs): events.append("spec")
    class Runner:
        round_name = "retained"
        state_path = tmp_path / "state.json"
        def __init__(self, subject, directory, *args, **kwargs):
            assert isinstance(subject, Adapter)
        def run_end_of_round(self, state):
            assert state.cumulative_mutations == {"policy": "retained"}
            events.append("shared_final")
    monkeypatch.setattr(interval_round_diagnostics, "IntervalDiagnosticsPlugin", Adapter)
    monkeypatch.setattr(round_manager, "RoundManager", Manager)
    monkeypatch.setattr(phase_runner, "PhaseRunner", Runner)
    monkeypatch.setattr(phase_state, "save_phase_state", lambda *args: events.append("save"))
    entrypoint(["--round", "2", "--config", str(config), "--baseline-from-config",
                "--native-interval-cases", str(plan), "--verified-interval-report", str(parent),
                "--verified-interval-report-sha256", "digest"])
    assert events == ["interval", "reuse", "validate", "spec", "save", "shared_final"]


def test_scenario_contract_preserves_cost_cap_and_consumed_window():
    config = {"initial_equity": 1000000, "holdout_start": "2026-04-01", "holdout_end": "2026-07-10",
              "date_range": {"start": "2025-10-01", "end": "2026-03-31"}}
    mutations = {"olr.cost.slippage_bps": 12.0}
    original = deepcopy((config, mutations))
    window = {"start": "2026-04-01", "end": "2026-07-10"}
    requests = exported_diagnostics.diagnostic_requests(config, mutations, window)
    assert list(requests) == ["training", "participation-stress", "cost-stress", "combined-stress",
                              "no_upward_compounding", "consumed_regression"]
    for name, participation, cost in (("participation-stress", .005, 12.),
                                       ("cost-stress", .01, 22.), ("combined-stress", .005, 22.)):
        assert requests[name]["config"]["capacity_scenario"] == {"name": name, "participation": participation}
        assert requests[name]["mutations"]["olr.cost.slippage_bps"] == cost
        assert requests[name]["mutations"]["olr.robustness.slippage_bps"] == cost
    assert requests["no_upward_compounding"]["mutations"] == {**mutations, "olr.allocation.equity_cap_krw": 1000000.}
    regression = requests["consumed_regression"]["config"]
    assert regression["date_range"] == window and regression["use_full_available_window"] is True
    assert "holdout_start" not in regression and "holdout_end" not in regression
    assert (config, mutations) == original


def test_direct_writer_executes_shared_requests_with_correct_source_window(tmp_path, monkeypatch):
    config = {"initial_equity": 1000000, "date_range": {"start": "2025-10-01", "end": "2026-03-31"}}
    mutations = {"olr.cost.slippage_bps": 5.}
    window = {"start": "2026-04-01", "end": "2026-07-10"}
    baseline = SimpleNamespace(metrics={})
    subject = SimpleNamespace(training_config=config, config={"regression_window": window},
        num_phases=1, execution_context={}, capability_level="real", _result_cache={},
        _evaluate_result=lambda _: baseline)
    expected = exported_diagnostics.diagnostic_requests(config, mutations, window)
    calls, exports = [], []
    monkeypatch.setattr(plugin, "_format_olr_diagnostics", lambda *args: "test")
    monkeypatch.setattr(plugin, "_greedy_from_state", lambda *args: None)
    monkeypatch.setattr(replay_cache, "load_olr_real_replay_bundle", lambda *args: "training-source")
    monkeypatch.setattr(replay_cache, "load_olr_declared_window_bundle", lambda *args: "regression-source")
    monkeypatch.setattr(final_diagnostics, "release_diagnostic_replay_memory", lambda *args: None)
    def run(config, mutations, *, replay_bundle):
        calls.append((config, mutations, replay_bundle))
        return baseline
    def export(result, config, mutations, directory):
        exports.append(directory.name)
        return {"config": config, "mutations": mutations}
    monkeypatch.setattr(runner, "run_olr_backtest", run)
    monkeypatch.setattr(final_diagnostics, "export_case", export)
    monkeypatch.setattr(final_diagnostics, "render_full_diagnostics", lambda *args, **kw: kw["cases"])
    result = final_diagnostics.write_full_diagnostics(subject, SimpleNamespace(cumulative_mutations=mutations),
                                                     tmp_path, round_num=2, round_name="test")
    assert result == expected
    assert exports == list(expected)
    assert calls == [(request["config"], request["mutations"],
                       "regression-source" if name == "consumed_regression" else "training-source")
                      for name, request in expected.items() if name != "training"]

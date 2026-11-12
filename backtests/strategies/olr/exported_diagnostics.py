"""Observer-only reporting of sealed, request-bound native OLR case exports.

Isolating replays avoids carrying native allocator state across scenarios. This
adapter performs no search or simulation and delegates publication to PhaseRunner.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from backtests.engine.capacity import CAPACITY_SCENARIOS
from backtests.strategies.olr.final_diagnostics import require_bound_result
from backtests.strategies.olr.plugin import OLROptimizationPlugin, _format_olr_diagnostics, _greedy_from_state
from strategy_olr.config import OLRConfig

REQUIRED_EXPORTS = frozenset({"request.json", "metrics.json", "audit_receipt.json", "cohorts.json",
    "trades.jsonl.gz", "fills.jsonl.gz", "decisions.jsonl.gz", "ledger.jsonl.gz", "equity.jsonl.gz"})


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def diagnostic_requests(config, mutations, regression_window):
    """Shared scenario contract for direct replay and sealed-export validation."""
    requests = {"training": {"config": config, "mutations": mutations}}
    cfg = OLRConfig.from_mapping(config, mutations)
    for name, participation, extra_bps in CAPACITY_SCENARIOS[1:]:
        requests[name] = {
            "config": {**config, "capacity_scenario": {"name": name, "participation": participation}},
            "mutations": {**mutations, "olr.cost.slippage_bps": cfg.slippage_bps + extra_bps,
                "olr.robustness.slippage_bps": cfg.slippage_bps + extra_bps},
        }
    requests["no_upward_compounding"] = {"config": config,
        "mutations": {**mutations, "olr.allocation.equity_cap_krw": float(config["initial_equity"])}}
    if not regression_window or set(regression_window) != {"start", "end"}:
        raise ValueError("Full diagnostics require an explicit consumed regression window")
    regression = {**config, "date_range": dict(regression_window), "use_full_available_window": True}
    regression.pop("holdout_start", None)
    regression.pop("holdout_end", None)
    requests["consumed_regression"] = {"config": regression, "mutations": mutations}
    return requests


def load_sealed_case(plan_path, plan, name, expected):
    directory = Path(plan["output"]) / name
    seal = read(directory / "export_receipt.json")
    if seal.get("case") != name or seal.get("plan_sha256") != digest(plan_path):
        raise ValueError("Diagnostic export belongs to another case or plan")
    if not REQUIRED_EXPORTS.issubset(seal.get("outputs", {})):
        raise ValueError("Diagnostic export is incomplete")
    for filename, sha in seal["outputs"].items():
        if Path(filename).name != filename or digest(directory / filename) != sha:
            raise ValueError("Diagnostic export changed or escapes its directory")
    request = read(directory / "request.json")
    if any(request.get(key) != expected[key] for key in ("config", "mutations")):
        raise ValueError("Diagnostic export differs from the requested recipe or scenario")
    metrics = read(directory / "metrics.json")
    if metrics.get("audit_receipt") != read(directory / "audit_receipt.json"):
        raise ValueError("Diagnostic audit receipt disagrees with saved metrics")
    result = SimpleNamespace(metrics=metrics)
    require_bound_result(result, expected["config"], expected["mutations"])
    return {"metrics": metrics, "path": str(directory), "cohorts": read(directory / "cohorts.json")}


class ExportedOLRDiagnosticsPlugin(OLROptimizationPlugin):
    def __init__(self, config, plan_path, *, output_dir):
        self.plan_path = Path(plan_path)
        self.plan = read(self.plan_path)
        self.exported_cases = {}
        super().__init__(config, output_dir=output_dir, max_workers=1)
        expected = diagnostic_requests(self.training_config, self.initial_mutations,
            self.config.get("regression_window"))
        if set(self.plan["cases"]) != set(expected):
            raise ValueError("Full diagnostics require every registered scenario exactly once")
        for name, request in expected.items():
            if any(self.plan["cases"][name].get(key) != request[key] for key in ("config", "mutations")):
                raise ValueError("Diagnostic plan differs from the saved strategy configuration")
            if name not in self.exported_cases:
                self.exported_cases[name] = load_sealed_case(self.plan_path, self.plan, name, request)

    def _evaluate_result(self, mutations):
        if self.canonicalize_mutations(mutations) != self.initial_mutations:
            raise ValueError("Export-only diagnostics cannot evaluate a different recipe")
        expected = {"config": self.training_config, "mutations": mutations}
        if "training" not in self.exported_cases:
            self.exported_cases["training"] = load_sealed_case(self.plan_path, self.plan, "training", expected)
        case = self.exported_cases["training"]
        metrics = case["metrics"]
        return SimpleNamespace(metrics=metrics,
            source_fingerprint=metrics["source_fingerprint"],
            feature_bundle_hash=metrics["feature_manifest_hash"])

    def write_full_diagnostics(self, state, output_dir, *, round_num=None, round_name=""):
        # Load the current reporting formatter explicitly; execution and receipt
        # validation still use the immutable engine selected by the CLI.
        path = Path(__file__).with_name("final_diagnostics.py")
        spec = importlib.util.spec_from_file_location("backtests.strategies.olr.reporting_diagnostics", path)
        formatter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(formatter)
        metrics = self._evaluate_result(state.cumulative_mutations).metrics
        text = _format_olr_diagnostics(self.num_phases, metrics,
            _greedy_from_state(state, metrics), self.execution_context)
        self.reporting_source = {"adapter_sha256": digest(__file__), "formatter_sha256": digest(path),
            "case_plan_sha256": digest(self.plan_path), "mode": "sealed_native_exports",
            "executing_native_source_generation": metrics["audit_receipt"]["source_generation_hash"]}
        return formatter.render_full_diagnostics(self, state, Path(output_dir), round_num=round_num,
            round_name=round_name, cases=self.exported_cases, training_text=text)

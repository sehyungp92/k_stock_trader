"""Tests for ExperimentRegistry."""
import tempfile
from collections import Counter
from pathlib import Path

from instrumentation.src.experiment_registry import ExperimentRegistry


class TestExperimentRegistry:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def _write_yaml(self, content: str) -> Path:
        p = Path(self.tmpdir) / "experiments.yaml"
        p.write_text(content)
        return p

    def test_load_empty_yaml(self):
        """No experiments, no errors."""
        p = self._write_yaml("experiments: []")
        reg = ExperimentRegistry(p)
        assert reg.active_experiments() == []

    def test_load_orchestrator_format(self):
        """List of experiment dicts parsed correctly."""
        p = self._write_yaml("""
experiments:
  - experiment_id: "exp_001"
    title: "Test tighter stop"
    hypothesis: "Tighter stop reduces drawdown"
    variants:
      control:
        params: {stop_mult: 2.2}
        allocation_pct: 50
      treatment:
        params: {stop_mult: 1.8}
        allocation_pct: 50
    primary_metric: "sharpe"
    start_date: "2026-01-01"
    strategy_type: "kmp"
    status: "active"
""")
        reg = ExperimentRegistry(p)
        exp = reg.get_experiment("exp_001")
        assert exp is not None
        assert exp.title == "Test tighter stop"
        assert len(exp.variants) == 2

    def test_load_legacy_format(self):
        """Dict of {exp_id: data} parsed correctly."""
        p = self._write_yaml("""
experiments:
  exp_legacy:
    hypothesis: "Test hypothesis"
    variants:
      control: {params: {x: 1}, allocation_pct: 50}
      treatment: {params: {x: 2}, allocation_pct: 50}
    status: "active"
    start_date: "2026-01-01"
""")
        reg = ExperimentRegistry(p)
        exp = reg.get_experiment("exp_legacy")
        assert exp is not None
        assert len(exp.variants) == 2

    def test_variant_assignment_deterministic(self):
        """Same trade_id + exp_id -> same variant."""
        p = self._write_yaml("""
experiments:
  - experiment_id: "exp_det"
    variants:
      control: {allocation_pct: 50}
      treatment: {allocation_pct: 50}
    status: "active"
    start_date: "2026-01-01"
""")
        reg = ExperimentRegistry(p)
        v1 = reg.assign_variant("exp_det", "trade_abc")
        v2 = reg.assign_variant("exp_det", "trade_abc")
        assert v1 == v2
        assert v1 in ("control", "treatment")

    def test_variant_assignment_distribution(self):
        """1000 hashes approximate 50/50 allocation."""
        p = self._write_yaml("""
experiments:
  - experiment_id: "exp_dist"
    variants:
      control: {allocation_pct: 50}
      treatment: {allocation_pct: 50}
    status: "active"
    start_date: "2026-01-01"
""")
        reg = ExperimentRegistry(p)
        counts = Counter()
        for i in range(1000):
            v = reg.assign_variant("exp_dist", f"trade_{i}")
            counts[v] += 1

        # Should be roughly 50/50 (allow 10% margin)
        assert counts["control"] > 350
        assert counts["treatment"] > 350

    def test_active_experiments_filtered_by_date(self):
        """start_date/end_date respected."""
        p = self._write_yaml("""
experiments:
  - experiment_id: "future"
    start_date: "2099-01-01"
    status: "active"
    variants:
      a: {}
  - experiment_id: "past"
    start_date: "2020-01-01"
    end_date: "2020-12-31"
    status: "active"
    variants:
      a: {}
  - experiment_id: "current"
    start_date: "2020-01-01"
    status: "active"
    variants:
      a: {}
""")
        reg = ExperimentRegistry(p)
        active = reg.active_experiments(as_of="2026-03-15")
        ids = [e.experiment_id for e in active]
        assert "current" in ids
        assert "future" not in ids
        assert "past" not in ids

    def test_export_active_format(self):
        """Matches DailySnapshot expected structure."""
        p = self._write_yaml("""
experiments:
  - experiment_id: "exp_export"
    title: "Export Test"
    hypothesis: "Test export"
    variants:
      control: {allocation_pct: 50}
      treatment: {allocation_pct: 50}
    primary_metric: "sharpe"
    start_date: "2026-01-01"
    strategy_type: "kmp"
    status: "active"
""")
        reg = ExperimentRegistry(p)
        exported = reg.export_active(as_of="2026-03-15")
        assert "exp_export" in exported
        entry = exported["exp_export"]
        assert entry["title"] == "Export Test"
        assert "control" in entry["variants"]
        assert "treatment" in entry["variants"]
        assert entry["primary_metric"] == "sharpe"

    def test_get_variant_params(self):
        """Returns correct params dict for variant name."""
        p = self._write_yaml("""
experiments:
  - experiment_id: "exp_params"
    variants:
      control: {params: {stop_mult: 2.2}, allocation_pct: 50}
      treatment: {params: {stop_mult: 1.8}, allocation_pct: 50}
    status: "active"
    start_date: "2026-01-01"
""")
        reg = ExperimentRegistry(p)
        params = reg.get_variant_params("exp_params", "treatment")
        assert params == {"stop_mult": 1.8}

        params_ctrl = reg.get_variant_params("exp_params", "control")
        assert params_ctrl == {"stop_mult": 2.2}

        # Non-existent
        assert reg.get_variant_params("exp_params", "nonexist") == {}
        assert reg.get_variant_params("nonexist", "control") == {}

    def test_assign_variant_missing_experiment(self):
        """Returns empty string if experiment not found."""
        p = self._write_yaml("experiments: []")
        reg = ExperimentRegistry(p)
        assert reg.assign_variant("nonexist", "trade_1") == ""

    def test_reload(self):
        """Re-read config file."""
        p = self._write_yaml("experiments: []")
        reg = ExperimentRegistry(p)
        assert len(reg.active_experiments()) == 0

        p.write_text("""
experiments:
  - experiment_id: "new_exp"
    status: "active"
    start_date: "2026-01-01"
    variants:
      a: {}
""")
        reg.reload()
        assert len(reg.active_experiments(as_of="2026-03-15")) == 1

    def test_missing_config_file(self):
        """Gracefully handle missing file."""
        p = Path(self.tmpdir) / "nonexistent.yaml"
        reg = ExperimentRegistry(p)
        assert reg.active_experiments() == []

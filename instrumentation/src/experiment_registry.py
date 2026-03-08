"""Experiment registry — loads experiment configs and assigns variants."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger("instrumentation.experiment_registry")


@dataclass
class ExperimentVariant:
    """One variant in an experiment."""
    name: str
    params: dict = field(default_factory=dict)
    allocation_pct: float = 50.0


@dataclass
class ExperimentMetadata:
    """Configuration for a single experiment."""
    experiment_id: str
    title: str = ""
    hypothesis: str = ""
    variants: list[ExperimentVariant] = field(default_factory=list)
    primary_metric: str = "sharpe"
    start_date: str = ""
    end_date: Optional[str] = None
    strategy_type: str = ""
    status: str = "active"
    success_metric: str = "pnl"
    max_duration_days: int = 30
    min_trades_per_variant: int = 30
    allocation_method: str = "hash"  # "hash" for deterministic


class ExperimentRegistry:
    """Loads experiments from YAML and provides variant assignment.

    YAML format (config/experiments.yaml):

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
            success_metric: "pnl"
            start_date: "2026-03-15"
            max_duration_days: 14
            strategy_type: "kmp"
            status: "active"
    """

    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        self._experiments: dict[str, ExperimentMetadata] = {}
        self._load()

    def _load(self) -> None:
        """Load experiments from YAML file."""
        if not self._config_path.exists():
            return

        try:
            data = yaml.safe_load(self._config_path.read_text()) or {}
        except Exception as e:
            logger.warning("Failed to load experiments.yaml: %s", e)
            return

        raw_experiments = data.get("experiments", [])
        if isinstance(raw_experiments, dict):
            # Legacy format: {exp_id: {hypothesis, variants, ...}}
            for exp_id, exp_data in raw_experiments.items():
                self._parse_experiment(exp_id, exp_data)
        elif isinstance(raw_experiments, list):
            # Orchestrator format: [{experiment_id, title, variants, ...}]
            for exp_data in raw_experiments:
                exp_id = exp_data.get("experiment_id", "")
                if exp_id:
                    self._parse_experiment(exp_id, exp_data)

    def _parse_experiment(self, exp_id: str, data: dict) -> None:
        """Parse a single experiment entry."""
        variants = []
        raw_variants = data.get("variants", {})

        if isinstance(raw_variants, dict):
            for vname, vdata in raw_variants.items():
                if isinstance(vdata, dict):
                    variants.append(ExperimentVariant(
                        name=vname,
                        params=vdata.get("params", {}),
                        allocation_pct=vdata.get("allocation_pct", 50.0),
                    ))
                else:
                    variants.append(ExperimentVariant(name=vname))
        elif isinstance(raw_variants, list):
            # Simple list of variant names
            pct = 100.0 / len(raw_variants) if raw_variants else 50.0
            for vname in raw_variants:
                variants.append(ExperimentVariant(name=vname, allocation_pct=pct))

        self._experiments[exp_id] = ExperimentMetadata(
            experiment_id=exp_id,
            title=data.get("title", ""),
            hypothesis=data.get("hypothesis", ""),
            variants=variants,
            primary_metric=data.get("primary_metric", "sharpe"),
            start_date=data.get("start_date", ""),
            end_date=data.get("end_date"),
            strategy_type=data.get("strategy_type", ""),
            status=data.get("status", "active"),
            success_metric=data.get("success_metric", "pnl"),
            max_duration_days=data.get("max_duration_days", 30),
            min_trades_per_variant=data.get("min_trades_per_variant", 30),
            allocation_method=data.get("allocation_method", "hash"),
        )

    def reload(self) -> None:
        """Re-read config file (for hot-reload on file change)."""
        self._experiments.clear()
        self._load()

    def get_experiment(self, experiment_id: str) -> ExperimentMetadata | None:
        return self._experiments.get(experiment_id)

    def active_experiments(self, as_of: str | None = None) -> list[ExperimentMetadata]:
        """Return experiments that are currently active."""
        today = as_of or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        active = []
        for exp in self._experiments.values():
            if exp.status != "active":
                continue
            if exp.start_date and exp.start_date > today:
                continue
            if exp.end_date and exp.end_date < today:
                continue
            active.append(exp)
        return active

    def assign_variant(self, experiment_id: str, trade_id: str) -> str:
        """Assign a trade to a variant deterministically.

        Uses hash(trade_id + experiment_id) for reproducible assignment.
        Returns empty string if experiment not found.
        """
        exp = self.get_experiment(experiment_id)
        if exp is None or not exp.variants:
            return ""

        raw = f"{trade_id}|{experiment_id}"
        hash_val = int(hashlib.sha256(raw.encode()).hexdigest(), 16)
        bucket = hash_val % 10000  # 0.01% granularity

        total_alloc = sum(v.allocation_pct for v in exp.variants)
        if total_alloc <= 0:
            return exp.variants[0].name

        cumulative = 0.0
        for variant in exp.variants:
            cumulative += variant.allocation_pct / total_alloc * 10000
            if bucket < cumulative:
                return variant.name

        return exp.variants[-1].name

    def get_variant_params(self, experiment_id: str, variant_name: str) -> dict:
        """Get parameter overrides for a specific variant."""
        exp = self.get_experiment(experiment_id)
        if exp is None:
            return {}
        for v in exp.variants:
            if v.name == variant_name:
                return v.params
        return {}

    def export_active(self, as_of: str | None = None) -> dict:
        """Export active experiments for DailySnapshot inclusion."""
        result = {}
        for exp in self.active_experiments(as_of):
            result[exp.experiment_id] = {
                "title": exp.title,
                "hypothesis": exp.hypothesis,
                "variants": [v.name for v in exp.variants],
                "primary_metric": exp.primary_metric,
                "start_date": exp.start_date,
                "strategy_type": exp.strategy_type,
            }
        return result

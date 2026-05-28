from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .plugin import PortfolioSynergyOptimizationPlugin


@dataclass(slots=True)
class PortfolioSynergyBacktestResult:
    metrics: dict[str, Any]
    trades: list[dict[str, Any]] = field(default_factory=list)
    blocked_trades: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    source_fingerprint: str = ""
    feature_manifest_hash: str = ""
    candidate_snapshot_hash: str = ""


def run_portfolio_synergy_backtest(config: dict[str, Any] | None = None, mutations: dict[str, Any] | None = None, **kwargs: Any) -> PortfolioSynergyBacktestResult:
    plugin = PortfolioSynergyOptimizationPlugin(
        config or {},
        output_dir=Path(kwargs.get("output_dir", ".")),
        max_workers=int(kwargs.get("max_workers", 1) or 1),
        capability_level=str(kwargs.get("capability_level", "source_artifact_portfolio_replay")),
    )
    result = plugin._evaluate({**plugin.initial_mutations, **dict(mutations or {})})
    metrics = dict(result.metrics)
    return PortfolioSynergyBacktestResult(
        metrics=metrics,
        trades=list(result.accepted),
        blocked_trades=list(result.blocked),
        decisions=[],
        source_fingerprint=str(metrics.get("source_fingerprint", "")),
        feature_manifest_hash=str(metrics.get("feature_manifest_hash", "")),
        candidate_snapshot_hash=str(metrics.get("candidate_snapshot_hash", "")),
    )


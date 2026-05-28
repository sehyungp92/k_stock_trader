from __future__ import annotations

import importlib
from typing import Any

BACKTEST_RUNNERS = {
    "kalcb": ("backtests.strategies.kalcb.runner", "run_kalcb_backtest"),
    "olr": ("backtests.strategies.olr.runner", "run_olr_backtest"),
    "portfolio_synergy": ("backtests.strategies.portfolio_synergy.runner", "run_portfolio_synergy_backtest"),
}

OPTIMIZATION_PLUGINS = {
    "kalcb": ("backtests.strategies.kalcb.plugin", "KALCBOptimizationPlugin"),
    "olr": ("backtests.strategies.olr.plugin", "OLROptimizationPlugin"),
    "portfolio_synergy": ("backtests.strategies.portfolio_synergy.plugin", "PortfolioSynergyOptimizationPlugin"),
}


def get_backtest_runner(strategy: str):
    key = strategy.lower()
    if key not in BACKTEST_RUNNERS:
        raise ValueError(f"Unsupported strategy: {strategy}")
    module_name, attr = BACKTEST_RUNNERS[key]
    return getattr(importlib.import_module(module_name), attr)


def create_plugin(strategy: str, config: dict[str, Any] | None = None, **kwargs):
    key = strategy.lower()
    if key not in OPTIMIZATION_PLUGINS:
        raise ValueError(f"Unsupported strategy: {strategy}")
    if key == "kalcb":
        from backtests.strategies.kalcb.fixed_trade_plan_phase import (
            KALCBFixedTradePlanOptimizationPlugin,
            should_use_fixed_trade_plan_phase,
        )

        if should_use_fixed_trade_plan_phase(config, kwargs.get("output_dir")):
            return KALCBFixedTradePlanOptimizationPlugin(config, **kwargs)
    module_name, attr = OPTIMIZATION_PLUGINS[key]
    plugin_cls = getattr(importlib.import_module(module_name), attr)
    return plugin_cls(config, **kwargs)

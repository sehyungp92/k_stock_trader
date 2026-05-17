from __future__ import annotations

import importlib
from typing import Any

BACKTEST_RUNNERS = {
    "kmp": ("backtests.strategies.kmp.runner", "run_kmp_backtest"),
    "kpr": ("backtests.strategies.kpr.runner", "run_kpr_backtest"),
    "nulrimok": ("backtests.strategies.nulrimok.runner", "run_nulrimok_backtest"),
}

OPTIMIZATION_PLUGINS = {
    "kmp": ("backtests.strategies.kmp.plugin", "KMPOptimizationPlugin"),
    "kpr": ("backtests.strategies.kpr.plugin", "KPROptimizationPlugin"),
    "nulrimok": ("backtests.strategies.nulrimok.plugin", "NulrimokOptimizationPlugin"),
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
    module_name, attr = OPTIMIZATION_PLUGINS[key]
    plugin_cls = getattr(importlib.import_module(module_name), attr)
    return plugin_cls(config, **kwargs)

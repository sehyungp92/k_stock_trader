from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {resolved}")
    validate_backtest_config(data)
    return data


def validate_backtest_config(config: dict[str, Any]) -> None:
    if "date_range" in config:
        raw = config["date_range"] or {}
        start = raw.get("start")
        end = raw.get("end")
        if start and end and date.fromisoformat(str(start)) > date.fromisoformat(str(end)):
            raise ValueError("date_range.start must be <= date_range.end")
    if "strategy" in config and str(config["strategy"]).lower() not in {"kalcb", "olr", "portfolio_synergy"}:
        raise ValueError(f"Unsupported strategy in config: {config['strategy']}")


def normalize_runtime_config(strategy: str, config: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(config or {})
    config.setdefault("strategy", strategy)
    config.setdefault("capability_level", "synthetic")
    config.setdefault("initial_equity", 10_000_000.0)
    return config

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def normalize_symbol(symbol: Any) -> str:
    text = str(symbol or "").strip()
    return text.zfill(6) if text else ""


def normalize_sector(sector: Any) -> str:
    text = str(sector or "").strip().upper()
    return text or "UNKNOWN"


def normalize_sector_map(sector_map: Mapping[Any, Any] | None) -> dict[str, str]:
    return {
        normalize_symbol(symbol): normalize_sector(sector)
        for symbol, sector in dict(sector_map or {}).items()
        if normalize_symbol(symbol) and normalize_sector(sector) != "UNKNOWN"
    }


def load_canonical_sector_map(
    config: Mapping[str, Any] | None = None,
    *,
    default_path: str | Path = "config/olr/sector_map.yaml",
    fallback: Mapping[Any, Any] | None = None,
) -> dict[str, str]:
    raw_config = dict(config or {})
    raw = raw_config.get("sector_map")
    if isinstance(raw, str):
        raw = _load_sector_map_from_path(_resolve_config_path(raw))
    elif isinstance(raw, Mapping):
        raw = dict(raw)
    elif raw_config.get("sector_map_path"):
        raw = _load_sector_map_from_path(_resolve_config_path(str(raw_config["sector_map_path"])))
    else:
        path = _resolve_config_path(str(default_path))
        raw = _load_sector_map_from_path(path) if path.exists() else dict(fallback or {})
    return normalize_sector_map(raw)


def _resolve_config_path(raw: str) -> Path:
    path = Path(raw)
    if not path.exists() and not path.is_absolute():
        config_path = Path("config") / path
        if config_path.exists():
            path = config_path
    return path


def _load_sector_map_from_path(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        return {}
    raw = payload.get("sector_map", payload)
    return dict(raw or {}) if isinstance(raw, Mapping) else {}

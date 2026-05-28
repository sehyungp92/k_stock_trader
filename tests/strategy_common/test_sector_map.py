from __future__ import annotations

from pathlib import Path

import yaml

from strategy_common.sector_map import load_canonical_sector_map


def test_olr_sector_map_is_decoupled_from_gamma_config() -> None:
    config_path = Path("config/optimization/olr.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["sector_map_path"] == "olr/sector_map.yaml"

    sector_map = load_canonical_sector_map(config)
    assert sector_map["005930"] == "SEMICONDUCTORS"
    assert sector_map["000660"] == "SEMICONDUCTORS"
    assert len(sector_map) >= 100

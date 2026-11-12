from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from kis_core.input_manifest import InputManifest, frozen_input_loader, signature
from backtests.core.shared_market_inputs import compare_consumed_bars, selected_intraday_paths
from backtests.strategies.kalcb.replay_cache import _load_symbol_frame as kalcb_frame
from backtests.strategies.olr.research_sweep import _load_symbol_frame as olr_frame
from backtests.strategies.portfolio_synergy.source_replay import _load_pinned_parquet_bars, _merge_strategy_bars
from scripts.prepare_olr_kalcb_integrity_inputs import shared_market_contract
from strategy_common.clock import KST
from strategy_common.market import MarketBar


@pytest.fixture
def generation(tmp_path):
    root = tmp_path / "repo"
    bar_root = root / "data/kis_intraday_parquet/036460"
    bar_root.mkdir(parents=True)
    for ending, price in (("20260512", 100), ("20260513", 200)):
        pd.DataFrame([{"timestamp": pd.Timestamp("2026-03-31 09:00", tz=KST),
                       "open": price, "high": price + 1, "low": price - 1,
                       "close": price, "volume": 1000}]).to_parquet(
            bar_root / f"036460_5m_20250512_{ending}.parquet")
    (root / "config/olr").mkdir(parents=True)
    (root / "config/universe_103.yaml").write_text('symbols: ["036460"]')
    (root / "config/olr/sector_map.yaml").write_text('sector_map: {"036460": ENERGY}')
    (root / "kis_core/data").mkdir(parents=True)
    (root / "kis_core/data/krx_intraday_halts.yaml").write_text('evidence_files: {}\nhalts: []\nmarkets: {}\n')
    policy = {"data_root": "data/kis_intraday_parquet", "daily_data_root": "data/krx_daily_parquet",
              "universe": "config/universe_103.yaml", "sector_map_path": "config/olr/sector_map.yaml",
              "timeframe": "5m", "snapshot_end": "20260512", "snapshot_end_overrides": {"036460": "20260513"},
              "train_start": "2025-05-12", "train_end": "2026-03-31", "holdout_start": "2026-04-01", "data_end": "2026-05-12"}
    daily = root / policy["daily_data_root"]
    daily.mkdir(parents=True)
    (daily / "metadata.json").write_text('{}')
    contract, paths = shared_market_contract(root, {"market_inputs": policy})
    # Even when another version is present, consumers must select the declared one.
    paths.update(bar_root.glob("*.parquet"))
    return InputManifest.create(paths, source_root=root, destination=tmp_path / "frozen",
                                evidence={"shared_market_inputs": contract})


def test_all_strategy_loaders_select_same_declared_override(generation):
    root = generation.root / "data/kis_intraday_parquet"
    with generation.activate():
        k = kalcb_frame(root, "036460", "5m", date(2026, 3, 31), data_snapshot_end=date(2026, 5, 12))
        o = olr_frame(root, "036460", "5m", date(2026, 3, 31))
        bars = _load_pinned_parquet_bars({(date(2026, 3, 31), "036460")}, intraday_root=root,
                                        data_snapshot_end="20260512")
    pd.testing.assert_frame_equal(k, o)
    assert k.iloc[0]["close"] == 200
    assert bars[0].close == 200


def test_missing_or_wrong_root_cannot_fall_back(generation):
    with generation.activate():
        with pytest.raises(FileNotFoundError, match="No shared intraday input"):
            selected_intraday_paths(generation.root / "data/kis_intraday_parquet", "999999", "5m")
        with pytest.raises(ValueError, match="root differs"):
            selected_intraday_paths(generation.root / "different", "036460", "5m")


def test_tampered_selected_input_rejected_by_both_loaders(generation):
    selected = generation.root / generation.payload["evidence"]["shared_market_inputs"]["intraday_files"]["036460"][0]
    selected.write_bytes(b"changed")
    with generation.activate():
        for loader in (kalcb_frame, olr_frame):
            with pytest.raises(ValueError, match="bytes changed"):
                loader(generation.root / "data/kis_intraday_parquet", "036460", "5m", date(2026, 3, 31))


def test_shared_config_identity_and_config_relative_path(generation):
    config = {"universe": "universe_103.yaml", "sector_map_path": "olr/sector_map.yaml"}
    bound = generation.remap(config)
    assert Path(bound["universe"]) == generation.root / "config/universe_103.yaml"
    assert Path(bound["sector_map_path"]) == generation.root / "config/olr/sector_map.yaml"
    assert bound["market_input_hash"] == generation.receipt()["market_input_hash"]
    assert bound["holdout_start"] == "2026-04-01"


def test_nested_loader_rejects_different_generation(generation, tmp_path):
    root = Path(generation.payload["source_root"])
    other = InputManifest.create([root / "config/universe_103.yaml"], source_root=root,
                                  destination=tmp_path / "other")
    @frozen_input_loader
    def loader(config):
        pytest.fail("Mismatched generation reached loader")
    with generation.activate(), pytest.raises(ValueError, match="differs from active"):
        loader({"input_manifest_path": str(other.path)})


@pytest.mark.parametrize("field,value", [("open", 101), ("volume", 999), ("is_completed", False)])
def test_overlap_conflict_rejected_before_shared_execution(field, value):
    bar = MarketBar("005930", datetime(2026, 3, 31, 9, tzinfo=KST), "5m", 100, 110, 90, 100, 1000)
    other = replace(bar, **{field: value})
    for check in (compare_consumed_bars, _merge_strategy_bars):
        with pytest.raises(ValueError, match="Conflicting shared market bar"):
            check([bar], [other])


def test_different_strategy_coverage_is_allowed():
    bar = MarketBar("005930", datetime(2026, 3, 31, 9, tzinfo=KST), "5m", 100, 110, 90, 100, 1000)
    result = compare_consumed_bars([bar], [bar, replace(bar, symbol="000660")])
    assert result["overlapping_bars"] == 1 and result["conflicting_bars"] == 0
    assert result["strategies"]["KALCB"]["market_bar_hash"] != result["strategies"]["OLR"]["market_bar_hash"]


def test_market_contract_cannot_reference_different_file_digest(generation):
    contract = generation.payload["evidence"]["shared_market_inputs"]
    name = next(iter(contract["files"]))
    contract["files"][name] = "0" * 64
    contract["market_input_hash"] = signature({k: v for k, v in contract.items() if k != "market_input_hash"})
    with pytest.raises(ValueError, match="not bound to the manifest"):
        generation.receipt()


def test_config_relative_frozen_input_survives_original_removal(generation):
    (Path(generation.payload["source_root"]) / "config/olr/sector_map.yaml").unlink()
    assert Path(generation.remap({"sector_map_path": "olr/sector_map.yaml"})["sector_map_path"]).is_file()


def test_nested_loader_aggregates_parent_read_observations(generation):
    from kis_core.input_manifest import active_manifest, read_input_bytes
    @frozen_input_loader
    def loader(config):
        assert active_manifest() is generation
        return read_input_bytes(config["sector_map_path"])
    with generation.activate():
        loader({"input_manifest_path": str(generation.path)})
    assert any("sector_map.yaml" in path for path in generation.loaded)


@pytest.mark.parametrize("config", [{"end": "2026-05-13"}, {"start": "2025-05-11"},
                                    {"date_range": {"end": "2026-06-01"}}, {"sector_map": {"036460": "different"}}])
def test_shared_contract_rejects_undeclared_date_or_sector_override(generation, config):
    with pytest.raises(ValueError):
        generation.remap(config)


def test_supplement_must_match_archived_session_and_never_overlap(tmp_path):
    import json
    import hashlib
    from scripts.prepare_olr_kalcb_integrity_inputs import verified_supplement
    root = tmp_path / "repo"
    root.mkdir()
    stamps = list(pd.date_range("2026-03-30 09:00", periods=76, freq="5min", tz=KST)) + [pd.Timestamp("2026-03-30 15:30", tz=KST)]
    frame = pd.DataFrame({"timestamp": stamps, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000})
    original = root / "036460_original.parquet"
    original.write_bytes(b"")
    frame.to_parquet(original, index=False)
    archive = InputManifest.create([original], source_root=root, destination=root / "archive")
    derived = root / "036460_supplement.parquet"
    frame.to_parquet(derived, index=False)
    proof = {"source_manifest": str(archive.path), "source_manifest_hash": archive.identity,
        "source_file": original.name, "source_sha256": archive.digest(original),
        "derived_sha256": hashlib.sha256(derived.read_bytes()).hexdigest()}
    provenance = root / "proof.json"
    provenance.write_text(json.dumps(proof))
    entry = {"date": "2026-03-30", "path": derived.name, "provenance": provenance.name}
    primary = root / "primary.parquet"
    later = frame.copy()
    later.timestamp += pd.Timedelta(days=1)
    later.to_parquet(primary, index=False)
    assert verified_supplement(root, "036460", entry, [primary])[0] == derived
    with pytest.raises(ValueError, match="overlaps"):
        verified_supplement(root, "036460", entry, [original])
    changed = frame.copy()
    changed.loc[0, "volume"] = 999
    changed.to_parquet(derived, index=False)
    with pytest.raises(ValueError, match="content"):
        verified_supplement(root, "036460", entry, [primary])

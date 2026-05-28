from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

from backtests.strategies.kalcb import replay_cache as kalcb_replay_cache
from backtests.strategies.olr import replay_cache as olr_replay_cache
from backtests.strategies.olr.research_sweep import OLRResearchSweepDataset
from deployment.olr_kalcb.artifacts import generate_kalcb_daily
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_kalcb.artifact_store import KALCBArtifactStore
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.research import candidate_config_fingerprint
from strategy_kalcb.research import finalize_candidate_snapshot
from strategy_kalcb.research_generator import generate_candidate_snapshot as generate_kalcb_candidate_snapshot
from strategy_olr.config import OLRConfig
from strategy_olr.research import FINAL_CANDIDATE_CONFIG_HASH_VERSION, final_candidate_config_fingerprint
from strategy_olr.research_generator import generate_afternoon_candidate_snapshot, generate_candidate_snapshot as generate_olr_candidate_snapshot

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "live_replay_parity"


def test_kalcb_live_generator_matches_replay_snapshot_fixture(tmp_path):
    trade_date = date(2026, 2, 2)
    manifest = _fixture_manifest("kalcb")
    rows = _daily_fixture(trade_date)
    cfg = KALCBConfig.from_mapping({"kalcb.research.min_adv20_krw": 1_000_000, "kalcb.research.top_long_count": 2})
    sector_map = {"005930": "SEMIS", "000660": "SEMIS", "035420": "INTERNET"}
    config_hash = candidate_config_fingerprint(cfg, {}, sector_map)

    live_final, _result = generate_kalcb_daily(
        rows,
        trade_date,
        config=cfg,
        sector_map=sector_map,
        artifact_root=tmp_path / "kalcb_live_store",
        source_fingerprint=manifest["source_fingerprint"],
        candidate_config_hash=config_hash,
    )
    replay_final = kalcb_replay_cache._load_or_build_snapshot(
        trade_date,
        rows,
        cfg,
        source_fingerprint=manifest["source_fingerprint"],
        candidate_config_hash=config_hash,
        requested_universe_count=len(rows),
        data_available_symbols=sorted(str(symbol).zfill(6) for symbol in rows),
        unavailable_symbols=(),
        sector_map=sector_map,
        store=KALCBArtifactStore(tmp_path / "kalcb_replay_store"),
    )

    assert live_final.artifact_hash == replay_final.artifact_hash
    assert [candidate.symbol for candidate in live_final.candidates] == [candidate.symbol for candidate in replay_final.candidates]
    assert live_final.metadata["artifact_stage"] == "daily_finalized_candidate"
    assert live_final.metadata["source"] == replay_final.metadata["source"] == "real_kis_krx_parquet"


def test_deployment_kalcb_generator_persists_finalized_artifact(tmp_path):
    trade_date = date(2026, 2, 2)
    manifest = _fixture_manifest("kalcb")
    rows = _daily_fixture(trade_date)
    mutations = {"kalcb.research.min_adv20_krw": 1_000_000, "kalcb.research.top_long_count": 2}
    cfg = KALCBConfig.from_mapping(mutations)
    sector_map = {"005930": "SEMIS", "000660": "SEMIS", "035420": "INTERNET"}

    snapshot, result = generate_kalcb_daily(
        rows,
        trade_date,
        config=cfg,
        config_mutations=mutations,
        sector_map=sector_map,
        artifact_root=tmp_path,
        source_fingerprint=manifest["source_fingerprint"],
    )
    loaded = KALCBArtifactStore(tmp_path).load_snapshot(trade_date)

    assert loaded is not None
    assert result.stage == "daily_finalized_candidate"
    assert result.path == tmp_path / f"candidate_snapshot_{trade_date.isoformat()}.json"
    assert loaded.artifact_hash == snapshot.artifact_hash


def test_kalcb_live_generator_uses_replay_config_mutation_hash(tmp_path):
    trade_date = date(2026, 2, 2)
    manifest = _fixture_manifest("kalcb")
    rows = _daily_fixture(trade_date)
    mutations = {
        "kalcb.session.ws_budget": 3,
        "kalcb.frontier.enabled": True,
        "kalcb.frontier.size": 5,
        "kalcb.research.min_adv20_krw": 1_000_000,
        "kalcb.research.top_long_count": 2,
    }
    cfg = KALCBConfig.from_mapping(mutations)
    sector_map = {"005930": "SEMIS", "000660": "SEMIS", "035420": "INTERNET"}
    expected_hash = kalcb_replay_cache._candidate_config_hash(cfg, mutations, sector_map)

    live_final, _result = generate_kalcb_daily(
        rows,
        trade_date,
        config=cfg,
        config_mutations=mutations,
        sector_map=sector_map,
        artifact_root=tmp_path / "kalcb_live_store",
        source_fingerprint=manifest["source_fingerprint"],
    )
    replay_final = kalcb_replay_cache._load_or_build_snapshot(
        trade_date,
        rows,
        cfg,
        source_fingerprint=manifest["source_fingerprint"],
        candidate_config_hash=expected_hash,
        requested_universe_count=len(rows),
        data_available_symbols=sorted(str(symbol).zfill(6) for symbol in rows),
        unavailable_symbols=(),
        sector_map=sector_map,
        store=KALCBArtifactStore(tmp_path / "kalcb_replay_store"),
    )

    assert live_final.metadata["candidate_config_hash"] == expected_hash
    assert replay_final.metadata["candidate_config_hash"] == expected_hash
    assert live_final.artifact_hash == replay_final.artifact_hash


def test_olr_daily_live_generator_matches_replay_snapshot_fixture(tmp_path):
    trade_date = date(2026, 2, 2)
    manifest = _fixture_manifest("olr")
    rows = _daily_fixture(trade_date)
    cfg = OLRConfig.from_mapping({"olr.research.min_adv20_krw": 1_000_000, "olr.research.top_long_count": 2, "olr.signal.daily_min_score": 0.0})

    config_mapping = {"olr.research.min_adv20_krw": 1_000_000, "olr.research.top_long_count": 2, "olr.signal.daily_min_score": 0.0}
    live = generate_olr_candidate_snapshot(rows, trade_date, config=cfg, artifact_root=None, source_fingerprint=manifest["source_fingerprint"])
    dataset = _olr_fixture_dataset(trade_date, rows, _bars_fixture(trade_date), config_mapping, manifest["source_fingerprint"])
    replay = olr_replay_cache._stage1_snapshots(
        dataset,
        {},
        olr_replay_cache._stage1_config_hash(cfg, {}),
        {"artifact_root": str(tmp_path / "olr_stage1_cache")},
    )[trade_date]

    assert live.artifact_hash == replay.artifact_hash
    assert live.metadata["artifact_stage"] == "stage1_daily_candidate"


def test_olr_afternoon_live_generator_matches_replay_snapshot_fixture(tmp_path):
    trade_date = date(2026, 2, 2)
    manifest = _fixture_manifest("olr")
    config_mapping = {
        "olr.research.min_adv20_krw": 1_000_000,
        "olr.research.top_long_count": 2,
        "olr.signal.daily_min_score": 0.0,
        "olr.afternoon.top_n": 1,
        "olr.afternoon.score_mode": "momentum",
    }
    cfg = OLRConfig.from_mapping(config_mapping)
    daily = generate_olr_candidate_snapshot(
        _daily_fixture(trade_date),
        trade_date,
        config=cfg,
        artifact_root=None,
        source_fingerprint=manifest["source_fingerprint"],
    )
    bars = _bars_fixture(trade_date)

    live = generate_afternoon_candidate_snapshot(daily, bars, config=cfg, artifact_root=tmp_path)
    dataset = _olr_fixture_dataset(trade_date, _daily_fixture(trade_date), bars, config_mapping, manifest["source_fingerprint"])
    replay = olr_replay_cache._load_or_build_stage2_snapshots(
        dataset,
        cfg,
        {},
        (trade_date,),
        olr_replay_cache._stage1_config_hash(cfg, {}),
        olr_replay_cache._candidate_config_hash(cfg, {}),
        {"artifact_root": str(tmp_path / "replay_cache")},
    )[trade_date]

    expected_config_hash = final_candidate_config_fingerprint(cfg)
    assert live.artifact_hash == replay.artifact_hash
    assert live.metadata["artifact_stage"] == "final_afternoon_1430"
    assert live.metadata["candidate_config_hash"] == replay.metadata["candidate_config_hash"] == expected_config_hash
    assert live.metadata["final_candidate_config_hash"] == replay.metadata["final_candidate_config_hash"] == expected_config_hash
    assert live.metadata["final_candidate_config_hash_version"] == FINAL_CANDIDATE_CONFIG_HASH_VERSION


def test_artifact_hash_changes_when_fixture_input_or_config_changes():
    trade_date = date(2026, 2, 2)
    rows = _daily_fixture(trade_date)
    cfg = KALCBConfig.from_mapping({"kalcb.research.min_adv20_krw": 1_000_000, "kalcb.research.top_long_count": 2})
    changed_rows = {key: [dict(row) for row in value] for key, value in rows.items()}
    changed_rows["005930"][-1]["close"] += 100
    changed_cfg = KALCBConfig.from_mapping({"kalcb.research.min_adv20_krw": 1_000_000, "kalcb.research.top_long_count": 1})

    base = generate_kalcb_candidate_snapshot(rows, trade_date, config=cfg, artifact_root=None)
    changed_input = generate_kalcb_candidate_snapshot(changed_rows, trade_date, config=cfg, artifact_root=None)
    changed_config = generate_kalcb_candidate_snapshot(rows, trade_date, config=changed_cfg, artifact_root=None)

    assert changed_input.artifact_hash != base.artifact_hash
    assert changed_config.artifact_hash != base.artifact_hash


def test_olr_afternoon_artifact_hash_changes_when_final_config_fingerprint_changes():
    trade_date = date(2026, 2, 2)
    manifest = _fixture_manifest("olr")
    base_mapping = {
        "olr.research.min_adv20_krw": 1_000_000,
        "olr.research.top_long_count": 2,
        "olr.signal.daily_min_score": 0.0,
        "olr.afternoon.top_n": 1,
        "olr.afternoon.score_mode": "momentum",
    }
    changed_mapping = {**base_mapping, "olr.afternoon.max_score": 999_998.0}
    base_cfg = OLRConfig.from_mapping(base_mapping)
    changed_cfg = OLRConfig.from_mapping(changed_mapping)
    daily = generate_olr_candidate_snapshot(
        _daily_fixture(trade_date),
        trade_date,
        config=base_cfg,
        artifact_root=None,
        source_fingerprint=manifest["source_fingerprint"],
    )
    bars = _bars_fixture(trade_date)

    base = generate_afternoon_candidate_snapshot(daily, bars, config=base_cfg, artifact_root=None)
    changed = generate_afternoon_candidate_snapshot(daily, bars, config=changed_cfg, artifact_root=None)

    assert [candidate.symbol for candidate in changed.candidates] == [candidate.symbol for candidate in base.candidates]
    assert changed.metadata["final_candidate_config_hash"] != base.metadata["final_candidate_config_hash"]
    assert changed.artifact_hash != base.artifact_hash


def _daily_fixture(trade_date: date) -> dict[str, list[dict]]:
    return {
        "005930": _rows(trade_date, start=5_000, drift=45),
        "000660": _rows(trade_date, start=4_800, drift=28),
        "035420": _rows(trade_date, start=4_000, drift=15),
    }


def _rows(trade_date: date, *, start: float, drift: float, days: int = 80) -> list[dict]:
    first = trade_date - timedelta(days=days)
    rows = []
    for index in range(days):
        day = first + timedelta(days=index)
        close = start + drift * index
        rows.append(
            {
                "date": day.isoformat(),
                "open": close - max(drift * 0.5, 1.0),
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000,
            }
        )
    return rows


def _bars_fixture(trade_date: date) -> dict[tuple[date, str], tuple[MarketBar, ...]]:
    return {
        (trade_date, "005930"): (
            _bar("005930", trade_date, 9, 0, 100.0, 103.0, 99.0, 102.0),
            _bar("005930", trade_date, 14, 25, 102.0, 105.0, 101.0, 104.0),
        ),
        (trade_date, "000660"): (
            _bar("000660", trade_date, 9, 0, 100.0, 101.0, 99.0, 100.5),
            _bar("000660", trade_date, 14, 25, 100.5, 101.0, 100.0, 100.7),
        ),
    }


def _bar(symbol: str, trade_date: date, hour: int, minute: int, open_: float, high: float, low: float, close: float) -> MarketBar:
    return MarketBar(
        symbol=symbol,
        timestamp=datetime(trade_date.year, trade_date.month, trade_date.day, hour, minute, tzinfo=KST),
        timeframe="5m",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=10_000,
        is_completed=True,
        source="fixture",
        source_fingerprint="fixture-source",
    )


def _fixture_manifest(strategy_id: str) -> dict:
    return json.loads((FIXTURE_ROOT / strategy_id / "manifest.json").read_text(encoding="utf-8"))


def _olr_fixture_dataset(
    trade_date: date,
    rows: dict[str, list[dict]],
    bars: dict[tuple[date, str], tuple[MarketBar, ...]],
    config: dict,
    source_fingerprint: str,
) -> OLRResearchSweepDataset:
    symbols = tuple(sorted(rows))
    intraday_symbols = {symbol for _day, symbol in bars}
    return OLRResearchSweepDataset(
        config=dict(config),
        source_fingerprint=source_fingerprint,
        daily_source_fingerprint=source_fingerprint,
        intraday_source_fingerprint=source_fingerprint,
        data_root=FIXTURE_ROOT,
        daily_data_root=FIXTURE_ROOT,
        timeframe="5m",
        symbols=symbols,
        requested_symbols=symbols,
        excluded_symbols={},
        intraday_available_symbols=tuple(sorted(intraday_symbols)),
        intraday_unavailable_symbols=tuple(symbol for symbol in symbols if symbol not in intraday_symbols),
        daily_by_symbol=rows,
        flow_by_symbol={},
        foreign_flow_by_symbol={},
        institutional_flow_by_symbol={},
        index_by_code={},
        sector_map={},
        trading_dates=(trade_date,),
        bars_by_key=bars,
        train_start=trade_date,
        train_end=trade_date,
        holdout_start=trade_date + timedelta(days=1),
        coverage_report={},
    )

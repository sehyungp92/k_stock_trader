from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from backtests.strategies.kalcb import replay_cache
from strategy_kalcb.artifact_store import KALCBArtifactStore
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.models import KALCBDailySnapshot
from strategy_kalcb.research import (
    KALCB_FINAL_ARTIFACT_STAGE,
    build_research_snapshot,
    candidate_config_fingerprint,
    daily_selection_from_snapshot,
    finalize_candidate_snapshot,
    run_daily_selection,
)
from strategy_kalcb.research_generator import generate_candidate_snapshot, generate_finalized_candidate_snapshot


def test_research_filters_liquid_clean_names_and_selects_top_longs():
    trade_date = date(2026, 1, 31)
    daily_by_symbol = {
        "005930": _rows(trade_date, start=5_000, drift=40, volume=1_000_000),
        "000660": _rows(trade_date, start=4_500, drift=25, volume=900_000),
        "000001": _rows(trade_date, start=500, drift=10, volume=1_000_000),
        "000002": _rows(trade_date, start=5_000, drift=20, volume=100),
        "000003": _rows(trade_date, start=5_000, drift=20, volume=1_000_000, bad_last=True),
    }
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.research.top_long_count": 2,
            "kalcb.research.min_price_krw": 1_000,
            "kalcb.research.min_adv20_krw": 2_000_000_000,
        }
    )

    research = build_research_snapshot(daily_by_symbol, trade_date, cfg, sector_map={"005930": "semis", "000660": "semis"})
    selected = daily_selection_from_snapshot(research, cfg)

    assert [candidate.symbol for candidate in selected.candidates] == ["005930", "000660"]
    assert selected.metadata["candidate_pool_count"] == 2
    assert set(selected.metadata["rejected_symbols"]) == {"000001", "000002", "000003"}
    assert selected.candidates[0].sector == "SEMIS"
    assert selected.candidates[0].selection_score > selected.candidates[1].selection_score
    assert selected.candidates[0].metadata["research_config_hash"]
    assert selected.candidates[0].metadata["sector_map_hash"]


def test_research_uses_strictly_prior_completed_daily_rows():
    trade_date = date(2026, 1, 31)
    same_day_spike = _rows(trade_date, start=8_000, drift=-10, volume=1_000_000)
    same_day_spike.append({"date": trade_date.isoformat(), "open": 20_000, "high": 21_000, "low": 19_900, "close": 21_000, "volume": 10_000_000})
    steady_leader = _rows(trade_date, start=5_000, drift=45, volume=1_000_000)
    cfg = KALCBConfig.from_mapping({"kalcb.research.top_long_count": 1})

    research = build_research_snapshot({"005930": same_day_spike, "000660": steady_leader}, trade_date, cfg)
    selected = daily_selection_from_snapshot(research, cfg)

    assert research.symbols["005930"].prior_day_close != 21_000
    assert selected.candidates[0].symbol == "000660"


def test_research_default_fingerprint_is_causal_and_skips_bad_dates():
    trade_date = date(2026, 1, 31)
    base_rows = _rows(trade_date, start=5_000, drift=40, volume=1_000_000)
    noisy_rows = list(base_rows) + [
        {"date": trade_date.isoformat(), "open": 20_000, "high": 21_000, "low": 19_900, "close": 21_000, "volume": 10_000_000},
        {"date": "not-a-date", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
    ]
    cfg = KALCBConfig.from_mapping({"kalcb.research.min_adv20_krw": 1_000_000})

    clean = build_research_snapshot({"005930": base_rows}, trade_date, cfg)
    noisy = build_research_snapshot({"005930": noisy_rows}, trade_date, cfg)
    corrected_prior = [dict(row) for row in base_rows]
    corrected_prior[-1]["close"] += 10
    corrected = build_research_snapshot({"005930": corrected_prior}, trade_date, cfg)

    assert noisy.source_fingerprint == clean.source_fingerprint
    assert corrected.source_fingerprint != clean.source_fingerprint
    assert noisy.symbols["005930"].prior_day_close == clean.symbols["005930"].prior_day_close
    assert noisy.metadata["research_lookahead_policy"] == "strict_prior_daily_rows_only"
    assert noisy.metadata["research_as_of_date"] == (trade_date - timedelta(days=1)).isoformat()
    assert noisy.metadata["research_causal_source_fingerprint"] == clean.metadata["research_causal_source_fingerprint"]


def test_research_accepts_trade_date_key_in_prepared_daily_rows():
    trade_date = date(2026, 1, 31)
    rows = _rows(trade_date, start=5_000, drift=40, volume=1_000_000)
    for row in rows:
        row["trade_date"] = row.pop("date")
    cfg = KALCBConfig.from_mapping({"kalcb.research.top_long_count": 1, "kalcb.research.min_adv20_krw": 1_000_000})

    research = build_research_snapshot({"005930": rows}, trade_date, cfg)
    selected = daily_selection_from_snapshot(research, cfg)

    assert selected.candidates[0].metadata["prior_day_date"] == (trade_date - timedelta(days=1)).isoformat()
    assert selected.candidates[0].metadata["research_as_of_date"] == (trade_date - timedelta(days=1)).isoformat()


def test_research_scores_sector_participation_and_accumulation():
    trade_date = date(2026, 1, 31)
    cfg = KALCBConfig.from_mapping({"kalcb.research.top_long_count": 3})
    daily_by_symbol = {
        "005930": _rows(trade_date, start=5_000, drift=35, volume=1_000_000, last_volume_mult=2.0),
        "000660": _rows(trade_date, start=5_000, drift=30, volume=1_000_000, last_volume_mult=2.0),
        "035420": _rows(trade_date, start=5_000, drift=28, volume=1_000_000, last_volume_mult=0.5),
    }

    research = build_research_snapshot(
        daily_by_symbol,
        trade_date,
        cfg,
        sector_map={"005930": "semis", "000660": "semis", "035420": "internet"},
    )
    selected = daily_selection_from_snapshot(research, cfg)
    semis = [candidate for candidate in selected.candidates if candidate.sector == "SEMIS"]

    assert research.sectors["SEMIS"].participation == pytest.approx(1.0)
    assert semis
    assert semis[0].metadata["research_score_components"]["sector_participation_score"] == pytest.approx(62.5)
    assert semis[0].accumulation_score > 0


def test_research_daily_sector_score_metadata_and_floor_alias():
    trade_date = date(2026, 1, 31)
    daily_by_symbol = {
        "005930": _rows(trade_date, start=5_000, drift=35, volume=1_000_000),
        "000660": _rows(trade_date, start=5_000, drift=25, volume=1_000_000),
        "035420": _rows(trade_date, start=5_000, drift=5, volume=1_000_000),
    }
    cfg = KALCBConfig.from_mapping({"kalcb.research.top_long_count": 3, "kalcb.research.min_adv20_krw": 1_000_000})
    gated = KALCBConfig.from_mapping(
        {
            "kalcb.research.top_long_count": 3,
            "kalcb.research.min_adv20_krw": 1_000_000,
            "kalcb.research.min_sector_daily_score_pct": 101.0,
        }
    )

    selected = daily_selection_from_snapshot(
        build_research_snapshot(daily_by_symbol, trade_date, cfg, sector_map={"005930": "SEMIS", "000660": "SEMIS", "035420": "INTERNET"}),
        cfg,
    )
    rejected = daily_selection_from_snapshot(
        build_research_snapshot(daily_by_symbol, trade_date, gated, sector_map={"005930": "SEMIS", "000660": "SEMIS", "035420": "INTERNET"}),
        gated,
    )

    assert selected.candidates
    assert "sector_daily_score_pct" in selected.candidates[0].metadata
    assert gated.research_min_sector_daily_score_pct == 101.0
    assert rejected.candidates == ()
    assert all("sector_daily_score_below_floor" in reasons for reasons in rejected.metadata["rejected_symbols"].values())


def test_research_emits_structural_campaign_metadata_and_active_overflow_split():
    trade_date = date(2026, 1, 31)
    daily_by_symbol = {
        f"{index:06d}": _rows(trade_date, start=5_000 + index * 10, drift=10 + index, volume=1_000_000)
        for index in range(1, 13)
    }
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.session.ws_budget": 3,
            "kalcb.research.top_long_count": 2,
            "kalcb.research.structural_frontier_count": 5,
            "kalcb.research.min_adv20_krw": 1_000_000,
        }
    )

    selected = daily_selection_from_snapshot(build_research_snapshot(daily_by_symbol, trade_date, cfg), cfg)

    assert len(selected.candidates) == 5
    assert len(selected.metadata["active_symbols"]) == 3
    assert len(selected.metadata["frontier_symbols"]) == 5
    assert selected.metadata["overflow_symbol_count"] == 9
    assert selected.candidates[0].metadata["structural_campaign"]["score_uses_ex_post_labels"] is False
    assert selected.candidates[0].metadata["structural_source_role"] == "active"
    assert selected.candidates[3].metadata["structural_source_role"] == "overflow"
    assert selected.candidates[0].selection_score == pytest.approx(selected.candidates[0].metadata["structural_campaign_score"] * 10.0)


def test_research_structural_flow_uses_prior_rows_and_ignores_same_day_flow():
    trade_date = date(2026, 1, 31)
    rows = {"005930": _rows(trade_date, start=5_000, drift=35, volume=1_000_000)}
    prior_flow = _flow_rows(trade_date, tail_value=3_000_000.0)
    polluted_flow = [*prior_flow, {"date": trade_date.isoformat(), "foreign_net": -99_000_000.0, "inst_net": -99_000_000.0}]
    changed_prior = [*prior_flow]
    changed_prior[-1] = {**changed_prior[-1], "foreign_net": -99_000_000.0, "inst_net": -99_000_000.0}
    cfg = KALCBConfig.from_mapping({"kalcb.research.min_adv20_krw": 1_000_000, "kalcb.research.top_long_count": 1})

    base = daily_selection_from_snapshot(build_research_snapshot(rows, trade_date, cfg, daily_flow_by_symbol={"005930": prior_flow}), cfg)
    polluted = daily_selection_from_snapshot(build_research_snapshot(rows, trade_date, cfg, daily_flow_by_symbol={"005930": polluted_flow}), cfg)
    changed = daily_selection_from_snapshot(build_research_snapshot(rows, trade_date, cfg, daily_flow_by_symbol={"005930": changed_prior}), cfg)

    assert polluted.source_fingerprint == base.source_fingerprint
    assert polluted.candidates[0].metadata["accumulation_score_pct"] == pytest.approx(base.candidates[0].metadata["accumulation_score_pct"])
    assert changed.source_fingerprint != base.source_fingerprint
    assert changed.candidates[0].metadata["accumulation_score_pct"] != pytest.approx(base.candidates[0].metadata["accumulation_score_pct"])


def test_research_weight_and_soft_gate_aliases_are_tunable():
    trade_date = date(2026, 1, 31)
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.research.weights.relative_strength": 0.50,
            "kalcb.research.weights.daily_trend": 0.20,
            "kalcb.research.weights.compression": 0.10,
            "kalcb.research.weights.accumulation": 0.10,
            "kalcb.research.weights.stock_regime": 0.05,
            "kalcb.research.weights.sector_regime": 0.03,
            "kalcb.research.weights.sector_participation": 0.02,
            "kalcb.research.min_rs_percentile": 60.0,
            "kalcb.research.min_adv20_krw": 1_000_000,
        }
    )
    daily_by_symbol = {
        "005930": _rows(trade_date, start=5_000, drift=35, volume=1_000_000),
        "000660": _rows(trade_date, start=5_000, drift=-5, volume=1_000_000),
    }

    selected = daily_selection_from_snapshot(build_research_snapshot(daily_by_symbol, trade_date, cfg), cfg)

    assert selected.candidates
    assert [candidate.symbol for candidate in selected.candidates] == ["005930"]
    assert "rs_below_floor" in selected.metadata["rejected_symbols"]["000660"]


def test_run_daily_selection_accepts_rows_or_prebuilt_snapshot():
    trade_date = date(2026, 1, 31)
    daily_by_symbol = {
        "005930": _rows(trade_date, start=5_000, drift=35, volume=1_000_000),
        "000660": _rows(trade_date, start=5_000, drift=20, volume=1_000_000),
    }
    cfg = KALCBConfig.from_mapping({"kalcb.research.top_long_count": 2, "kalcb.research.min_adv20_krw": 1_000_000})
    research = build_research_snapshot(daily_by_symbol, trade_date, cfg, source_fingerprint="provided-source")

    from_rows = run_daily_selection(daily_by_symbol, trade_date, config=cfg, artifact_root=None, source_fingerprint="provided-source")
    from_snapshot = run_daily_selection(research, config=cfg, artifact_root=None)

    assert from_snapshot.artifact_hash == from_rows.artifact_hash
    assert from_snapshot.source_fingerprint == "provided-source"
    assert [candidate.symbol for candidate in from_snapshot.candidates] == [candidate.symbol for candidate in from_rows.candidates]


def test_run_daily_selection_rejects_raw_rows_without_trade_date():
    with pytest.raises(ValueError, match="requires trade_date"):
        run_daily_selection({"005930": []}, artifact_root=None)


def test_low_level_kalcb_generator_is_non_persistent_by_default(tmp_path, monkeypatch):
    trade_date = date(2026, 1, 31)
    rows = {"005930": _rows(trade_date, start=5_000, drift=35, volume=1_000_000)}
    cfg = KALCBConfig.from_mapping({"kalcb.research.top_long_count": 1, "kalcb.research.min_adv20_krw": 1_000_000})

    monkeypatch.chdir(tmp_path)
    snapshot = generate_candidate_snapshot(rows, trade_date, config=cfg, source_fingerprint="provided-source")

    assert snapshot.metadata.get("artifact_stage") != KALCB_FINAL_ARTIFACT_STAGE
    assert not (tmp_path / "data" / "strategy" / "kalcb" / f"candidate_snapshot_{trade_date.isoformat()}.json").exists()
    with pytest.raises(ValueError, match="does not persist"):
        generate_candidate_snapshot(rows, trade_date, config=cfg, artifact_root=tmp_path / "bad_store")
    assert not (tmp_path / "bad_store").exists()
    with pytest.raises(ValueError, match="finalized executable"):
        KALCBArtifactStore(tmp_path / "store").save_snapshot(snapshot)
    assert not (tmp_path / "store").exists()


def test_finalized_kalcb_generator_persists_executable_artifact(tmp_path):
    trade_date = date(2026, 1, 31)
    rows = {"005930": _rows(trade_date, start=5_000, drift=35, volume=1_000_000)}
    mutations = {"kalcb.research.top_long_count": 1, "kalcb.research.min_adv20_krw": 1_000_000}
    cfg = KALCBConfig.from_mapping(mutations)
    store_root = tmp_path / "store"

    snapshot = generate_finalized_candidate_snapshot(
        rows,
        trade_date,
        config=cfg,
        config_mutations=mutations,
        sector_map={"005930": "SEMIS"},
        artifact_root=store_root,
        source_fingerprint="provided-source",
    )
    loaded = KALCBArtifactStore(store_root).load_snapshot(trade_date)

    assert loaded is not None
    assert snapshot.metadata["artifact_stage"] == KALCB_FINAL_ARTIFACT_STAGE
    assert snapshot.metadata["candidate_config_hash"]
    assert loaded.artifact_hash == snapshot.artifact_hash
    assert loaded.source_fingerprint == "provided-source"


def test_finalized_kalcb_generator_requires_config_hash_source_for_explicit_config(tmp_path):
    trade_date = date(2026, 1, 31)
    rows = {"005930": _rows(trade_date, start=5_000, drift=35, volume=1_000_000)}
    cfg = KALCBConfig.from_mapping({"kalcb.research.top_long_count": 1})

    with pytest.raises(ValueError, match="candidate_config_hash or config_mutations"):
        generate_finalized_candidate_snapshot(
            rows,
            trade_date,
            config=cfg,
            sector_map={"005930": "SEMIS"},
            artifact_root=tmp_path / "store",
            source_fingerprint="provided-source",
        )


def test_finalized_kalcb_generator_rejects_mismatched_config_hash(tmp_path):
    trade_date = date(2026, 1, 31)
    rows = {"005930": _rows(trade_date, start=5_000, drift=35, volume=1_000_000)}
    mutations = {"kalcb.research.top_long_count": 1}
    cfg = KALCBConfig.from_mapping(mutations)

    with pytest.raises(ValueError, match="does not match config_mutations"):
        generate_finalized_candidate_snapshot(
            rows,
            trade_date,
            config=cfg,
            config_mutations=mutations,
            candidate_config_hash="wrong-hash",
            sector_map={"005930": "SEMIS"},
            artifact_root=tmp_path / "store",
            source_fingerprint="provided-source",
        )


def test_empty_finalized_kalcb_hash_binds_finalization_metadata():
    trade_date = date(2026, 1, 31)
    base = KALCBDailySnapshot(
        trade_date=trade_date,
        candidates=(),
        source_fingerprint="same-source",
        generated_at=datetime(2026, 1, 31, tzinfo=timezone.utc),
        metadata={
            "artifact_stage": KALCB_FINAL_ARTIFACT_STAGE,
            "candidate_config_hash": "config-a",
            "source": "real_kis_krx_parquet",
            "requested_universe_count": 10,
            "data_available_symbol_count": 0,
            "source_universe_count": 10,
            "candidate_pool_count": 0,
        },
    )
    changed = KALCBDailySnapshot(
        trade_date=trade_date,
        candidates=(),
        source_fingerprint=base.source_fingerprint,
        generated_at=base.generated_at,
        metadata={**base.metadata, "candidate_config_hash": "config-b"},
    )

    assert base.artifact_hash != changed.artifact_hash


def test_finalize_candidate_snapshot_matches_replay_frontier_materialization():
    trade_date = date(2026, 1, 31)
    daily_by_symbol = {
        f"{index:06d}": _rows(trade_date, start=5_000 + index * 10, drift=10 + index, volume=1_000_000)
        for index in range(1, 10)
    }
    cfg = KALCBConfig.from_mapping(
        {
            "kalcb.session.ws_budget": 3,
            "kalcb.frontier.enabled": True,
            "kalcb.frontier.size": 5,
            "kalcb.research.top_long_count": 2,
            "kalcb.research.structural_frontier_count": 7,
            "kalcb.research.min_adv20_krw": 1_000_000,
        }
    )
    sector_map = {symbol: "SEMIS" for symbol in daily_by_symbol}
    base = daily_selection_from_snapshot(build_research_snapshot(daily_by_symbol, trade_date, cfg, sector_map=sector_map), cfg)
    config_hash = candidate_config_fingerprint(cfg, {}, sector_map)

    finalized = finalize_candidate_snapshot(
        base,
        config=cfg,
        candidate_config_hash=config_hash,
        source="real_kis_krx_parquet",
        sector_map_hash=replay_cache.stable_signature(sector_map),
        requested_universe_count=len(daily_by_symbol),
        data_available_symbols=sorted(daily_by_symbol),
        unavailable_symbols=(),
        source_universe_count=len(daily_by_symbol),
    )
    active_seed = replay_cache._select_active_seed(list(base.candidates), cfg)
    expected_frontier = replay_cache._build_frontier_order(list(base.candidates), active_seed, cfg, 5)

    assert [candidate.symbol for candidate in finalized.candidates] == [candidate.symbol for candidate in expected_frontier]
    assert finalized.metadata["artifact_stage"] == KALCB_FINAL_ARTIFACT_STAGE
    assert finalized.metadata["candidate_config_hash"] == config_hash
    assert finalized.metadata["active_symbol_count"] == len(active_seed)
    assert replay_cache._candidate_config_hash(cfg, {}, sector_map) == config_hash


def _rows(
    trade_date: date,
    *,
    start: float,
    drift: float,
    volume: float,
    days: int = 70,
    bad_last: bool = False,
    last_volume_mult: float = 1.0,
) -> list[dict]:
    first = trade_date - timedelta(days=days)
    rows: list[dict] = []
    for index in range(days):
        current = first + timedelta(days=index)
        close = start + drift * index
        open_ = close - max(drift * 0.5, 1.0)
        high = close * 1.01
        low = open_ * 0.99
        row_volume = volume * (last_volume_mult if index == days - 1 else 1.0)
        rows.append(
            {
                "date": current.isoformat(),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": row_volume,
            }
        )
    if bad_last:
        rows[-1]["high"] = rows[-1]["low"] - 1
    return rows


def _flow_rows(trade_date: date, *, tail_value: float) -> list[dict]:
    start = trade_date - timedelta(days=40)
    rows: list[dict] = []
    for index in range(35):
        value = tail_value if index >= 30 else 100_000.0 * ((index % 3) - 1)
        rows.append({"date": (start + timedelta(days=index)).isoformat(), "foreign_net": value, "inst_net": value * 0.5})
    return rows

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import pytest

from backtests.strategies.olr import allocation_holdout_eval, allocation_sweep
from backtests.strategies.olr.allocation_sweep import (
    ALLOCATION_OFFICIAL_AUDIT_CACHE_VERSION,
    ALLOCATION_OFFICIAL_AUDIT_EVIDENCE_VERSION,
    OFFICIAL_AUCTION_EXIT_RECOVERY_SESSIONS,
    _annotate_proxy_official_ranks,
    _expanded_audit_count,
    _official_followup_session_dates,
    _key_summary,
    _official_replay_pass,
    _official_trade_plan_reject_reason,
    _proxy_official_rank_diagnostics,
)
from backtests.strategies.olr.trade_plan_sweep import CandidateSource, OLRTradePlanSpec, _name_plan
from strategy_common.clock import KST
from strategy_olr.config import OLRConfig
from strategy_olr.execution import OLRAllocationPlan, OLREntryPlan, OLRExitPlan, OLRTradeOutcome, summarize_olr_portfolio_proxy


def test_olr_audit_rows_record_proxy_and_official_rank_changes() -> None:
    rows = [
        {
            "name": "proxy_first_official_last",
            "official_mtm_metrics": {"official_mtm_net_return_pct": 0.01},
            "official_proxy_abs_net_delta": 0.04,
        },
        {
            "name": "proxy_second_official_first",
            "official_mtm_metrics": {"official_mtm_net_return_pct": 0.05},
            "official_proxy_abs_net_delta": 0.02,
        },
    ]

    _annotate_proxy_official_ranks(rows, {"proxy_first_official_last": 1, "proxy_second_official_first": 2})
    diagnostics = _proxy_official_rank_diagnostics(rows)

    assert rows[0]["proxy_rank"] == 1
    assert rows[0]["official_rank"] == 2
    assert rows[0]["rank_change"] == 1
    assert diagnostics["proxy_official_rank_correlation"] == pytest.approx(-1.0)


def test_olr_audit_coverage_expands_when_proxy_alignment_is_weak() -> None:
    diagnostics = {"count": 4, "proxy_official_rank_correlation": 0.10, "max_official_proxy_abs_net_delta": 0.01}

    expanded = _expanded_audit_count(4, 12, diagnostics)

    assert expanded > 4
    assert diagnostics["expansion_reason"] == "weak_proxy_official_rank_correlation"


def test_olr_official_audit_key_summary_is_stable_and_explicit() -> None:
    summary = _key_summary(["2026-01-02|B", "2026-01-01|A"])

    assert summary["count"] == 2
    assert summary["keys"] == ["2026-01-01|A", "2026-01-02|B"]
    assert summary["hash"]
    assert ALLOCATION_OFFICIAL_AUDIT_CACHE_VERSION.endswith("v10")
    assert ALLOCATION_OFFICIAL_AUDIT_EVIDENCE_VERSION.endswith("v2")


def test_olr_holdout_audit_records_selected_order_fill_nonfill_keys() -> None:
    decision = SimpleNamespace(
        strategy_id="OLR",
        decision_code="ENTRY_SUBMITTED",
        timestamp=datetime(2026, 1, 5, 14, 30, tzinfo=KST),
        symbol="005930",
        reason="entry",
        metadata={"qty": 10, "candidate_rank": 1, "source_artifact_hash": "artifact-a"},
    )
    fill = SimpleNamespace(
        strategy_id="OLR",
        timestamp=datetime(2026, 1, 5, 15, 30, tzinfo=KST),
        side="BUY",
        symbol="005930",
        reason="entry",
        qty=10,
        price=100.0,
        metadata={"source_artifact_hash": "artifact-a"},
    )
    nonfill = SimpleNamespace(
        strategy_id="OLR",
        submitted_at=datetime(2026, 1, 6, 15, 30, tzinfo=KST),
        side="SELL",
        order_type="CLOSE_AUCTION",
        symbol="005930",
        reason="exit",
        qty=10,
        metadata={"source_artifact_hash": "artifact-a", "auction_nonfill_key": "nf-a"},
    )
    result = SimpleNamespace(
        decisions=[decision],
        replay_result=SimpleNamespace(
            broker=SimpleNamespace(
                fills=[fill],
                rejected_orders=[],
                expired_orders=[nonfill],
                orders=[],
                positions={},
            )
        ),
    )

    evidence = allocation_holdout_eval._official_audit_key_evidence(result)

    assert evidence["selected_candidate_keys"]["count"] == 1
    assert evidence["submitted_order_keys"]["count"] == 1
    assert evidence["fill_keys"]["count"] == 1
    assert evidence["nonfill_order_keys"]["count"] == 1


def test_olr_official_audit_uses_train_only_followup_sessions_for_auction_recovery() -> None:
    sessions = [date(2026, 1, day) for day in range(5, 14)]
    followups = _official_followup_session_dates(sessions, date(2026, 1, 6))

    assert followups == tuple(sessions[2 : 2 + OFFICIAL_AUCTION_EXIT_RECOVERY_SESSIONS])
    assert date(2026, 1, 14) not in followups


def test_olr_official_replay_gate_rejects_stale_orders_or_unclosed_positions() -> None:
    clean = {
        "official_trade_plan_supported": 1.0,
        "same_bar_fill_count": 0.0,
        "end_open_position_count": 0.0,
        "open_order_count": 0.0,
        "entry_fill_count": 10.0,
        "exit_fill_count": 10.0,
    }

    assert _official_replay_pass(clean)
    assert not _official_replay_pass({**clean, "open_order_count": 1.0})
    assert not _official_replay_pass({**clean, "end_open_position_count": 1.0})
    assert not _official_replay_pass({**clean, "exit_fill_count": 9.0})
    assert not _official_replay_pass({**clean, "official_trade_plan_supported": 0.0})


def test_olr_official_trade_plan_supports_full_managed_exit_plan() -> None:
    source_name = "src1_stage2"
    entry = OLREntryPlan("confirm", "confirm_next_bar")
    simple_managed = OLRTradePlanSpec(
        "",
        source_name,
        entry,
        OLRExitPlan("managed_decision_low", mode="managed", stop_mode="decision_low", hard_stop_enabled=True),
    )
    target_managed = OLRTradePlanSpec(
        "",
        source_name,
        entry,
        OLRExitPlan("managed_target", mode="managed", hard_stop_enabled=True, target_r=1.0),
    )

    assert _official_trade_plan_reject_reason(simple_managed) == ""
    assert _official_trade_plan_reject_reason(target_managed) == ""


def test_olr_allocation_rows_consume_swept_trade_plan_spec() -> None:
    source = CandidateSource(
        rank=1,
        name="src1_stage2",
        stage1_name="stage1",
        stage2_name="stage2",
        score=1.0,
        mutations={},
    )
    spec = _name_plan(
        OLRTradePlanSpec(
            "",
            source.name,
            OLREntryPlan("confirm_next_bar_test", "confirm_next_bar"),
            OLRExitPlan("next_close_test", mode="next_close", hard_stop_enabled=False),
        )
    )
    compiled = SimpleNamespace(
        selection_counts_by_source={source.name: {}},
        selections_by_source={source.name: ()},
        eligible_dates=(date(2026, 1, 5),),
        next_session_by_date={},
        dataset=SimpleNamespace(config={"initial_equity": 1_000_000.0}),
    )

    rows = allocation_sweep._evaluate_source(
        source,
        compiled,
        OLRConfig(),
        [],
        [OLRAllocationPlan("fixed_slots")],
        {},
        trade_specs_by_source={source.name: (spec,)},
    )

    assert rows[0].trade_spec.entry.mode == "confirm_next_bar"
    assert rows[0].trade_spec.exit.mode == "next_close"
    assert rows[0].name.startswith(spec.name)
    assert "__close_auction_next_close__" not in rows[0].name


def test_olr_portfolio_proxy_carries_overnight_cash_forward() -> None:
    first = _olr_outcome(
        date(2026, 1, 5),
        "005930",
        datetime(2026, 1, 5, 14, 35),
        datetime(2026, 1, 6, 15, 30),
        rank=1,
        gross_return=0.10,
    )
    overlapping = _olr_outcome(
        date(2026, 1, 6),
        "000660",
        datetime(2026, 1, 6, 14, 35),
        datetime(2026, 1, 7, 15, 30),
        rank=1,
        gross_return=0.10,
    )

    metrics = summarize_olr_portfolio_proxy(
        [first, overlapping],
        session_dates=(date(2026, 1, 5), date(2026, 1, 6)),
        selection_counts={date(2026, 1, 5): 1, date(2026, 1, 6): 1},
        slot_count=1,
        allocation=OLRAllocationPlan("all_in", mode="selected_equal", target_gross_exposure=1.0, max_position_pct=1.0),
        initial_equity=10_000.0,
        config=OLRConfig(slippage_bps=0.0, commission_bps=0.0, tax_bps_on_sell=0.0),
    )

    assert metrics["portfolio_proxy_deployed_trade_count"] == 1.0
    assert metrics["portfolio_proxy_cash_rejected_count"] == 1.0
    assert metrics["portfolio_proxy_net_return_pct"] == pytest.approx(0.10)


def test_olr_portfolio_proxy_blocks_same_symbol_overlap() -> None:
    first = _olr_outcome(
        date(2026, 1, 5),
        "005930",
        datetime(2026, 1, 5, 14, 35),
        datetime(2026, 1, 6, 15, 30),
        rank=1,
        gross_return=0.10,
    )
    overlapping_same_symbol = _olr_outcome(
        date(2026, 1, 6),
        "005930",
        datetime(2026, 1, 6, 14, 35),
        datetime(2026, 1, 7, 15, 30),
        rank=1,
        gross_return=0.10,
    )

    metrics = summarize_olr_portfolio_proxy(
        [first, overlapping_same_symbol],
        session_dates=(date(2026, 1, 5), date(2026, 1, 6)),
        selection_counts={date(2026, 1, 5): 1, date(2026, 1, 6): 1},
        slot_count=1,
        allocation=OLRAllocationPlan("half", mode="selected_equal", target_gross_exposure=0.5, max_position_pct=0.5),
        initial_equity=10_000.0,
        config=OLRConfig(slippage_bps=0.0, commission_bps=0.0, tax_bps_on_sell=0.0),
    )

    assert metrics["portfolio_proxy_deployed_trade_count"] == 1.0
    assert metrics["portfolio_proxy_symbol_blocked_count"] == 1.0
    assert metrics["portfolio_proxy_net_return_pct"] == pytest.approx(0.05)


def test_olr_rank_weighted_proxy_uses_selected_pool_denominator() -> None:
    fourth_rank_only = _olr_outcome(
        date(2026, 1, 5),
        "005930",
        datetime(2026, 1, 5, 14, 35),
        datetime(2026, 1, 5, 15, 30),
        rank=4,
        gross_return=0.10,
    )

    metrics = summarize_olr_portfolio_proxy(
        [fourth_rank_only],
        session_dates=(date(2026, 1, 5),),
        selection_counts={date(2026, 1, 5): 4},
        slot_count=4,
        allocation=OLRAllocationPlan("ranked", mode="rank_weighted", target_gross_exposure=1.0, max_position_pct=0.5, rank_decay=1.0),
        initial_equity=10_000.0,
        config=OLRConfig(slippage_bps=0.0, commission_bps=0.0, tax_bps_on_sell=0.0),
    )

    assert metrics["portfolio_proxy_deployed_trade_count"] == 1.0
    assert metrics["portfolio_proxy_net_return_pct"] == pytest.approx(0.012)


def test_olr_score_weighted_proxy_uses_candidate_scores() -> None:
    high_score = _olr_outcome(
        date(2026, 1, 5),
        "005930",
        datetime(2026, 1, 5, 14, 35),
        datetime(2026, 1, 5, 15, 30),
        rank=1,
        gross_return=0.10,
        metadata={"candidate_score": 10.0},
    )
    low_score = _olr_outcome(
        date(2026, 1, 5),
        "000660",
        datetime(2026, 1, 5, 14, 35),
        datetime(2026, 1, 5, 15, 30),
        rank=2,
        gross_return=0.0,
        metadata={"candidate_score": 1.0},
    )

    score_weighted = summarize_olr_portfolio_proxy(
        [high_score, low_score],
        session_dates=(date(2026, 1, 5),),
        selection_counts={date(2026, 1, 5): 2},
        slot_count=2,
        allocation=OLRAllocationPlan("score", mode="score_weighted", target_gross_exposure=1.0, max_position_pct=1.0),
        initial_equity=10_000.0,
        config=OLRConfig(slippage_bps=0.0, commission_bps=0.0, tax_bps_on_sell=0.0),
    )
    equal_weighted = summarize_olr_portfolio_proxy(
        [high_score, low_score],
        session_dates=(date(2026, 1, 5),),
        selection_counts={date(2026, 1, 5): 2},
        slot_count=2,
        allocation=OLRAllocationPlan("equal", mode="selected_equal", target_gross_exposure=1.0, max_position_pct=1.0),
        initial_equity=10_000.0,
        config=OLRConfig(slippage_bps=0.0, commission_bps=0.0, tax_bps_on_sell=0.0),
    )

    assert score_weighted["portfolio_proxy_net_return_pct"] > equal_weighted["portfolio_proxy_net_return_pct"]


def test_olr_portfolio_proxy_sizes_from_submission_price_not_future_fill() -> None:
    # A live market order is sized from the completed signal bar plus a buffer,
    # then filled on a later bar. The proxy must reject the same cash-overrun
    # case instead of using the future fill price to pre-shrink the order.
    outcome = _olr_outcome(
        date(2026, 1, 5),
        "005930",
        datetime(2026, 1, 5, 14, 35, tzinfo=KST),
        datetime(2026, 1, 6, 15, 30, tzinfo=KST),
        rank=1,
        gross_return=0.10,
        metadata={
            "entry_submission_time": datetime(2026, 1, 5, 14, 30, tzinfo=KST).isoformat(),
            "entry_sizing_price": 90.0,
        },
    )

    metrics = summarize_olr_portfolio_proxy(
        [outcome],
        session_dates=(date(2026, 1, 5),),
        selection_counts={date(2026, 1, 5): 1},
        slot_count=1,
        allocation=OLRAllocationPlan("all_in", mode="selected_equal", target_gross_exposure=1.0, max_position_pct=1.0),
        initial_equity=950.0,
        config=OLRConfig(slippage_bps=0.0, commission_bps=0.0, tax_bps_on_sell=0.0),
    )

    assert metrics["portfolio_proxy_deployed_trade_count"] == 0.0
    assert metrics["portfolio_proxy_cash_rejected_count"] == 1.0
    assert metrics["portfolio_proxy_net_return_pct"] == pytest.approx(0.0)


def test_olr_trade_plan_payload_sources_are_used_for_allocation() -> None:
    payload = {
        "candidate_sources": [
            {
                "rank": 1,
                "name": "src1_stage2",
                "stage1_name": "stage1",
                "stage2_name": "stage2",
                "score": 2.0,
                "mutations": {"olr.afternoon.top_n": 4},
            }
        ],
        "top_promoted": [
            {
                "spec": {
                    "name": "src1_stage2__confirm__next_close",
                    "candidate_source_name": "src1_stage2",
                    "entry": {"name": "confirm", "mode": "confirm_next_bar"},
                    "exit": {"name": "next_close", "mode": "next_close"},
                }
            }
        ],
    }

    specs = allocation_sweep._trade_specs_from_payload(payload)
    sources = allocation_sweep._sources_for_trade_specs(payload, specs)

    assert specs[0].entry.mode == "confirm_next_bar"
    assert specs[0].candidate_source_name == "src1_stage2"
    assert sources[0].name == "src1_stage2"
    assert sources[0].mutations == {"olr.afternoon.top_n": 4}


def _olr_outcome(
    trade_date: date,
    symbol: str,
    entry_time: datetime,
    exit_time: datetime,
    *,
    rank: int,
    gross_return: float,
    metadata: dict | None = None,
) -> OLRTradeOutcome:
    return OLRTradeOutcome(
        trade_date=trade_date,
        symbol=symbol,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=100.0,
        exit_price=100.0 * (1.0 + gross_return),
        stop_price=99.0,
        risk_per_share=1.0,
        gross_return_pct=gross_return,
        net_return_pct=gross_return,
        mfe_r=1.0,
        mae_r=0.0,
        mfe_capture=1.0,
        bars_held=1,
        entry_reason="confirm_next_bar",
        exit_reason="next_close",
        ambiguous_bar_count=0,
        stopped=False,
        target_hit=False,
        partial_hit=False,
        metadata={"candidate_rank": rank, **dict(metadata or {})},
    )

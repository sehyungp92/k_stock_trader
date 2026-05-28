from __future__ import annotations

from backtests.strategies.kalcb.phase_scoring import score_kalcb_phase


def test_kalcb_phase_score_uses_official_mtm_return_over_closed_trade_return() -> None:
    base = {
        "net_return_pct": 0.20,
        "official_mtm_net_return_pct": -0.01,
        "expected_total_r": 2.0,
        "entry_count": 40.0,
        "profit_factor": 1.5,
        "avg_r": 0.08,
        "mfe_capture": 0.36,
        "max_drawdown_pct": 0.004,
    }
    improved = {**base, "net_return_pct": -0.20, "official_mtm_net_return_pct": 0.03}

    assert score_kalcb_phase(1, improved) > score_kalcb_phase(1, base)


def test_legacy_return_fallback_requires_explicit_basis_marker() -> None:
    no_basis = {
        "net_return_pct": 0.20,
        "expected_total_r": 0.0,
        "entry_count": 20.0,
        "profit_factor": 1.2,
        "avg_r": 0.05,
        "mfe_capture": 0.35,
        "max_drawdown_pct": 0.002,
    }
    with_basis = {**no_basis, "net_return_pct_basis": "closed_trade_net_pnl_over_initial_equity"}

    assert score_kalcb_phase(1, with_basis) > score_kalcb_phase(1, no_basis)

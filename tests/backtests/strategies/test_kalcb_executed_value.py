from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from backtests.strategies.kalcb import executed_value as ev


def test_cash_label_accounts_for_partial_fills_and_tax_without_double_slippage():
    costs = SimpleNamespace(commission_bps=2, tax_bps_on_sell=18)
    def fill(side, qty, price, order):
        return SimpleNamespace(side=side, qty=qty, price=price, order_id=order, metadata={"order_qty": 10})
    fills = [fill("BUY", 3, 100, "b"), fill("BUY", 4, 102, "b"), fill("SELL", 7, 105, "s")]
    r = ev.ledger_value(fills, costs)
    assert r["net_value_krw"] == pytest.approx(735 - 708 - 708*.0002 - 735*.002)
    assert r["entry_orders_multislice"] == 1
    assert r["entry_orders_incomplete"] == 1
    assert ev.ledger_value(fills[:2], costs, terminal_inventory=7)["net_value_krw"] is None
    assert ev.ledger_value([], costs)["net_value_krw"] == 0
    assert ev.ledger_value([], costs, pending=1)["net_value_krw"] is None
    with pytest.raises(ValueError, match="inventory"):
        ev.ledger_value(fills, costs, terminal_inventory=1)


def test_eligibility_first_recovers_names_without_forcing_negative_value_trades():
    rows = [{"opportunity_id": str(i), "symbol": str(i)} for i in range(6)]
    predicate = lambda r, rank: int(r["symbol"]) >= 3
    old, _ = ev.admit(rows, [6, 5, 4, 3, 2, -1], predicate, construction="after", budget=2, shortlist=3)
    new, reasons = ev.admit(rows, [6, 5, 4, 3, 2, -1], predicate, construction="before", budget=2, shortlist=3, threshold=0)
    assert not old
    assert [r[0]["symbol"] for r in new] == ["3", "4"]
    assert reasons["5"] == "net_value_threshold"
    again, _ = ev.admit(rows[::-1], [-1, 2, 3, 4, 5, 6], predicate, construction="before", budget=2, shortlist=3, threshold=0)
    assert new == again


def test_future_or_unresolved_labels_are_never_fit():
    sessions = [f"2025-01-{d:02}" for d in range(1, 9)]
    rows = [{"trade_date": d, "label_available_at": d+"T15:30:00+09:00"} for d in sessions]
    rows[0]["label_available_at"] = "2025-01-07T10:00:00+09:00"
    rows[1]["label_available_at"] = None
    assert ev.mature_training_indices(rows, sessions, sessions[6], min_sessions=1, embargo=1) == [2, 3, 4]


def test_threshold_calibration_can_select_cash_and_reject_inventory_failures():
    rows = [{"trade_date": f"2025-01-{d:02}", "score": 8, "net_value_equity_bps": -2,
             "deadline_violation": False} for d in range(1, 22)]
    threshold, detail = ev.choose_threshold(rows)
    assert threshold == "cash"
    for row in rows:
        row.update(net_value_equity_bps=10, deadline_violation=True)
    assert ev.choose_threshold(rows)[0] == "cash"


def test_outcomes_cannot_enter_predictor_matrix():
    with pytest.raises(ValueError, match="allowlist"):
        ev.feature_matrix([], ["net_value_equity_bps"])


def test_opening_panel_membership_does_not_depend_on_later_bars():
    from tests.backtests.strategies.test_kalcb_first30_frontier_sweeps import _dataset, TRADE_DATE
    from backtests.strategies.kalcb.first30_signal_sweep import build_contexts
    dataset, _ = _dataset(symbols=("000001", "000002"))
    opening = replace(dataset, bars_by_key={k: tuple(b for b in bars if b.timestamp.hour == 9 and b.timestamp.minute < 30)
                                           for k, bars in dataset.bars_by_key.items()})
    full = build_contexts(dataset, require_post_entry_bars=False)[TRADE_DATE]
    causal = build_contexts(opening, require_post_entry_bars=False)[TRADE_DATE]
    assert [c.symbol for c in full] == [c.symbol for c in causal]
    for a, b in zip(full, causal):
        assert a.intraday == b.intraday
        assert a.sector_intraday == b.sector_intraday
    assert not build_contexts(opening)


def test_ranker_future_outcome_poison_cannot_change_current_forecasts():
    import numpy as np
    from scripts.kalcb_executed_value_ranker import fit_forecast
    sessions = [(datetime(2025, 1, 1) + timedelta(days=i)).date().isoformat() for i in range(40)]
    rows = [{"opportunity_id": day + "/A", "trade_date": day,
             "features": {key: (i % 7) / 10 for key in ev.FEATURES}, "quality_target": i / 100}
            for i, day in enumerate(sessions)]
    labels = {r["opportunity_id"]: {"label_available_at": r["trade_date"] + "T15:30:00+09:00",
              "label_valid": True, "net_value_equity_bps": i % 5 - 2} for i, r in enumerate(rows)}
    _, original, proof = fit_forecast(rows, labels, sessions, sessions[25], sessions[29], target="net", min_sessions=20)
    for row in rows[25:]:
        labels[row["opportunity_id"]]["net_value_equity_bps"] = 1e9
    _, poisoned, proof2 = fit_forecast(rows, labels, sessions, sessions[25], sessions[29], target="net", min_sessions=20)
    np.testing.assert_array_equal([s for _, s in original], [s for _, s in poisoned])
    assert proof["training_target_hash"] == proof2["training_target_hash"]


def test_policy_verifier_rejects_future_calibration_and_changed_admissions(tmp_path):
    import json
    from scripts.kalcb_executed_value_replay import verify_policy
    day = "2025-01-15"
    snapshot = ev.snapshot_for(day, [], source="fixture").to_json_dict()
    ev.write_json(tmp_path / "cash.snapshots.json", [snapshot])
    ev.write_json(tmp_path / "completion.json", {"complete": True})
    (tmp_path / "cash.admission.jsonl").write_text("")
    chronology = {"folds": [], "calibration": [{"valid_from": day,
        "latest_label_available_at": "2025-01-14T15:30:00+09:00"}]}
    def bind():
        ev.write_json(tmp_path / "chronology.json", chronology)
        names = ["cash.snapshots.json", "cash.admission.jsonl", "chronology.json"]
        ev.write_json(tmp_path / "policies_manifest.json", {"policies": ["cash"],
            "contract": {"budget": 8}, "files": {n: ev.digest(tmp_path / n) for n in names}})
    bind()
    assert len(verify_policy(tmp_path, "cash")[1]) == 1
    chronology["calibration"][0]["latest_label_available_at"] = day + "T15:30:00+09:00"
    bind()
    with pytest.raises(ValueError, match="Noncausal admission"):
        verify_policy(tmp_path, "cash")
    chronology["calibration"] = []
    bind()
    (tmp_path / "cash.admission.jsonl").write_text(json.dumps({"opportunity_id": day+"/A", "reason": "admitted"}) + "\n")
    with pytest.raises(ValueError, match="Changed policy artifact"):
        verify_policy(tmp_path, "cash")


def test_calibration_cannot_treat_unresolved_value_as_zero_profit():
    records = [{"trade_date": f"2025-01-{d:02}", "score": 20,
                "net_value_equity_bps": None, "deadline_violation": False} for d in range(1, 22)]
    threshold, detail = ev.choose_threshold(records)
    assert threshold == "cash"
    assert all(not trial["feasible"] for trial in detail["trials"][1:])


def test_snapshot_declares_actual_feature_availability_on_delayed_open():
    from datetime import date
    from strategy_kalcb.models import KALCBDailyCandidate
    day = date(2025, 11, 13)
    candidate = KALCBDailyCandidate("000100", day, 100, 90, 95, 3, 1000, 6000)
    row = {"candidate": candidate.to_json_dict(), "feature_available_at": "2025-11-13T10:30:00+09:00"}
    snapshot = ev.snapshot_for(day.isoformat(), [(row, 2)], source="fixture")
    assert snapshot.generated_at.isoformat() == row["feature_available_at"]
    assert snapshot.candidates[0].metadata["feature_available_at"] == row["feature_available_at"]


def test_portfolio_calibration_ignores_future_equity_and_rejects_past_inventory():
    from scripts.kalcb_executed_value_admission import select_portfolio_threshold
    daily = [{"day": f"2025-01-{d:02}", "equity": ev.REFERENCE_EQUITY + d * 100,
              "inventory": {}, "pending": 0} for d in range(1, 31)]
    case = {"daily": daily, "residual_sessions": {}, "trades": []}
    threshold, proof = select_portfolio_threshold([(2.0, case)], "2025-01-25")
    assert threshold == 2.0
    assert proof["selected"]["net_pnl_krw"] == 2400
    daily[-1]["equity"] = 1e20
    assert select_portfolio_threshold([(2.0, case)], "2025-01-25") == (threshold, proof)
    case["residual_sessions"]["2025-01-23"] = {"KALCB:A": 1}
    assert select_portfolio_threshold([(2.0, case)], "2025-01-25")[0] == "cash"


def test_physical_volume_validation_aggregates_sides_and_partial_fills():
    bar = {"symbol": "A", "timestamp": "2025-01-02T09:35:00+09:00", "volume": 5}
    fills = [{"qty": q, "metadata": {"execution_bar": bar}} for q in (2, 3)]
    assert not ev.physical_fill_failures(fills)
    fills.append({"qty": 1, "metadata": {"execution_bar": bar}})
    assert ev.physical_fill_failures(fills)[0]["quantity"] == 6


def test_future_invalid_execution_does_not_influence_earlier_calibration():
    from scripts.kalcb_executed_value_admission import select_portfolio_threshold
    case = {"daily": [{"day": f"2025-01-{d:02}", "equity": ev.REFERENCE_EQUITY + 100,
              "inventory": {}, "pending": 0} for d in range(1, 31)],
            "residual_sessions": {}, "trades": [], "invalid_execution_days": ["2025-01-29"]}
    assert select_portfolio_threshold([(5, case)], "2025-01-25")[0] == 5
    assert select_portfolio_threshold([(5, case)], "2025-01-30")[0] == "cash"

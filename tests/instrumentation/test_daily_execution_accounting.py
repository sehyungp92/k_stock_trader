from datetime import datetime, timezone
import json

import pytest

from instrumentation.src.daily_snapshot import DailySnapshotBuilder
from instrumentation.src.trade_logger import TradeLogger
from instrumentation.src.execution_accounting import daily_accounting
from instrumentation.src.sidecar import Sidecar


def stamp(day, hour):
    return datetime(2026, 1, day, hour, tzinfo=timezone.utc)


def enter(logger, trade_id="a", strategy="OLR"):
    logger.log_entry(trade_id, "005930", "LONG", 100, 10, 1000, "signal", "s", 1,
                     [], [], {}, exchange_timestamp=stamp(5, 1), strategy_id=strategy,
                     experiment_id="exp", experiment_variant="A")


def test_partial_overnight_restart_correction_and_consumer(tmp_path):
    from schemas.execution_accounting import ExecutionDailySnapshot, latest_snapshots
    from skills.build_daily_metrics import DailyMetricsBuilder
    from skills.build_weekly_metrics import WeeklyMetricsBuilder

    config = {"data_dir": str(tmp_path), "bot_id": "k_stock_trader", "strategy_type": "OLR"}
    logger = TradeLogger(config, None)
    enter(logger)
    logger.log_exit("a", 110, "SIGNAL", exited_qty=6, execution_id="x", exchange_timestamp=stamp(5, 2))
    builder = DailySnapshotBuilder(config)
    first = builder.build("2026-01-05")
    assert (first.total_trades, first.execution_count, first.gross_pnl, first.net_pnl, first.total_fees) == (0, 1, 60, None, None)
    assert first.max_concurrent_positions == 1 and first.max_drawdown_pct is None
    logger = TradeLogger(config, None)
    assert logger.log_exit("a", 110, "SIGNAL", exited_qty=6, execution_id="x") is None
    logger.log_exit("a", 90, "SIGNAL", exited_qty=4, execution_id="y", exchange_timestamp=stamp(6, 2), fees_paid=2)
    second = builder.build("2026-01-06")
    assert (second.total_trades, second.gross_pnl, second.net_pnl) == (1, -40, -42)
    assert second.unclassified_trade_count == 1 and second.win_rate is None
    assert second.max_concurrent_positions == 1
    assert second.completed_trades[0]["gross_pnl"] == 20
    assert second.completed_trades[0]["net_pnl"] is None
    assert second.per_strategy_summary["OLR"]["unclassified_trade_count"] == 1
    assert second.experiment_breakdown["exp:A"]["win_count"] == 0

    builder.save(first)
    builder.save(second)
    assert logger.reconcile_execution_costs("a", "x", 3, source_ref="broker:test-statement:1")
    assert not logger.reconcile_execution_costs("a", "x", 3, source_ref="broker:test-statement:1")
    first, second = builder.build("2026-01-05"), builder.build("2026-01-06")
    assert first.net_pnl == 57 and second.completed_trades[0]["net_pnl"] == 15
    assert second.total_trades == second.win_count == 1 and second.unclassified_trade_count == 0
    builder.save(first)
    builder.save(second)
    builder.save(second)
    sidecar = Sidecar(config)
    sidecar.accepts_execution_accounting = True
    records = []
    for path, event_type in sidecar._get_event_files():
        if event_type == "daily_snapshot":
            records.extend(sidecar._read_unsent_events(path, event_type))
    assert len(records) == 4 and len({r["event_id"] for r in records}) == 4
    payloads = [json.loads(r["payload"]) for r in records]
    for payload in payloads:
        ExecutionDailySnapshot.model_validate(payload)
    latest = latest_snapshots(list(reversed(payloads)) + payloads)
    assert len(latest) == 2
    summaries = [DailyMetricsBuilder(r["date"], "k_stock_trader").build_summary([], r) for r in latest]
    weekly = WeeklyMetricsBuilder("2026-01-05", "2026-01-11", ["k_stock_trader"]).build_bot_summary("k_stock_trader", summaries)
    assert (weekly.total_trades, weekly.gross_pnl, weekly.net_pnl, weekly.win_count) == (1, 20, 15, 1)
    # Incomplete initial observations also survive daily -> weekly without zero filling.
    old = [r for r in payloads if r["snapshot_sequence"] == 1]
    summaries = [DailyMetricsBuilder(r["date"], "k_stock_trader").build_summary([], r) for r in old]
    unknown = WeeklyMetricsBuilder("2026-01-05", "2026-01-11", []).build_bot_summary("k_stock_trader", summaries)
    assert unknown.net_pnl is None and unknown.unclassified_trade_count == 1 and unknown.win_rate is None
    for row in latest:
        DailyMetricsBuilder(row["date"], "k_stock_trader").write_curated([], [], tmp_path / "curated", daily_snapshot=row)


def test_interleaved_entry_after_partial_sale_retains_one_lifecycle(tmp_path):
    config = {"data_dir": str(tmp_path), "bot_id": "test"}
    logger = TradeLogger(config, None)
    enter(logger)
    logger.log_exit("a", 110, "SIGNAL", exited_qty=6, execution_id="x", fees_paid=1, exchange_timestamp=stamp(5, 2))
    # Four remaining at 100 plus another two at 130: six at average 110.
    logger.reconcile_open_position("a", 6, 110, "y")
    logger = TradeLogger(config, None)
    logger.log_exit("a", 120, "SIGNAL", exited_qty=6, execution_id="y", fees_paid=2, exchange_timestamp=stamp(5, 3))
    snapshot = DailySnapshotBuilder(config).build("2026-01-05")
    assert (snapshot.total_trades, snapshot.gross_pnl, snapshot.net_pnl) == (1, 120, 117)
    assert snapshot.completed_trades[0]["position_size"] == 12


def test_same_day_fragments_are_one_trade_and_duplicate_conflicts_fail(tmp_path):
    config = {"data_dir": str(tmp_path), "bot_id": "test"}
    logger = TradeLogger(config, None)
    enter(logger)
    logger.log_exit("a", 110, "SIGNAL", exited_qty=6, execution_id="x", exchange_timestamp=stamp(5, 2))
    logger.log_exit("a", 90, "SIGNAL", exited_qty=4, execution_id="y", exchange_timestamp=stamp(5, 3))
    snapshot = DailySnapshotBuilder(config).build("2026-01-05")
    assert (snapshot.total_trades, snapshot.gross_pnl, snapshot.net_pnl, snapshot.total_fees) == (1, 20, None, None)
    assert snapshot.win_count == snapshot.loss_count == 0
    rows = [json.loads(r) for p in (tmp_path / "trades").glob("*.jsonl") for r in p.read_text().splitlines()]
    assert daily_accounting(rows + rows, "2026-01-05")[0]["gross_pnl"] == 20
    bad = dict(rows[-1], gross_pnl=999)
    with pytest.raises(ValueError, match="Conflicting execution"):
        daily_accounting(rows + [bad], "2026-01-05")


def test_default_sidecar_contains_unknown_daily_economics(tmp_path):
    config = {"data_dir": str(tmp_path), "bot_id": "test"}
    logger = TradeLogger(config, None)
    enter(logger)
    logger.log_exit("a", 110, "SIGNAL", execution_id="x", exchange_timestamp=stamp(5, 2))
    builder = DailySnapshotBuilder(config)
    builder.save(builder.build("2026-01-05"))
    sidecar = Sidecar(config)
    path = tmp_path / "daily/daily_2026-01-05.jsonl"
    assert not sidecar._read_unsent_events(path, "daily_snapshot")


def test_snapshot_journal_recovers_after_cache_rename_failure(tmp_path, monkeypatch):
    import instrumentation.src.daily_snapshot as module
    config = {"data_dir": str(tmp_path), "bot_id": "test"}
    builder = DailySnapshotBuilder(config)
    snapshot = builder.build("2026-01-05")
    replace = module.os.replace
    def crash(*args):
        raise OSError("interrupted cache rename")
    monkeypatch.setattr(module.os, "replace", crash)
    with pytest.raises(OSError):
        builder.save(snapshot)
    monkeypatch.setattr(module.os, "replace", replace)
    builder.save(snapshot)
    snapshot.error_count = 1
    builder.save(snapshot)
    path = tmp_path / "daily/daily_2026-01-05.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["snapshot_sequence"] for row in rows] == [1, 2]
    # A torn append is neither exported nor treated as a committed revision.
    with path.open("a") as stream:
        stream.write('{"schema_version":"daily_execution_accounting_v2"')
    builder.save(snapshot)
    assert len(path.read_text().splitlines()) == 2

from instrumentation.src.trade_logger import TradeEvent, TradeLogger


def test_partial_exit_uses_executed_quantity_and_unknown_costs(tmp_path):
    logger = TradeLogger({"data_dir": str(tmp_path)}, None)
    logger._open_trades["a"] = TradeEvent("a", {}, pair="005930", entry_price=100, position_size=10)
    first = logger.log_exit("a", 110, "target", exited_qty=6, execution_id="x")
    assert first.gross_pnl == 60
    assert first.net_pnl is None and first.fees_paid is None
    assert first.remaining_qty == 4
    assert logger.get_open_trades()["a"].position_size == 4
    logger.reconcile_open_position("a", 10, 100, "x")
    assert logger.log_exit("a", 110, "target", exited_qty=6, execution_id="x") is None
    assert logger.get_open_trades()["a"].position_size == 4
    last = logger.log_exit("a", 110, "exit", exited_qty=4, execution_id="y", fees_paid=2)
    assert last.net_pnl == 38 and "a" not in logger.get_open_trades()


def test_entry_fragment_reconciliation_updates_open_cost_basis(tmp_path):
    logger = TradeLogger({"data_dir": str(tmp_path)}, None)
    logger._open_trades["a"] = TradeEvent("a", {}, entry_price=100, position_size=4)
    logger.reconcile_open_position("a", 10, 106, "x")
    event = logger.log_exit("a", 110, "target", exited_qty=6, execution_id="x")
    assert event.gross_pnl == 24 and event.remaining_qty == 4


def test_restart_replay_and_export_failure_keep_one_committed_execution(tmp_path, monkeypatch):
    logger = TradeLogger({"data_dir": str(tmp_path)}, None)
    logger._open_trades["a"] = TradeEvent("a", {}, pair="005930", entry_price=100, position_size=10)
    def failed_export():
        raise OSError("disk export interrupted")
    monkeypatch.setattr(logger._projection, "export", failed_export)
    first = logger.log_exit("a", 110, "target", exited_qty=6, execution_id="x", consumer_id="PCIM", consumer_sequence=7)
    assert first.gross_pnl == 60
    restarted = TradeLogger({"data_dir": str(tmp_path)}, None)
    assert restarted._projection.cursor("PCIM") == 7
    assert restarted.get_open_trades()["a"].position_size == 4
    restarted.reconcile_open_position("a", 10, 100, "x")
    assert restarted.log_exit("a", 110, "target", exited_qty=6, execution_id="x") is None
    restarted._projection.export()
    import json
    rows = [json.loads(line) for path in (tmp_path / "trades").glob("*.jsonl") for line in path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["gross_pnl"] == 60


def test_projection_write_failure_does_not_acknowledge_or_reduce_position(tmp_path, monkeypatch):
    logger = TradeLogger({"data_dir": str(tmp_path)}, None)
    logger._open_trades["a"] = TradeEvent("a", {}, entry_price=100, position_size=10)
    def failed_commit(*args, **kwargs):
        raise OSError("write failed")
    monkeypatch.setattr(logger._projection, "commit", failed_commit)
    assert logger.log_exit("a", 110, "target", exited_qty=6, execution_id="x", consumer_id="PCIM", consumer_sequence=7) is None
    assert logger._projection.cursor("PCIM") == 0
    assert logger.get_open_trades()["a"].position_size == 10

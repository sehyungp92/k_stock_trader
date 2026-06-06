from __future__ import annotations

import asyncio
import json
from pathlib import Path
from datetime import date, datetime, time
from types import SimpleNamespace

import pytest

from strategy_common.clock import KST
from strategy_common.market import MarketBar
from deployment.olr_kalcb.market_data_coordinator import KISWebSocketCompletedBarSource
from kis_core.ws_client import TickMessage
from scripts import run_olr_kalcb_runtime_session as operator


def test_operator_loads_sector_map_from_wrapped_json(tmp_path):
    path = tmp_path / "sector_map.json"
    path.write_text(json.dumps({"sector_map": {"5930": "semiconductors", "035420": "it"}}), encoding="utf-8")

    sector_map = operator._load_sector_map(path)

    assert sector_map == {"005930": "SEMICONDUCTORS", "035420": "IT"}


def test_operator_rejects_unknown_portfolio_policy_fields():
    with pytest.raises(ValueError, match="unknown portfolio policy fields"):
        operator._portfolio_config_from_payload({"max_gross_notional": 1_000_000.0, "paper_trading_approved": True})


def test_operator_fixture_health_ok_is_dry_run_only():
    checks = operator._load_health_checks(None, mode="dry_run", fixture_health_ok=True)

    assert checks["artifact_only_gate_passed"]["passed"] is True
    assert "fixture-only" in checks["market_data_ok"]["detail"]
    with pytest.raises(ValueError, match="only valid"):
        operator._load_health_checks(None, mode="paper", fixture_health_ok=True)


def test_operator_augments_paper_health_checks_with_raw_oms_health(monkeypatch):
    payload = {"status": "ok", "stop_protection_status": "ok", "idempotency_status": "ok"}
    monkeypatch.setattr(operator, "_fetch_oms_health_payload", lambda url: payload)

    checks = operator._with_oms_health_payload(
        {"paper_trading_approved": True},
        SimpleNamespace(oms_url="http://oms.example"),
        mode="paper",
    )

    assert checks["oms_health_payload"] == payload


def test_operator_has_watch_bars_runtime_entrypoint():
    args = operator._parser().parse_args(
        [
            "watch-bars",
            "--trade-date",
            "2026-02-02",
            "--mode",
            "paper",
            "--bars-parquet",
            "data/paper_live/olr_kalcb/2026-02-02/market_bars_5m.parquet",
        ]
    )

    assert args.command == "watch-bars"
    assert args.mode == "paper"
    assert args.poll_seconds > 0


def test_operator_kis_websocket_watch_does_not_require_bars_parquet():
    args = operator._parser().parse_args(
        [
            "watch-bars",
            "--trade-date",
            "2026-02-02",
            "--mode",
            "paper",
            "--market-data-source",
            "kis_websocket",
        ]
    )

    assert args.command == "watch-bars"
    assert args.bars_parquet is None


def test_operator_auto_market_data_source_is_promotional_ws_owned():
    dry_run = operator._parser().parse_args(
        [
            "dry-run-bars",
            "--trade-date",
            "2026-02-02",
            "--bars-parquet",
            "data/paper_live/olr_kalcb/2026-02-02/market_bars_5m.parquet",
        ]
    )
    paper = operator._parser().parse_args(
        [
            "watch-bars",
            "--trade-date",
            "2026-02-02",
            "--mode",
            "paper",
            "--bars-parquet",
            "data/paper_live/olr_kalcb/2026-02-02/market_bars_5m.parquet",
        ]
    )

    assert operator._resolve_market_data_source(dry_run, "dry_run") == "external_completed_bars"
    assert operator._resolve_market_data_source(paper, "paper") == "kis_websocket"


def test_operator_watch_processing_routes_unseen_bars_through_coordinator():
    trade_date = date(2026, 2, 2)
    bar = MarketBar(
        symbol="005930",
        timestamp=datetime.combine(trade_date, time(9, 35), tzinfo=KST),
        timeframe="5m",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1000.0,
    )

    class Plan:
        async def handle_bar(self, _bar):
            raise AssertionError("watch-bars must route through the market-data coordinator when one exists")

    class Coordinator:
        def __init__(self):
            self.routed = []

        async def route_completed_bar(self, plan, routed_bar):
            self.routed.append(routed_bar)
            return ()

    coordinator = Coordinator()
    seen: set[str] = set()

    processed = asyncio.run(operator._process_unseen_bars(Plan(), [bar], seen, coordinator=coordinator))
    duplicate = asyncio.run(operator._process_unseen_bars(Plan(), [bar], seen, coordinator=coordinator))

    assert processed == 1
    assert duplicate == 0
    assert coordinator.routed == [bar]


def test_kis_websocket_completed_bar_source_runs_client_and_emits_completed_5m_bar():
    timestamp = datetime.combine(date(2026, 2, 2), time(9, 0), tzinfo=KST)
    ticks = [
        TickMessage("005930", 100.0, 10.0, 10.0, 1000.0, 0.0, timestamp),
        TickMessage("005930", 101.0, 20.0, 30.0, 3030.0, 0.0, timestamp.replace(minute=1)),
        TickMessage("005930", 102.0, 30.0, 60.0, 6120.0, 0.0, timestamp.replace(minute=5)),
    ]

    class FakeWebSocketClient:
        def __init__(self):
            self.callbacks = []
            self.run_started = False

        def on_tick(self, callback):
            self.callbacks.append(callback)

        async def run(self, auto_reconnect=True):
            self.run_started = auto_reconnect
            for tick in ticks:
                for callback in self.callbacks:
                    callback(tick)
            await asyncio.sleep(60)

    client = FakeWebSocketClient()
    source = KISWebSocketCompletedBarSource(client)

    async def scenario():
        await source.start()
        bar = await source.next_bar(timeout_s=0.5)
        await source.stop()
        return bar

    bar = asyncio.run(scenario())

    assert client.run_started is True
    assert bar is not None
    assert bar.symbol == "005930"
    assert bar.timestamp == timestamp
    assert bar.timeframe == "5m"
    assert bar.open == 100.0
    assert bar.high == 101.0
    assert bar.close == 101.0
    assert bar.volume == 30.0
    assert bar.source == "kis_websocket"


def test_single_vps_wrappers_enforce_artifact_gate_before_runtime_restart():
    premarket = Path("infra/cron/olr_kalcb_premarket_restart.sh").read_text(encoding="utf-8")
    afternoon = Path("infra/cron/olr_kalcb_afternoon_restart.sh").read_text(encoding="utf-8")
    entrypoint = Path("deployment/olr_kalcb/runtime_service_entrypoint.sh").read_text(encoding="utf-8")

    assert "flock -n" in premarket
    assert "timeout \"$ARTIFACT_TIMEOUT\" docker compose run --rm runtime" in premarket
    assert "DAILY_UNIVERSE_FILE" in premarket
    assert "approved OLR/KALCB deployment universe file is missing" in premarket
    assert 'daily_args=(--daily-universe-file "$DAILY_UNIVERSE_FILE")' in premarket
    assert "generate_olr_kalcb_artifacts.py daily" in premarket
    assert "--mode artifact_only_stage1" in premarket
    assert premarket.index("generate_olr_kalcb_artifacts.py daily") < premarket.index("docker compose up -d runtime")

    assert "generate_olr_kalcb_artifacts.py afternoon" in afternoon
    assert "--mode artifact_only" in afternoon
    assert afternoon.index("generate_olr_kalcb_artifacts.py afternoon") < afternoon.index("--force-recreate runtime")

    assert "OLR_KALCB_TRADE_DATE must be set" in entrypoint
    assert "watch-bars" in entrypoint
    assert "--market-data-source" in entrypoint

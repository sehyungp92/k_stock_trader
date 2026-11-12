import time
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import pandas as pd
import requests

import pytest

from kis_core.kis_client import KoreaInvestAPI
from oms.adapter import BrokerQueryResult, BrokerOrder
from oms.broker_snapshot import collect_broker_snapshot
from oms.intent import Intent, IntentResult, IntentStatus, IntentType
from oms.oms_core import OMSCore


def core():
    oms = OMSCore(MagicMock(), require_reconciliation=True)
    oms.adapter.get_orders = AsyncMock(return_value=BrokerQueryResult(True, []))
    oms.adapter.get_balance_snapshot = AsyncMock(return_value=(BrokerQueryResult(True, []), 1000))
    oms.adapter.get_buyable_cash = AsyncMock(return_value=1000)
    return oms


def intent(kind=IntentType.ENTER, **kwargs):
    return Intent(intent_type=kind, strategy_id="KALCB", symbol="005930", desired_qty=1, **kwargs)


@pytest.mark.asyncio
async def test_startup_failure_retry_does_not_reserve_action_and_recovery_admits_it():
    oms = core()
    action = intent()
    oms._reserve_idempotency = AsyncMock(return_value=None)
    oms._process_intent = AsyncMock(return_value=IntentResult(action.intent_id, IntentStatus.EXECUTED))
    assert (await oms.submit_intent(action)).status == IntentStatus.DEFERRED
    oms._reserve_idempotency.assert_not_called()
    await oms._reconcile()
    assert (await oms.submit_intent(action)).status == IntentStatus.EXECUTED
    oms._reserve_idempotency.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_enables_gate_even_when_initial_broker_query_fails():
    oms = core()
    oms.require_reconciliation = False
    oms.adapter.get_orders.return_value = BrokerQueryResult(False)
    oms.start_reconciliation_loop = AsyncMock()
    await oms.start()
    assert oms.require_reconciliation
    assert (await oms.submit_intent(intent())).status == IntentStatus.DEFERRED
    oms.start_reconciliation_loop.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["get_orders", "get_balance_snapshot", "get_buyable_cash"])
async def test_query_failure_immediately_closes_gate_and_success_recovers(query):
    oms = core()
    await oms._reconcile()
    assert oms.reconciliation_health_payload()["broker_reconciliation_ready"]
    method = getattr(oms.adapter, query)
    saved = method.return_value
    method.return_value = (BrokerQueryResult(False), None) if query == "get_balance_snapshot" else (
        None if query == "get_buyable_cash" else BrokerQueryResult(False))
    oms._cash_observed_at = None
    with pytest.raises(RuntimeError):
        await oms._reconcile()
    assert (await oms.submit_intent(intent())).status == IntentStatus.DEFERRED
    method.return_value = saved
    await oms._reconcile()
    assert oms.reconciliation_health_payload()["broker_reconciliation_ready"]


@pytest.mark.asyncio
async def test_stale_and_in_progress_reconciliation_block_only_increases():
    oms = core()
    await oms._reconcile()
    oms._reconciliation_success_at = time.monotonic() - 61
    assert oms._entry_readiness_failure(intent())
    assert oms._entry_readiness_failure(intent(IntentType.SET_TARGET, target_qty=2))
    oms._process_intent = AsyncMock(return_value=IntentResult("management", IntentStatus.EXECUTED))
    for kind in (IntentType.EXIT, IntentType.FLATTEN, IntentType.CANCEL_ORDERS, IntentType.REDUCE):
        assert (await oms.submit_intent(intent(kind))).status == IntentStatus.EXECUTED
    assert not oms._entry_readiness_failure(intent(IntentType.SET_TARGET, target_qty=0))
    oms._reconciliation_success_at = time.monotonic()
    oms._reconciliation_running = True
    assert oms._entry_readiness_failure(intent())


@pytest.mark.asyncio
async def test_duplicate_returns_previous_result_during_outage():
    oms = core()
    action = intent()
    prior = IntentResult(action.intent_id, IntentStatus.EXECUTED, order_id="observed")
    oms._idem.put(action.idempotency_key, prior)
    assert await oms.submit_intent(action) is prior


@pytest.mark.asyncio
async def test_failure_during_planning_cannot_submit_a_buy():
    oms = core()
    await oms._reconcile()
    async def price(_):
        oms._reconciliation_error = "disconnected"
        return 100
    oms._get_current_price = price
    oms.adapter.submit_order = AsyncMock()
    result = await oms._plan_and_execute(intent(), 1, None)
    assert result.status == IntentStatus.DEFERRED
    oms.adapter.submit_order.assert_not_called()


@pytest.mark.asyncio
async def test_unowned_working_order_cannot_be_declared_reconciled():
    oms = core()
    oms.adapter.get_orders.return_value = BrokerQueryResult(True, [
        BrokerOrder("outside", "005930", "BUY", 1, 0, 100, "WORKING", "09:00:00")])
    with pytest.raises(RuntimeError, match="unowned"):
        await oms._reconcile()
    assert not oms.reconciliation_health_payload()["broker_reconciliation_ready"]


@pytest.mark.asyncio
async def test_canonical_allocation_is_not_silently_erased_by_broker_drift():
    oms = core()
    oms.require_persistence = True
    oms.state.set_allocation("005930", "KALCB", 10, cost_basis=100)
    pos = oms.state.get_position("005930")
    pos.real_qty = 0
    await oms._check_allocation_drift()
    assert pos.get_allocation("KALCB") == 10
    assert pos.frozen and oms.execution_recovery_required
    assert not oms.reconciliation_health_payload()["broker_reconciliation_ready"]


@pytest.mark.asyncio
async def test_correct_quantity_with_wrong_basis_does_not_admit_new_entries():
    oms = core()
    await oms._reconcile()
    oms.state.set_allocation("005930", "KALCB", 1, cost_basis=100)
    pos = oms.state.get_position("005930")
    pos.real_qty, pos.avg_price = 1, 110
    assert oms.reconciliation_health_payload()["broker_reconciliation_reason"] == "broker_allocation_cost_basis_mismatch"


def page(*, more=False, failed=False):
    body = SimpleNamespace(output=[], output1=[], output2=[{"tot_evlu_amt": "1000"}],
                           ctx_area_fk100="a", ctx_area_nk100="b")
    return SimpleNamespace(is_ok=lambda: not failed, get_body=lambda: body,
                           get_header=lambda: SimpleNamespace(tr_cont="M" if more else ""))


@pytest.mark.parametrize("method", ["get_acct_balance", "get_orders"])
@pytest.mark.parametrize("pages", [[None], [page(more=True), None], [page(more=True), page(more=True)]])
def test_failed_or_repeated_account_page_never_means_empty_account(method, pages):
    api = KoreaInvestAPI.__new__(KoreaInvestAPI)
    api.account_num = "fixture"
    api._url_fetch = MagicMock(side_effect=pages)
    with pytest.raises(RuntimeError):
        getattr(api, method)()


@pytest.mark.parametrize("method", ["get_acct_balance", "get_orders"])
def test_empty_success_and_all_pages_are_valid_account_observations(method):
    api = KoreaInvestAPI.__new__(KoreaInvestAPI)
    api.account_num = "fixture"
    api._url_fetch = MagicMock(side_effect=[page(more=True), page()])
    getattr(api, method)()
    assert api._url_fetch.call_count == 2
    assert api._url_fetch.call_args.kwargs["tr_cont"] == "N"


@pytest.mark.asyncio
async def test_collector_binds_paper_account_and_refuses_live_or_partial_results():
    adapter = core().adapter
    adapter.api = SimpleNamespace(is_paper_trading=True, using_url="https://openapivts.koreainvestment.com:29443", account_num="fixture")
    adapter.get_execution_orders = AsyncMock(return_value=BrokerQueryResult(True, []))
    packet = await collect_broker_snapshot(adapter, oms_id="paper", account_alias="test", history_start=date(2026, 1, 1))
    assert packet["queries_complete"] and packet["positions"] == {}
    assert len(packet["account_fingerprint"]) == 64
    assert "fixture" not in str(packet)
    adapter.api.is_paper_trading = False
    with pytest.raises(ValueError):
        await collect_broker_snapshot(adapter, oms_id="paper", account_alias="test")
    adapter.api.is_paper_trading = True
    adapter.get_execution_orders.return_value = BrokerQueryResult(False)
    with pytest.raises(RuntimeError):
        await collect_broker_snapshot(adapter, oms_id="paper", account_alias="test")


def test_saved_healthy_payload_cannot_override_current_connection_failure(monkeypatch):
    from scripts import run_olr_kalcb_runtime_session as operator
    def unavailable(_):
        raise ConnectionError("offline")
    monkeypatch.setattr(operator, "_fetch_oms_health_payload", unavailable)
    result = operator._with_oms_health_payload({"oms_health_payload": {"status": "ok"}, "account_ok": True},
                                              SimpleNamespace(oms_url="http://localhost"), mode="paper")
    assert "oms_health_payload" not in result
    assert not result["account_ok"]["passed"]


def test_initial_state_comes_from_current_oms_endpoints(monkeypatch):
    from scripts import run_olr_kalcb_runtime_session as operator
    responses = {"http://localhost/api/v1/state/account": {"equity": 1000, "buyable_cash": 900},
                 "http://localhost/api/v1/positions": {"005930": {"real_qty": 1}}}
    fetch = MagicMock(side_effect=lambda url: responses[url])
    monkeypatch.setattr(operator, "_fetch_oms_health_payload", fetch)
    account, positions = operator._fetch_oms_initial_state(SimpleNamespace(oms_url="http://localhost"))
    assert account["buyable_cash"] == 900 and positions["005930"]["real_qty"] == 1
    assert fetch.call_count == 2


def test_fresh_broker_evidence_preserves_operator_disabled_order_route(monkeypatch):
    from scripts import run_olr_kalcb_runtime_session as operator
    monkeypatch.setattr(operator, "_fetch_oms_health_payload", lambda _: {
        "broker_reconciliation_ready": True, "broker_mode": "paper", "broker_account_fingerprint": "fixture"})
    result = operator._with_oms_health_payload({"order_route_enabled": False},
                                              SimpleNamespace(oms_url="http://localhost"), mode="paper")
    assert result["account_ok"]["passed"]
    assert not result["order_route_enabled"]["passed"]


@pytest.mark.asyncio
async def test_paper_pending_orders_use_supported_history_and_actual_filled_quantity():
    from oms.adapter import KISExecutionAdapter
    api = KoreaInvestAPI.__new__(KoreaInvestAPI)
    api.is_paper_trading = True
    api._url_fetch = MagicMock(side_effect=AssertionError("Unsupported endpoint must not be called"))
    api.get_execution_orders = MagicMock(return_value=pd.DataFrame([
        {"odno": "partial", "pdno": "005930", "ord_qty": "10", "tot_ccld_qty": "4",
         "tot_ccld_amt": "424", "rmn_qty": "2", "ord_unpr": "107", "ord_dt": "20261002",
         "ord_tmd": "100000", "ord_gno_brno": "branch", "sll_buy_dvsn_cd": "02", "cncl_yn": "Y"},
        {"odno": "cancelled", "pdno": "005930", "ord_qty": "1", "tot_ccld_qty": "0",
         "tot_ccld_amt": "0", "rmn_qty": "0", "ord_unpr": "107", "ord_dt": "20261002",
         "ord_tmd": "100000", "ord_gno_brno": "branch", "sll_buy_dvsn_cd": "02", "cncl_yn": "Y"}]))
    adapter = KISExecutionAdapter(api)
    result = await adapter.get_orders()
    assert result.ok and len(result.data) == 1
    assert result.data[0].filled_qty == 4 and result.data[0].side == "BUY"
    history = await adapter.get_execution_orders("20261002", "20261002")
    assert history.ok and history.data[0].status == "WORKING"
    assert history.data[1].status == "CANCELLED"
    api._url_fetch.assert_not_called()


def test_get_cache_isolated_by_actual_broker_account(monkeypatch):
    from kis_core import kis_client as client
    response = requests.Response()
    response.status_code = 200
    response._content = b'{"rt_cd":"0","output":[]}'
    get = MagicMock(return_value=response)
    monkeypatch.setattr(client.requests, "get", get)
    monkeypatch.setattr(client, "_response_cache", {})
    monkeypatch.setattr(client._http_limiter, "wait", lambda: None)
    monkeypatch.setattr(client, "_get_circuit_breaker", lambda *_: SimpleNamespace(
        is_open=lambda: False, record_success=lambda: None))
    for account in ("paper-a", "paper-a", "paper-b"):
        api = KoreaInvestAPI.__new__(KoreaInvestAPI)
        api.account_num = account
        api.using_url = "https://openapivts.koreainvestment.com:29443"
        api.is_paper_trading = True
        api.custtype = "P"
        api.env = SimpleNamespace(has_real_fallback=False, get_base_headers=lambda: {})
        assert api._url_fetch("/uapi/domestic-stock/v1/trading/inquire-balance", "TTTC8434R", {}).is_ok()
    assert get.call_count == 2

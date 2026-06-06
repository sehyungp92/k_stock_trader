from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from oms import server
from oms.intent import IntentResult, IntentStatus


def test_idempotency_pending_and_manual_resolution_api_require_audit_reason(monkeypatch):
    persistence = SimpleNamespace()
    persistence.list_pending_idempotency = AsyncMock(
        return_value=[
            {
                "intent_id": "intent-1",
                "idempotency_key": "idem-1",
                "strategy_id": "KALCB",
                "symbol": "005930",
                "reservation_reconcile_status": "AMBIGUOUS",
            }
        ]
    )
    persistence.resolve_idempotency = AsyncMock(
        return_value=IntentResult(
            intent_id="intent-1",
            status=IntentStatus.DEFERRED,
            message="operator confirmed no broker order exists",
            order_id=None,
        )
    )
    monkeypatch.setattr(server, "_oms", SimpleNamespace(persistence=persistence, event_emitter=None))

    client = TestClient(server.app)

    pending = client.get("/api/v1/idempotency/pending?stale_after_sec=0")
    assert pending.status_code == 200
    assert pending.json()[0]["idempotency_key"] == "idem-1"
    persistence.list_pending_idempotency.assert_awaited_once_with(stale_after_sec=0.0)

    missing_reason = client.post("/api/v1/idempotency/idem-1/resolve", json={"status": "DEFERRED", "reason": ""})
    assert missing_reason.status_code == 400
    assert "reason is required" in missing_reason.json()["detail"]

    resolved = client.post(
        "/api/v1/idempotency/idem-1/resolve",
        json={"status": "DEFERRED", "reason": "operator confirmed no broker order exists"},
    )
    assert resolved.status_code == 200
    assert resolved.json()["status"] == "DEFERRED"
    assert resolved.json()["message"] == "operator confirmed no broker order exists"
    persistence.resolve_idempotency.assert_awaited_once()
    assert persistence.resolve_idempotency.await_args.args[0] == "idem-1"
    assert persistence.resolve_idempotency.await_args.kwargs["reason"] == "operator confirmed no broker order exists"


def test_health_degrades_while_idempotency_ambiguity_is_unresolved(monkeypatch):
    class _State:
        def get_all_positions(self):
            return {}

    class _API:
        def get_circuit_breaker_status(self):
            return {"state": "CLOSED"}

    persistence = SimpleNamespace()
    persistence.consecutive_failures = 0
    persistence._is_connected = lambda: True
    persistence.idempotency_health = AsyncMock(
        return_value={
            "status": "degraded",
            "pending_count": 0,
            "ambiguous_count": 1,
        }
    )
    oms = SimpleNamespace(
        adapter=SimpleNamespace(api=_API()),
        state=_State(),
        persistence=persistence,
        require_persistence=True,
        stop_health_payload=lambda: {
            "stop_protection_status": "ok",
            "unprotected_positions_count": 0,
            "active_stop_count": 0,
            "triggered_stop_count": 0,
            "stop_watcher_price_stale_count": 0,
        },
    )
    monkeypatch.setattr(server, "_oms", oms)

    client = TestClient(server.app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["idempotency_status"] == "degraded"
    persistence.idempotency_health.assert_awaited_once()


def test_health_fails_closed_on_incomplete_or_malformed_stop_health(monkeypatch):
    class _State:
        def get_all_positions(self):
            return {}

    class _API:
        def get_circuit_breaker_status(self):
            return {"state": "CLOSED"}

    def _install_oms(stop_health):
        oms = SimpleNamespace(
            adapter=SimpleNamespace(api=_API()),
            state=_State(),
            persistence=None,
            require_persistence=False,
            stop_health_payload=lambda: stop_health,
        )
        monkeypatch.setattr(server, "_oms", oms)

    client = TestClient(server.app)

    _install_oms(
        {
            "stop_protection_status": "ok",
            "active_stop_count": 0,
            "triggered_stop_count": 0,
            "stop_watcher_price_stale_count": 0,
        }
    )
    missing_response = client.get("/health")
    assert missing_response.status_code == 200
    assert missing_response.json()["status"] == "error"
    assert "stop_health_missing(unprotected_positions_count)" in missing_response.json()["recon_status"]

    _install_oms(
        {
            "stop_protection_status": "ok",
            "unprotected_positions_count": "not-a-count",
            "active_stop_count": 0,
            "triggered_stop_count": 0,
            "stop_watcher_price_stale_count": 0,
        }
    )
    malformed_response = client.get("/health")
    assert malformed_response.status_code == 200
    assert malformed_response.json()["status"] == "error"
    assert "stop_health_invalid(unprotected_positions_count)" in malformed_response.json()["recon_status"]

    _install_oms(
        {
            "stop_protection_status": "ok",
            "unprotected_positions_count": 0,
            "active_stop_count": 1,
            "triggered_stop_count": 0,
            "stop_watcher_price_stale_count": 0,
        }
    )
    watcher_response = client.get("/health")
    assert watcher_response.status_code == 200
    assert watcher_response.json()["status"] == "error"
    assert "stop_watcher_missing" in watcher_response.json()["recon_status"]


def test_health_reports_stale_stop_prices_even_when_overall_already_degraded(monkeypatch):
    class _State:
        def get_all_positions(self):
            return {}

    class _API:
        def get_circuit_breaker_status(self):
            return {"state": "OPEN"}

    oms = SimpleNamespace(
        adapter=SimpleNamespace(api=_API()),
        state=_State(),
        persistence=None,
        require_persistence=False,
        stop_health_payload=lambda: {
            "stop_protection_status": "ok",
            "unprotected_positions_count": 0,
            "active_stop_count": 2,
            "triggered_stop_count": 1,
            "stop_watcher_last_check_age_sec": 5.0,
            "stop_watcher_price_stale_count": 1,
        },
    )
    monkeypatch.setattr(server, "_oms", oms)

    client = TestClient(server.app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["stop_protection_status"] == "degraded"
    assert payload["active_stop_count"] == 2
    assert payload["triggered_stop_count"] == 1
    assert payload["stop_watcher_price_stale_count"] == 1
    assert "stop_price_stale(1)" in payload["recon_status"]

"""Account-bound, read-only evidence from the same adapter used by OMS."""
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import math
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo


def broker_identity(api):
    mode = ("paper" if getattr(api, "is_paper_trading", None) is True else
            "live" if getattr(api, "is_paper_trading", None) is False else "unknown")
    account = getattr(api, "account_num", None)
    url = getattr(api, "using_url", None)
    parsed = urlsplit(url) if isinstance(url, str) else None
    host = parsed.hostname if parsed else None
    expected = {"paper": "openapivts.koreainvestment.com", "live": "openapi.koreainvestment.com"}
    verified = (mode in expected and host == expected[mode] and parsed.scheme == "https"
                and parsed.port == {"paper": 29443, "live": 9443}[mode]
                and isinstance(account, str) and bool(account.strip()))
    return {"mode": mode if verified else "unknown",
            "account_fingerprint": sha256(f"{mode}:{account}".encode()).hexdigest() if verified else "",
            "endpoint_host": host}


def _number(value, *, positive=False):
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError("Invalid broker numeric observation")
    if positive and float(value) <= 0:
        raise ValueError("Missing positive broker cost basis")
    return float(value)


async def collect_broker_snapshot(adapter, *, oms_id, account_alias, history_start=None):
    """No orders, account mutations or inferred fills. Errors never mean empty."""
    identity = broker_identity(adapter.api)
    if identity["mode"] != "paper" or not oms_id or not account_alias:
        raise ValueError("Verified KIS paper route and named account scope required")
    started = datetime.now(timezone.utc)
    today = started.astimezone(ZoneInfo("Asia/Seoul")).date()
    first = history_start or today
    if first > today:
        raise ValueError("Execution history start cannot be in the future")
    positions, equity = await adapter.get_balance_snapshot()
    orders = await adapter.get_orders()
    history = await adapter.get_execution_orders(first.strftime("%Y%m%d"), today.strftime("%Y%m%d"))
    cash = await adapter.get_buyable_cash()
    if not all(item.ok is True for item in (positions, orders, history)) or equity is None or cash is None:
        raise RuntimeError("Incomplete broker observation: positions, working orders, executions and buying power required")
    _number(equity)
    _number(cash)
    quantities, basis = {}, {}
    for p in positions.data:
        qty = _number(p.qty)
        if qty != int(qty) or p.symbol in quantities:
            raise ValueError("Invalid or duplicated broker position")
        if qty:
            quantities[p.symbol] = int(qty)
            basis[p.symbol] = _number(p.avg_price, positive=True)
    for order in orders.data + history.data:
        if (not order.order_id or order.side not in {"BUY", "SELL"}
                or not 0 <= order.filled_qty <= order.qty or order.qty <= 0):
            raise ValueError("Invalid broker order observation")
    return {"schema_version": "kis-paper-broker-snapshot-v1", "source": "KISExecutionAdapter",
            "oms_id": oms_id, "account_alias": account_alias, **identity,
            "as_of": started.isoformat(), "completed_at": datetime.now(timezone.utc).isoformat(),
            "positions": quantities, "cost_basis": basis, "equity_krw": equity,
            "buyable_cash_krw": cash, "working_orders": [asdict(x) for x in orders.data],
            "execution_orders": [asdict(x) for x in history.data],
            "history_start": first.isoformat(), "history_end": today.isoformat(),
            "queries_complete": True,
            "unverified": ["fees_and_realized_net_pnl", "connected_order_lifecycle", "market_session_and_strategy_artifacts"]}

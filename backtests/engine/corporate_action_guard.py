"""Abort unsupported action exposure; never cancel an otherwise valid trade."""
from datetime import date


def verify_zero_exposure(broker, day, events):
    if not isinstance(day, date): raise ValueError("Expected session date")
    records = []
    for event in events:
        if event["ex_date"] != day.isoformat(): continue
        symbol = event["symbol"]
        holdings = {str(key): pos.qty for key, pos in broker.positions.items()
                    if key[1] == symbol and pos.qty}
        orders = [order.order_id for order in broker.orders if order.symbol == symbol]
        if holdings or orders:
            raise ValueError(f"Unsupported corporate-action exposure {symbol} {day}; "
                             "implement verified entitlement and pending-order accounting before replay")
        records.append({"symbol": symbol, "date": str(day), "type": event["type"],
                        "held_shares": 0, "pending_orders": 0, "account_adjustment_required": False})
    return records

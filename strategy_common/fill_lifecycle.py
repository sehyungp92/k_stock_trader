"""Execution delta identity and completion shared by cores and runtime adapters."""
import json
import math


def entry_order_key(fill):
    meta = dict(getattr(fill, "metadata", {}) or {})
    return json.dumps([meta.get("account_id", ""), meta.get("order_date", fill.timestamp.date().isoformat()),
                       str(fill.order_id)], separators=(",", ":"))


def execution_key(fill):
    meta = dict(getattr(fill, "metadata", {}) or {})
    identity = meta.get("execution_id") or meta.get("kis_exec_id")
    if not identity and meta.get("filled_qty_after") is not None:
        identity = f"cumulative:{meta['filled_qty_after']}"
    if not identity:
        return None
    return json.dumps([meta.get("account_id", ""), meta.get("order_date", fill.timestamp.date().isoformat()),
        str(fill.order_id), str(identity)], separators=(",", ":"))


def is_duplicate_fill(meta, fill, *, namespace="core_executions"):
    key = execution_key(fill)
    previous = meta.get(namespace, {}).get(key) if key else None
    if previous is None:
        return False
    if previous != [int(fill.qty), float(fill.price), str(fill.side), str(fill.symbol)]:
        raise ValueError("Execution identity was reused with different content")
    return True


def record_fill_delta(meta, fill, *, namespace="core_executions"):
    if int(fill.qty) <= 0 or not math.isfinite(float(fill.price)) or float(fill.price) <= 0:
        raise ValueError("Execution requires positive quantity and finite positive price")
    if is_duplicate_fill(meta, fill, namespace=namespace):
        return False
    key = execution_key(fill)
    processed = meta.setdefault(namespace, {})
    signature = [int(fill.qty), float(fill.price), str(fill.side), str(fill.symbol)]
    if key and key in processed:
        if processed[key] != signature:
            raise ValueError("Execution identity was reused with different content")
        return False
    detail = dict(fill.metadata or {})
    if detail.get("filled_qty_before") is not None and detail.get("filled_qty_after") is not None:
        if int(detail["filled_qty_after"]) - int(detail["filled_qty_before"]) != int(fill.qty):
            raise ValueError("Cumulative execution evidence disagrees with fill delta")
        cumulative = meta.setdefault(f"{namespace}_quantities", {})
        order_key = json.dumps([detail.get("account_id", ""),
            detail.get("order_date", fill.timestamp.date().isoformat()), str(fill.order_id)])
        previous = cumulative.get(order_key, 0)
        if int(detail["filled_qty_before"]) != previous:
            raise ValueError("Execution gap or out-of-order delivery; catch up before applying")
        cumulative[order_key] = int(detail["filled_qty_after"])
    if key:
        processed[key] = signature
    return True


def order_fill_complete(fill, metadata):
    intended = metadata.get("order_qty") or metadata.get("submitted_qty")
    after = metadata.get("filled_qty_after")
    return intended is not None and after is not None and int(after) == int(intended)

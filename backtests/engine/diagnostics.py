"""Measured comparisons of fixed-input replay paths; these never promote runs."""
from collections import Counter
from dataclasses import asdict

from backtests.auto.shared.cache_keys import stable_signature
from backtests.engine.integrity import audit_broker


def paired_report(legacy, corrected):
    def fills(result):
        return [{k: v for k, v in asdict(fill).items() if k not in {"metadata", "order_id"}}
                for fill in result.broker.fills]

    left, right = fills(legacy), fills(corrected)
    a, b = Counter(map(stable_signature, left)), Counter(map(stable_signature, right))
    first = next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
                 min(len(left), len(right)) if len(left) != len(right) else None)
    audit = audit_broker(corrected.broker)
    return {
        "legacy_clock": legacy.broker.execution_clock,
        "corrected_clock": corrected.broker.execution_clock,
        "first_different_fill_index": first,
        "first_legacy_fill": left[first] if first is not None and first < len(left) else None,
        "first_corrected_fill": right[first] if first is not None and first < len(right) else None,
        "legacy_only_fills": sum((a - b).values()), "corrected_only_fills": sum((b - a).values()),
        "unchanged_fills": sum((a & b).values()),
        "legacy_final_cash": legacy.broker.cash, "corrected_final_cash": corrected.broker.cash,
        "legacy_final_equity": legacy.equity_curve[-1], "corrected_final_equity": corrected.equity_curve[-1],
        "legacy_rejections": len(legacy.broker.rejected_orders),
        "corrected_rejections": len(corrected.broker.rejected_orders),
        "corrected_rejected_admissions": [row for row in corrected.broker.journal if row["event"] == "reject"],
        "corrected_terminal_positions": [asdict(p) for p in corrected.broker.positions.values()],
        "corrected_audit": audit,
        "attribution_limit": "First changed fill includes downstream account effects; it does not prove causal admission attribution.",
    }

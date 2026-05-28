from __future__ import annotations

from collections import Counter


def format_kalcb_diagnostics(result) -> str:
    trades = list(result.trades)
    decisions = list(result.decisions)
    entries = Counter(str(trade.route_metadata.get("entry_type", "UNKNOWN")) for trade in trades)
    exits = Counter(str(trade.exit_reason or "UNKNOWN") for trade in trades)
    rejects = Counter(str(decision.reason) for decision in decisions if decision.decision_code == "entry_rejected")
    lines = [
        "KALCB diagnostics",
        f"source_fingerprint: {result.source_fingerprint}",
        f"candidate_snapshot_hash: {result.candidate_snapshot_hash}",
        f"feature_bundle_hash: {result.feature_bundle_hash}",
        f"trades: {int(result.metrics.get('total_trades', 0))}",
        f"decision_count: {int(result.metrics.get('decision_count', 0))}",
        f"profit_factor: {result.metrics.get('profit_factor', 0.0):.3f}",
        f"expected_total_r: {result.metrics.get('expected_total_r', 0.0):.3f}",
        f"max_drawdown_pct: {result.metrics.get('max_drawdown_pct', 0.0):.4f}",
        f"active_symbol_max: {int(result.metrics.get('active_symbol_max', 0))}",
        f"entries: {dict(entries)}",
        f"exits: {dict(exits)}",
        f"top_rejections: {dict(rejects.most_common(8))}",
    ]
    if not decisions:
        lines.append("warning: no decision events emitted")
    return "\n".join(lines)

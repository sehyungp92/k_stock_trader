"""
Daily Snapshot — End-of-day aggregate for a single bot.

Reads today's trade events, missed opportunities, process scores, and errors
from JSONL files, then computes a comprehensive daily rollup. The sidecar
picks up the saved JSON file and forwards it to the central relay.
"""

from __future__ import annotations

import json
import hashlib
import os

from .execution_accounting import daily_accounting, economics, outcome_stats, event_day
from collections import Counter
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DailySnapshot:
    """End-of-day aggregate for a single bot."""
    date: str                          # YYYY-MM-DD
    bot_id: str
    strategy_type: str

    schema_version: str = "daily_execution_accounting_v2"
    execution_count: int = 0
    unknown_cost_execution_count: int = 0
    unclassified_trade_count: int = 0
    costs_complete: bool = True
    known_fees: float = 0.0
    accounting_day_basis: str = "UTC"
    completed_trades: list = field(default_factory=list)
    snapshot_revision: str = ""

    # Trade counts
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    breakeven_count: int = 0

    # PnL
    gross_pnl: float = 0.0
    net_pnl: Optional[float] = 0.0              # after fees
    total_fees: Optional[float] = 0.0
    best_trade_pnl: Optional[float] = 0.0
    worst_trade_pnl: Optional[float] = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Risk
    max_drawdown_pct: Optional[float] = 0.0     # intraday peak-to-trough (cumulative PnL)
    max_exposure: float = 0.0         # max simultaneous position value
    max_concurrent_positions: int = 0
    total_risk_deployed_pct: float = 0.0
    profit_factor: Optional[float] = 0.0        # gross wins / gross losses
    win_rate: Optional[float] = 0.0
    exposure_pct: float = 0.0         # % of time in position

    # Rolling metrics (require historical data)
    sharpe_rolling_30d: Optional[float] = None
    sortino_rolling_30d: Optional[float] = None
    calmar_rolling_30d: Optional[float] = None

    # Missed opportunities
    missed_count: int = 0
    missed_would_have_won: int = 0    # where first_hit == "TP"
    missed_potential_pnl: float = 0.0  # sum of hypothetical TP PnL
    top_missed_filter: str = ""       # filter that blocked the most winners

    # Process quality
    avg_process_quality: float = 0.0
    process_scores_distribution: Dict[str, int] = field(default_factory=dict)
    # e.g. {"good_process": 5, "neutral": 2, "bad_process": 1}
    root_cause_distribution: Dict[str, int] = field(default_factory=dict)
    # e.g. {"normal_win": 3, "normal_loss": 2, "regime_mismatch": 1, ...}

    # Regime breakdown
    regime_breakdown: Dict[str, dict] = field(default_factory=dict)
    # e.g. {"trending_up": {"trades": 3, "pnl": 150, "win_rate": 0.67}, ...}

    # Exit efficiency
    avg_exit_efficiency: Optional[float] = None

    # Execution quality
    avg_entry_slippage_bps: Optional[float] = None
    avg_exit_slippage_bps: Optional[float] = None
    avg_entry_latency_ms: Optional[float] = None

    # Per-strategy breakdown (single-key dict for mono-strategy processes)
    per_strategy_summary: dict = field(default_factory=dict)

    # Experiment A/B variant breakdown
    experiment_breakdown: dict = field(default_factory=dict)
    # Key: "{experiment_id}:{experiment_variant}"
    # Value: per-variant aggregate stats

    # Active experiment metadata from registry
    active_experiments: dict = field(default_factory=dict)

    # Health
    error_count: int = 0
    uptime_pct: float = 100.0
    data_gaps: int = 0                 # number of missing snapshot intervals

    def to_dict(self) -> dict:
        d = asdict(self)
        # Coerce None → 0.0 for fields the assistant schema types as non-Optional float
        for key in ("sharpe_rolling_30d", "sortino_rolling_30d", "calmar_rolling_30d"):
            if d.get(key) is None:
                d[key] = 0.0
        return d


class DailySnapshotBuilder:
    """
    Reads today's trade events, missed opportunities, and process scores,
    then computes the daily aggregate.

    Usage:
        builder = DailySnapshotBuilder(config)
        snapshot = builder.build(date_str="2026-03-01")
        builder.save(snapshot)
    """

    def __init__(self, config: dict, experiment_registry=None):
        self.bot_id = config["bot_id"]
        self.strategy_type = config.get("strategy_type", "unknown")
        self.data_dir = Path(config["data_dir"])
        self._experiment_registry = experiment_registry

    def build(self, date_str: str = None) -> DailySnapshot:
        """Build daily snapshot for the given date (default: today)."""
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Closing outcomes can include fragments from previous days. Corrections
        # restate the original day, so retain all observation revisions.
        history = []
        for path in sorted((self.data_dir / "trades").glob("trades_*.jsonl")):
            history.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        trades = [t for t in history if event_day(t, "entry_time" if t.get("stage") == "entry" else "exit_time") == date_str]
        missed = self._load_missed(date_str)
        scores = self._load_scores(date_str)
        errors = self._load_errors(date_str)

        snapshot = DailySnapshot(
            date=date_str,
            bot_id=self.bot_id,
            strategy_type=self.strategy_type,
        )

        metrics, executions, completed = daily_accounting(history, date_str)
        for key, value in metrics.items():
            setattr(snapshot, key, value)
        snapshot.completed_trades = completed
        for target, source, digits in (
            ("avg_entry_slippage_bps", "entry_slippage_bps", 2),
            ("avg_exit_slippage_bps", "exit_slippage_bps", 2),
            ("avg_entry_latency_ms", "entry_latency_ms", 1),
            ("avg_exit_efficiency", "exit_efficiency", 4),
        ):
            values = [r[source] for r in executions if r.get(source) is not None]
            setattr(snapshot, target, round(sum(values) / len(values), digits) if values else None)

        snapshot.regime_breakdown = self._group_accounting(executions, completed, "market_regime")
        snapshot.per_strategy_summary = self._group_accounting(
            executions, completed, "strategy_id", self.strategy_type.upper())
        snapshot.experiment_breakdown = self._group_accounting(executions, completed, "experiment_id")
        snapshot.experiment_breakdown.pop("unknown", None)

        entries = {t["trade_id"]: t for t in history if t.get("stage") == "entry"}
        snapshot.total_risk_deployed_pct = round(sum(
            (t.get("sizing_context") or {}).get("target_risk_pct", 0) or 0
            for t in entries.values() if event_day(t, "entry_time") == date_str), 6)
        # A partial exit does not close a position. Include overnight positions.
        from .execution_accounting import closed_lifecycles, project_executions
        all_closed = {t["trade_id"]: t for t in closed_lifecycles(project_executions(history))}
        position_events = []
        for trade_id, entry in entries.items():
            start = event_day(entry, "entry_time")
            closing = all_closed.get(trade_id)
            end = event_day(closing) if closing else "9999-12-31"
            if start <= date_str <= end:
                position_events.append((entry["entry_time"] if start == date_str else "", 1))
                if end == date_str:
                    position_events.append((closing["exit_time"], -1))
        active = peak = 0
        for _, delta in sorted(position_events, key=lambda e: (e[0], -e[1])):
            active += delta
            peak = max(peak, active)
        snapshot.max_concurrent_positions = peak

        if not snapshot.costs_complete:
            snapshot.max_drawdown_pct = None
        else:
            cumulative = peak_pnl = drawdown = 0.0
            for row in executions:
                cumulative += row["net_pnl"]
                peak_pnl = max(peak_pnl, cumulative)
                drawdown = max(drawdown, peak_pnl - cumulative)
            snapshot.max_drawdown_pct = round(drawdown / peak_pnl * 100, 4) if peak_pnl else (100.0 if drawdown else 0.0)

        # --- MISSED OPPORTUNITIES ---
        snapshot.missed_count = len(missed)
        missed_winners = [m for m in missed if m.get("first_hit") == "TP"]
        snapshot.missed_would_have_won = len(missed_winners)

        if missed:
            # Sum potential PnL from missed winners
            snapshot.missed_potential_pnl = round(
                sum(m.get("hypothetical_pnl", 0) or 0 for m in missed_winners), 4
            )

            # Find the filter that blocked the most would-have-won trades
            filter_win_counts: Counter = Counter()
            for m in missed_winners:
                filter_win_counts[m.get("blocked_by", "unknown")] += 1
            if filter_win_counts:
                snapshot.top_missed_filter = filter_win_counts.most_common(1)[0][0]

        # --- PROCESS QUALITY ---
        if scores:
            quality_scores = [s.get("process_quality_score", 50) for s in scores]
            snapshot.avg_process_quality = round(sum(quality_scores) / len(quality_scores), 1)

            # Classification distribution
            classifications = Counter(s.get("classification", "neutral") for s in scores)
            snapshot.process_scores_distribution = dict(classifications)

            # Root cause distribution
            all_causes: list = []
            for s in scores:
                all_causes.extend(s.get("root_causes", []))
            snapshot.root_cause_distribution = dict(Counter(all_causes))

        # --- ERRORS ---
        snapshot.error_count = len(errors)

        # --- ACTIVE EXPERIMENTS ---
        if self._experiment_registry is not None:
            try:
                snapshot.active_experiments = self._experiment_registry.export_active(date_str)
            except Exception:
                pass

        return snapshot

    @staticmethod
    def _group_accounting(executions, completed, field, default="unknown"):
        def key(row):
            value = row.get(field) or default
            if field == "experiment_id" and value != default:
                value += ":" + (row.get("experiment_variant") or "")
            return value
        result = {}
        for group in sorted({key(r) for r in [*executions, *completed]}):
            stats = outcome_stats([r for r in completed if key(r) == group])
            result[group] = {**economics([r for r in executions if key(r) == group]),
                             **stats, "trades": stats["total_trades"]}
        return result

    def save(self, snapshot: DailySnapshot):
        """Append content-addressed revisions; retain the latest JSON for local readers."""
        daily_dir = self.data_dir / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        payload = snapshot.to_dict()
        payload.pop("snapshot_revision", None)
        revision = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
        snapshot.snapshot_revision = revision
        payload["snapshot_revision"] = revision
        filepath = daily_dir / f"daily_{snapshot.date}.json"
        revisions = filepath.with_suffix(".jsonl")
        from filelock import FileLock
        with FileLock(str(filepath) + ".lock"):
            previous = json.loads(filepath.read_text(encoding="utf-8")) if filepath.exists() else {}
            if revisions.exists():
                content = revisions.read_bytes()
                # A newline commits a journal record. The sidecar never exports
                # an unterminated v2 record. Discard an interrupted append only.
                boundary = content.rfind(b"\n") + 1
                if boundary != len(content):
                    with revisions.open("r+b") as stream:
                        stream.truncate(boundary)
                        stream.flush()
                        os.fsync(stream.fileno())
                records = [json.loads(line) for line in content[:boundary].splitlines() if line.strip()]
                if records:
                    previous = records[-1]
            if previous.get("snapshot_revision") != revision:
                payload["snapshot_sequence"] = int(previous.get("snapshot_sequence", 0)) + 1
                with revisions.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(payload, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            else:
                payload = previous
            temporary = filepath.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            os.replace(temporary, filepath)

    def _load_jsonl(self, directory: str, prefix: str, date_str: str) -> list:
        """Load events from a JSONL file for the given date.

        Args:
            directory: subdirectory under data_dir (e.g. "trades", "missed")
            prefix: filename prefix (e.g. "trades", "missed", "scores")
            date_str: date string in YYYY-MM-DD format

        Returns:
            List of parsed event dicts. Empty list if file missing or empty.
        """
        filepath = self.data_dir / directory / f"{prefix}_{date_str}.jsonl"
        if not filepath.exists():
            return []
        events = []
        for line in filepath.read_text().strip().split("\n"):
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return events

    def _load_trades(self, date_str: str) -> list:
        return self._load_jsonl("trades", "trades", date_str)

    def _load_missed(self, date_str: str) -> list:
        return self._load_jsonl("missed", "missed", date_str)

    def _load_scores(self, date_str: str) -> list:
        return self._load_jsonl("scores", "scores", date_str)

    def _load_errors(self, date_str: str) -> list:
        return self._load_jsonl("errors", "instrumentation_errors", date_str)

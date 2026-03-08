"""
Daily Snapshot — End-of-day aggregate for a single bot.

Reads today's trade events, missed opportunities, process scores, and errors
from JSONL files, then computes a comprehensive daily rollup. The sidecar
picks up the saved JSON file and forwards it to the central relay.
"""

from __future__ import annotations

import json
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

    # Trade counts
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    breakeven_count: int = 0

    # PnL
    gross_pnl: float = 0.0
    net_pnl: float = 0.0              # after fees
    total_fees: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Risk
    max_drawdown_pct: float = 0.0     # intraday peak-to-trough (cumulative PnL)
    max_exposure: float = 0.0         # max simultaneous position value
    max_concurrent_positions: int = 0
    total_risk_deployed_pct: float = 0.0
    profit_factor: float = 0.0        # gross wins / gross losses
    win_rate: float = 0.0
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
        return asdict(self)


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

        trades = self._load_trades(date_str)
        missed = self._load_missed(date_str)
        scores = self._load_scores(date_str)
        errors = self._load_errors(date_str)

        snapshot = DailySnapshot(
            date=date_str,
            bot_id=self.bot_id,
            strategy_type=self.strategy_type,
        )

        # --- TRADE AGGREGATES ---
        completed = [t for t in trades if t.get("stage") == "exit" and t.get("pnl") is not None]
        snapshot.total_trades = len(completed)

        if completed:
            pnls = [t["pnl"] for t in completed]
            fees = [t.get("fees_paid", 0) or 0 for t in completed]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]

            snapshot.win_count = len(wins)
            snapshot.loss_count = len(losses)
            snapshot.breakeven_count = len([p for p in pnls if p == 0])
            snapshot.gross_pnl = round(sum(pnls) + sum(fees), 4)  # add back fees for gross
            snapshot.net_pnl = round(sum(pnls), 4)
            snapshot.total_fees = round(sum(fees), 4)
            snapshot.best_trade_pnl = round(max(pnls), 4)
            snapshot.worst_trade_pnl = round(min(pnls), 4)
            snapshot.avg_win = round(sum(wins) / len(wins), 4) if wins else 0
            snapshot.avg_loss = round(sum(losses) / len(losses), 4) if losses else 0
            snapshot.win_rate = round(len(wins) / len(completed), 4) if completed else 0

            gross_wins = sum(wins) if wins else 0
            gross_losses = abs(sum(losses)) if losses else 0
            snapshot.profit_factor = round(gross_wins / gross_losses, 4) if gross_losses > 0 else float('inf')

            # Slippage averages
            entry_slips = [t.get("entry_slippage_bps") for t in completed if t.get("entry_slippage_bps") is not None]
            exit_slips = [t.get("exit_slippage_bps") for t in completed if t.get("exit_slippage_bps") is not None]
            latencies = [t.get("entry_latency_ms") for t in completed if t.get("entry_latency_ms") is not None]

            snapshot.avg_entry_slippage_bps = round(sum(entry_slips) / len(entry_slips), 2) if entry_slips else None
            snapshot.avg_exit_slippage_bps = round(sum(exit_slips) / len(exit_slips), 2) if exit_slips else None
            snapshot.avg_entry_latency_ms = round(sum(latencies) / len(latencies), 1) if latencies else None

            # Regime breakdown
            regime_data: Dict[str, dict] = {}
            for t in completed:
                regime = t.get("market_regime", "unknown")
                if regime not in regime_data:
                    regime_data[regime] = {"trades": 0, "pnl": 0, "wins": 0}
                regime_data[regime]["trades"] += 1
                regime_data[regime]["pnl"] += t["pnl"]
                if t["pnl"] > 0:
                    regime_data[regime]["wins"] += 1
            for regime, data in regime_data.items():
                data["pnl"] = round(data["pnl"], 4)
                data["win_rate"] = round(data["wins"] / data["trades"], 4) if data["trades"] > 0 else 0
            snapshot.regime_breakdown = regime_data

            # Exit efficiency average
            exit_effs = [t.get("exit_efficiency") for t in completed if t.get("exit_efficiency") is not None]
            snapshot.avg_exit_efficiency = round(sum(exit_effs) / len(exit_effs), 4) if exit_effs else None

            # Total risk deployed (sum of target_risk_pct from entry sizing_context)
            entries = [t for t in trades if t.get("stage") == "entry"]
            risk_sum = 0.0
            for t in entries:
                sc = t.get("sizing_context")
                if sc and sc.get("target_risk_pct") is not None:
                    risk_sum += sc["target_risk_pct"]
            snapshot.total_risk_deployed_pct = round(risk_sum, 6)

            # Max concurrent positions — track entry/exit events by timestamp
            events: List[tuple] = []  # (timestamp_str, delta)
            for t in trades:
                if t.get("stage") == "entry" and t.get("entry_time"):
                    events.append((t["entry_time"], 1))
                elif t.get("stage") == "exit" and t.get("exit_time"):
                    events.append((t["exit_time"], -1))
            events.sort(key=lambda x: x[0])
            concurrent = 0
            peak = 0
            for _, delta in events:
                concurrent += delta
                peak = max(peak, concurrent)
            snapshot.max_concurrent_positions = peak

            # Max drawdown (peak-to-trough of cumulative PnL by exit time)
            sorted_exits = sorted(completed, key=lambda t: t.get("exit_time", ""))
            cum_pnl = 0.0
            peak_pnl = 0.0
            max_dd = 0.0
            for t in sorted_exits:
                cum_pnl += t["pnl"]
                if cum_pnl > peak_pnl:
                    peak_pnl = cum_pnl
                dd = peak_pnl - cum_pnl
                if dd > max_dd:
                    max_dd = dd
            if peak_pnl > 0:
                snapshot.max_drawdown_pct = round(max_dd / peak_pnl * 100, 4)
            elif max_dd > 0:
                # All losses — drawdown is 100% from zero peak
                snapshot.max_drawdown_pct = 100.0

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

        # --- PER-STRATEGY SUMMARY ---
        # Mono-strategy: one key matching strategy_type
        snapshot.per_strategy_summary = {
            self.strategy_type: {
                "trades": snapshot.total_trades,
                "win_count": snapshot.win_count,
                "loss_count": snapshot.loss_count,
                "gross_pnl": snapshot.gross_pnl,
                "net_pnl": snapshot.net_pnl,
                "win_rate": (
                    snapshot.win_count / snapshot.total_trades * 100
                    if snapshot.total_trades > 0 else 0
                ),
                "avg_win": snapshot.avg_win,
                "avg_loss": snapshot.avg_loss,
                "best_trade_pnl": snapshot.best_trade_pnl,
                "worst_trade_pnl": snapshot.worst_trade_pnl,
                "avg_entry_slippage_bps": snapshot.avg_entry_slippage_bps,
            }
        }

        # --- EXPERIMENT BREAKDOWN ---
        snapshot.experiment_breakdown = self._build_experiment_breakdown(completed)

        # --- ACTIVE EXPERIMENTS ---
        if self._experiment_registry is not None:
            try:
                snapshot.active_experiments = self._experiment_registry.export_active(date_str)
            except Exception:
                pass

        return snapshot

    def _build_experiment_breakdown(self, completed_trades: list) -> dict:
        """Group completed trades by experiment_id:experiment_variant."""
        groups: dict[str, list[dict]] = {}
        for t in completed_trades:
            exp_id = t.get("experiment_id") or ""
            exp_var = t.get("experiment_variant") or ""
            if not exp_id:
                continue
            key = f"{exp_id}:{exp_var}"
            groups.setdefault(key, []).append(t)

        breakdown = {}
        for key, trades in groups.items():
            exp_id, exp_var = key.split(":", 1)
            wins = [t for t in trades if (t.get("pnl") or 0) > 0]
            losses = [t for t in trades if (t.get("pnl") or 0) < 0]
            gross_pnl = sum(t.get("pnl", 0) + t.get("fees_paid", 0) for t in trades)
            net_pnl = sum(t.get("pnl", 0) for t in trades)
            scores = [t.get("process_quality_score", 0) for t in trades
                      if t.get("process_quality_score") is not None]
            mfe_vals = [t["mfe_pct"] for t in trades if t.get("mfe_pct") is not None]
            mae_vals = [t["mae_pct"] for t in trades if t.get("mae_pct") is not None]

            breakdown[key] = {
                "experiment_id": exp_id,
                "experiment_variant": exp_var,
                "trades": len(trades),
                "win_count": len(wins),
                "loss_count": len(losses),
                "gross_pnl": round(gross_pnl, 2),
                "net_pnl": round(net_pnl, 2),
                "win_rate": round(len(wins) / len(trades), 4) if trades else 0.0,
                "avg_win": round(
                    sum(t["pnl"] for t in wins) / len(wins), 2
                ) if wins else 0.0,
                "avg_loss": round(
                    sum(t["pnl"] for t in losses) / len(losses), 2
                ) if losses else 0.0,
                "avg_process_quality": round(
                    sum(scores) / len(scores), 1
                ) if scores else 0.0,
                "avg_mfe_pct": round(
                    sum(mfe_vals) / len(mfe_vals), 4
                ) if mfe_vals else None,
                "avg_mae_pct": round(
                    sum(mae_vals) / len(mae_vals), 4
                ) if mae_vals else None,
            }

        return breakdown

    def save(self, snapshot: DailySnapshot):
        """Save daily snapshot as a single JSON file."""
        daily_dir = self.data_dir / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        filepath = daily_dir / f"daily_{snapshot.date}.json"
        with open(filepath, "w") as f:
            json.dump(snapshot.to_dict(), f, indent=2, default=str)

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

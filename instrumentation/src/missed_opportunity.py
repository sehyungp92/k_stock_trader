"""
Missed Opportunity Logger — records signals blocked by filters/risk limits
and backfills hypothetical outcomes using simulation policies.

Adapted for Korean equity market (KRX):
- Side is always LONG (no retail short selling)
- Fees: KRX commission ~0.015% + securities tax ~0.18% on sell = ~0.20% RT
- Slippage: KRX tick sizes vary by price band, 5bps default is reasonable
- Data provider: KIS API get_minute_bars(symbol, minutes=N) -> DataFrame
"""

from __future__ import annotations

import json
import hashlib
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict

from loguru import logger

from .event_metadata import EventMetadata, create_event_metadata
from .market_snapshot import MarketSnapshot, MarketSnapshotService


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SimulationPolicy:
    """
    Defines assumptions for hypothetical outcome calculation.
    Loaded from instrumentation/config/simulation_policies.yaml.
    Must be defined per strategy -- different strategies have different TP/SL logic.

    Korean market defaults:
    - fee_bps=20  (~0.20% round trip: commission + securities tax)
    - slippage_bps=5  (1-3 ticks typical on KRX)
    - Side is always LONG (no retail short selling)
    """
    entry_fill_model: str = "mid"         # "mid" | "bid_ask" | "next_trade"
    slippage_model: str = "fixed_bps"     # "fixed_bps" | "spread_proportional" | "empirical"
    slippage_bps: float = 5.0             # used if model is fixed_bps
    fees_included: bool = True
    fee_bps: float = 20.0                 # KRX: commission ~1.5bps + tax ~18bps on sell
    tp_sl_logic: str = "atr_based"        # "fixed_pct" | "atr_based" | "trailing"
    tp_value: float = 2.0                 # multiplier (ATR) or percentage, depends on logic
    sl_value: float = 1.0                 # multiplier (ATR) or percentage
    max_hold_bars: int = 100              # timeout for simulation (1-min bars)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MissedOpportunityEvent:
    """A signal that fired but was not executed."""
    event_metadata: dict
    market_snapshot: dict                  # snapshot at signal time

    bot_id: str = ""
    pair: str = ""                         # KRX stock code, e.g. "005930"
    side: str = "LONG"                     # always LONG for Korean retail
    signal: str = ""                       # human-readable signal description
    signal_id: str = ""                    # machine identifier
    signal_strength: float = 0.0
    signal_time: str = ""                  # when the signal fired (KST ISO)
    blocked_by: str = ""                   # which filter or limit blocked it
    block_reason: str = ""                 # additional context on why

    hypothetical_entry_price: float = 0.0  # price used for simulation

    # Backfilled outcomes (None until computed)
    outcome_1h: Optional[float] = None     # price 1h after signal
    outcome_4h: Optional[float] = None
    outcome_24h: Optional[float] = None
    outcome_pnl_1h: Optional[float] = None    # hypothetical PnL % after 1h
    outcome_pnl_4h: Optional[float] = None
    outcome_pnl_24h: Optional[float] = None
    would_have_hit_tp: Optional[bool] = None
    would_have_hit_sl: Optional[bool] = None
    bars_to_tp: Optional[int] = None       # how many 1-min bars until TP hit
    bars_to_sl: Optional[int] = None
    first_hit: Optional[str] = None        # "TP" | "SL" | "TIMEOUT" | "PENDING"

    # Simulation transparency
    simulation_policy: Optional[dict] = None    # which assumptions were used
    simulation_confidence: float = 0.0          # 0-1, how reliable is this
    assumption_tags: List[str] = field(default_factory=list)
    backfill_status: str = "pending"            # "pending" | "partial" | "complete" | "failed"

    # Strategy context
    strategy_params_at_signal: Optional[dict] = None
    market_regime: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class MissedOpportunityLogger:
    """
    Logs missed opportunities and manages outcome backfill.

    Usage:
        mol = MissedOpportunityLogger(config, snapshot_service)

        # When a signal is blocked by a gate/filter:
        mol.log_missed(
            pair="005930",
            side="LONG",
            signal="momentum breakout confirmed",
            signal_id="kmp_breakout",
            signal_strength=0.75,
            blocked_by="volume_gate",
            block_reason="Volume ratio 0.8x below threshold 1.5x",
            strategy_params={...},
            strategy_type="kmp",
        )

        # Periodically (every 5 minutes or so):
        mol.run_backfill(data_provider=kis_api)
    """

    def __init__(self, config: dict, snapshot_service: MarketSnapshotService):
        self.bot_id = config.get("bot_id", "unknown")
        self.data_dir = Path(config.get("data_dir", "instrumentation/data")) / "missed"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_service = snapshot_service
        self.data_source_id = config.get("data_source_id", "kis_api")

        # Load simulation policies from YAML (falls back to defaults)
        self.simulation_policies = self._load_simulation_policies(config)

        # Pending backfills queue (thread-safe)
        self._pending_backfills: List[Dict] = []
        self._backfill_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Policy loading
    # ------------------------------------------------------------------

    def _load_simulation_policies(self, config: dict) -> Dict[str, SimulationPolicy]:
        """Load per-strategy simulation policies from YAML config file."""
        policies: Dict[str, SimulationPolicy] = {}

        # Try multiple paths for the policy file
        search_paths = [
            Path(config.get("policy_file", "")),
            Path("instrumentation/config/simulation_policies.yaml"),
            Path(__file__).resolve().parent.parent / "config" / "simulation_policies.yaml",
        ]

        for policy_file in search_paths:
            if not policy_file.is_file():
                continue
            try:
                import yaml
                with open(policy_file) as f:
                    raw = yaml.safe_load(f)
                if not raw or "simulation_policies" not in raw:
                    continue
                for name, params in raw["simulation_policies"].items():
                    if isinstance(params, dict):
                        policies[name] = SimulationPolicy(**params)
                logger.info(
                    f"Loaded {len(policies)} simulation policies from {policy_file}"
                )
                break
            except ImportError:
                logger.warning("PyYAML not installed -- using default simulation policy")
                break
            except Exception as e:
                logger.warning(f"Failed to load simulation policies from {policy_file}: {e}")
                continue

        # Always ensure a default policy exists
        if "default" not in policies:
            policies["default"] = SimulationPolicy()

        return policies

    def _get_policy(self, strategy_type: Optional[str] = None) -> SimulationPolicy:
        """Get simulation policy for a strategy, falling back to default."""
        if strategy_type and strategy_type.lower() in self.simulation_policies:
            return self.simulation_policies[strategy_type.lower()]
        return self.simulation_policies.get("default", SimulationPolicy())

    # ------------------------------------------------------------------
    # Hypothetical entry price
    # ------------------------------------------------------------------

    def _compute_hypothetical_entry(
        self, snapshot: MarketSnapshot, side: str, policy: SimulationPolicy
    ) -> float:
        """
        Compute the hypothetical entry price based on simulation policy.

        For Korean equities, side is always LONG.  Slippage is added
        (we assume we'd buy at a slightly worse price).
        """
        if policy.entry_fill_model == "mid":
            base_price = snapshot.mid
        elif policy.entry_fill_model == "bid_ask":
            # LONG always buys at ask
            base_price = snapshot.ask if side == "LONG" else snapshot.bid
        elif policy.entry_fill_model == "next_trade":
            base_price = snapshot.last_trade_price
        else:
            base_price = snapshot.mid

        # Fallback if price is zero (degraded snapshot)
        if not base_price or base_price <= 0:
            base_price = snapshot.last_trade_price or snapshot.mid or 0.0

        # Apply slippage
        if policy.slippage_model == "fixed_bps":
            slippage = base_price * policy.slippage_bps / 10_000
        elif policy.slippage_model == "spread_proportional":
            if snapshot.ask and snapshot.bid and snapshot.ask > snapshot.bid:
                slippage = (snapshot.ask - snapshot.bid) * 0.5
            else:
                slippage = base_price * policy.slippage_bps / 10_000
        else:
            slippage = base_price * policy.slippage_bps / 10_000

        # Korean retail: always LONG, slippage always costs more
        if side == "LONG":
            return base_price + slippage
        else:
            return base_price - slippage

    # ------------------------------------------------------------------
    # Main logging entry point
    # ------------------------------------------------------------------

    def log_missed(
        self,
        pair: str,
        side: str,
        signal: str,
        signal_id: str,
        signal_strength: float,
        blocked_by: str,
        block_reason: str = "",
        strategy_params: Optional[dict] = None,
        strategy_type: Optional[str] = None,
        market_regime: str = "",
        exchange_timestamp: Optional[datetime] = None,
        bar_id: Optional[str] = None,
    ) -> MissedOpportunityEvent:
        """
        Call this when a signal fires but is blocked by a gate, filter, or
        risk limit.

        Hook into EACH filter in the strategy's filter chain.  When a filter
        returns False (blocking the trade), call this method.

        All exceptions are caught internally -- this will never propagate
        errors to the caller.
        """
        try:
            now = datetime.now(timezone.utc)
            exch_ts = exchange_timestamp or now

            # Capture market snapshot at signal time
            snapshot = self.snapshot_service.capture_now(pair)
            policy = self._get_policy(strategy_type)

            hyp_entry = self._compute_hypothetical_entry(snapshot, side, policy)

            # Build assumption tags for transparency
            assumption_tags = [
                f"{policy.entry_fill_model}_fill",
                (
                    f"{policy.slippage_bps}bps_slippage"
                    if policy.slippage_model == "fixed_bps"
                    else f"{policy.slippage_model}_slippage"
                ),
            ]
            if policy.fees_included:
                assumption_tags.append(f"{policy.fee_bps}bps_fees")
            else:
                assumption_tags.append("no_fees")
            assumption_tags.append(f"{policy.tp_sl_logic}_tp_sl")

            # Deterministic signal hash for idempotency
            signal_hash = hashlib.sha256(
                f"{pair}|{side}|{signal_id}|{exch_ts.isoformat()}".encode()
            ).hexdigest()[:12]

            metadata = create_event_metadata(
                bot_id=self.bot_id,
                event_type="missed_opportunity",
                payload_key=signal_hash,
                exchange_timestamp=exch_ts,
                data_source_id=self.data_source_id,
                bar_id=bar_id,
            )

            event = MissedOpportunityEvent(
                event_metadata=metadata.to_dict(),
                market_snapshot=snapshot.to_dict(),
                bot_id=self.bot_id,
                pair=pair,
                side=side,
                signal=signal,
                signal_id=signal_id,
                signal_strength=signal_strength,
                signal_time=exch_ts.isoformat(),
                blocked_by=blocked_by,
                block_reason=block_reason,
                hypothetical_entry_price=hyp_entry,
                simulation_policy=policy.to_dict(),
                assumption_tags=assumption_tags,
                strategy_params_at_signal=strategy_params,
                market_regime=market_regime,
                backfill_status="pending",
            )

            self._write_event(event)

            # Queue for later backfill
            with self._backfill_lock:
                self._pending_backfills.append({
                    "event_id": metadata.event_id,
                    "pair": pair,
                    "side": side,
                    "entry_price": hyp_entry,
                    "signal_time": exch_ts,
                    "policy": policy,
                    "snapshot": snapshot,
                    "file_date": now.strftime("%Y-%m-%d"),
                })

            logger.debug(
                f"Missed opportunity logged: {pair} {side} blocked_by={blocked_by}"
            )
            return event

        except Exception as e:
            self._write_error("log_missed", f"{pair}_{signal_id}", e)
            # Return a safe empty event -- never propagate errors
            return MissedOpportunityEvent(event_metadata={}, market_snapshot={})

    # ------------------------------------------------------------------
    # Backfill logic
    # ------------------------------------------------------------------

    def run_backfill(self, data_provider) -> int:
        """
        Process pending backfills.  Call this periodically (e.g. every 5 min)
        or after enough time has passed for outcomes to be known.

        Args:
            data_provider: KIS API client (KoreaInvestAPI) -- used to fetch
                minute bars via ``data_provider.get_minute_bars(symbol, minutes=N)``
                which returns a pandas DataFrame with columns
                [timestamp, open, high, low, close, volume].

        Returns:
            Number of backfills completed in this run.
        """
        now = datetime.now(timezone.utc)
        completed: List[Dict] = []

        with self._backfill_lock:
            pending = list(self._pending_backfills)

        for item in pending:
            try:
                elapsed = now - item["signal_time"]

                # Need at least some elapsed time for meaningful backfill
                if elapsed < timedelta(minutes=5):
                    continue

                # Full backfill requires enough bars have passed
                # (KRX session is 390 minutes, so 24h includes at least one full session)
                is_full = elapsed >= timedelta(hours=24)

                outcomes = self._compute_outcomes(
                    item, data_provider, partial=not is_full, elapsed=elapsed
                )

                if outcomes:
                    status = "complete" if is_full else "partial"
                    self._update_event(
                        item["event_id"], item["file_date"], outcomes, status=status
                    )
                    if is_full:
                        completed.append(item)

            except Exception as e:
                self._write_error(
                    "run_backfill", item.get("event_id", "unknown"), e
                )

        # Remove completed backfills from queue
        if completed:
            with self._backfill_lock:
                for c in completed:
                    try:
                        self._pending_backfills.remove(c)
                    except ValueError:
                        pass

        return len(completed)

    def _compute_outcomes(
        self,
        item: dict,
        data_provider,
        partial: bool,
        elapsed: timedelta,
    ) -> Optional[dict]:
        """
        Compute hypothetical outcomes using historical 1-minute candle data
        from the KIS API.

        The data_provider is expected to be a KoreaInvestAPI instance with
        ``get_minute_bars(ticker, minutes=N)`` returning a pandas DataFrame.
        """
        try:
            pair = item["pair"]
            side = item["side"]
            entry_price = item["entry_price"]
            signal_time = item["signal_time"]
            policy: SimulationPolicy = item["policy"]
            snapshot: MarketSnapshot = item["snapshot"]

            # ---------------------------------------------------------------
            # Fetch candles via KIS API
            # get_minute_bars returns DataFrame[timestamp, open, high, low, close, volume]
            # sorted ascending.  We request enough bars to cover the elapsed time.
            # ---------------------------------------------------------------
            minutes_needed = min(int(elapsed.total_seconds() / 60) + 10, 400)
            candles_df = data_provider.get_minute_bars(pair, minutes=minutes_needed)

            if candles_df is None or candles_df.empty or len(candles_df) < 2:
                return None

            # Filter candles to only those after the signal time
            # (KIS timestamps are KST-aware)
            candles_df = candles_df[candles_df["timestamp"] >= signal_time]
            if candles_df.empty:
                return None

            # ---------------------------------------------------------------
            # Compute TP/SL prices based on policy
            # ---------------------------------------------------------------
            if policy.tp_sl_logic == "atr_based":
                atr = snapshot.atr_14 or (entry_price * 0.01)  # fallback 1%
                # Korean retail: always LONG
                if side == "LONG":
                    tp_price = entry_price + (atr * policy.tp_value)
                    sl_price = entry_price - (atr * policy.sl_value)
                else:
                    tp_price = entry_price - (atr * policy.tp_value)
                    sl_price = entry_price + (atr * policy.sl_value)
            elif policy.tp_sl_logic == "fixed_pct":
                if side == "LONG":
                    tp_price = entry_price * (1 + policy.tp_value / 100)
                    sl_price = entry_price * (1 - policy.sl_value / 100)
                else:
                    tp_price = entry_price * (1 - policy.tp_value / 100)
                    sl_price = entry_price * (1 + policy.sl_value / 100)
            else:
                # Default: ATR-based with fallback
                atr = snapshot.atr_14 or (entry_price * 0.01)
                if side == "LONG":
                    tp_price = entry_price + (atr * 2)
                    sl_price = entry_price - atr
                else:
                    tp_price = entry_price - (atr * 2)
                    sl_price = entry_price + atr

            # ---------------------------------------------------------------
            # Walk through candles, check TP/SL hits and time-based outcomes
            # ---------------------------------------------------------------
            would_have_hit_tp = False
            would_have_hit_sl = False
            bars_to_tp: Optional[int] = None
            bars_to_sl: Optional[int] = None
            first_hit = "TIMEOUT"

            price_1h: Optional[float] = None
            price_4h: Optional[float] = None
            price_24h: Optional[float] = None

            for i, row in enumerate(candles_df.itertuples(index=False)):
                candle_time = row.timestamp
                # Ensure timezone-aware comparison
                if candle_time.tzinfo is None:
                    try:
                        from zoneinfo import ZoneInfo
                    except ImportError:
                        from backports.zoneinfo import ZoneInfo
                    candle_time = candle_time.replace(tzinfo=ZoneInfo("Asia/Seoul"))

                # Convert to UTC for elapsed comparison
                candle_utc = candle_time.astimezone(timezone.utc)
                candle_elapsed = candle_utc - signal_time

                high = float(row.high)
                low = float(row.low)
                close = float(row.close)

                # Record time-based price outcomes
                if candle_elapsed >= timedelta(hours=1) and price_1h is None:
                    price_1h = close
                if candle_elapsed >= timedelta(hours=4) and price_4h is None:
                    price_4h = close
                if candle_elapsed >= timedelta(hours=24) and price_24h is None:
                    price_24h = close

                # Enforce max_hold_bars timeout
                if i >= policy.max_hold_bars:
                    break

                # Check TP/SL (only until first definitive hit)
                if not would_have_hit_tp and not would_have_hit_sl:
                    if side == "LONG":
                        if high >= tp_price:
                            would_have_hit_tp = True
                            bars_to_tp = i
                            if first_hit == "TIMEOUT":
                                first_hit = "TP"
                        if low <= sl_price:
                            would_have_hit_sl = True
                            bars_to_sl = i
                            if first_hit == "TIMEOUT" or (
                                first_hit == "TP" and bars_to_sl <= bars_to_tp
                            ):
                                first_hit = "SL"
                    else:
                        # SHORT path (not used on KRX, but kept for completeness)
                        if low <= tp_price:
                            would_have_hit_tp = True
                            bars_to_tp = i
                            if first_hit == "TIMEOUT":
                                first_hit = "TP"
                        if high >= sl_price:
                            would_have_hit_sl = True
                            bars_to_sl = i
                            if first_hit == "TIMEOUT" or (
                                first_hit == "TP" and bars_to_sl <= bars_to_tp
                            ):
                                first_hit = "SL"

            # Resolve same-bar ambiguity (conservative: assume SL hit first)
            if bars_to_tp is not None and bars_to_sl is not None:
                if bars_to_tp < bars_to_sl:
                    first_hit = "TP"
                elif bars_to_sl < bars_to_tp:
                    first_hit = "SL"
                else:
                    first_hit = "SL"  # conservative assumption

            # If still pending (not enough bars yet)
            if partial and not would_have_hit_tp and not would_have_hit_sl:
                first_hit = "PENDING"

            # ---------------------------------------------------------------
            # Compute hypothetical PnL at each time horizon
            # ---------------------------------------------------------------
            fee_factor = policy.fee_bps / 10_000 if policy.fees_included else 0

            def compute_pnl(exit_price: Optional[float]) -> Optional[float]:
                if exit_price is None or entry_price <= 0:
                    return None
                if side == "LONG":
                    gross = (exit_price - entry_price) / entry_price
                else:
                    gross = (entry_price - exit_price) / entry_price
                # Fees on both entry and exit
                return round((gross - 2 * fee_factor) * 100, 4)

            # ---------------------------------------------------------------
            # Confidence score based on data completeness
            # ---------------------------------------------------------------
            confidence = 0.3  # baseline for having any data
            if price_1h is not None:
                confidence += 0.2
            if price_4h is not None:
                confidence += 0.2
            if price_24h is not None:
                confidence += 0.2
            if would_have_hit_tp or would_have_hit_sl:
                confidence += 0.1

            return {
                "outcome_1h": price_1h,
                "outcome_4h": price_4h,
                "outcome_24h": price_24h,
                "outcome_pnl_1h": compute_pnl(price_1h),
                "outcome_pnl_4h": compute_pnl(price_4h),
                "outcome_pnl_24h": compute_pnl(price_24h),
                "would_have_hit_tp": would_have_hit_tp,
                "would_have_hit_sl": would_have_hit_sl,
                "bars_to_tp": bars_to_tp,
                "bars_to_sl": bars_to_sl,
                "first_hit": first_hit,
                "simulation_confidence": round(confidence, 2),
            }

        except Exception as e:
            self._write_error(
                "compute_outcomes", item.get("event_id", "unknown"), e
            )
            return None

    # ------------------------------------------------------------------
    # JSONL file I/O
    # ------------------------------------------------------------------

    def _update_event(
        self, event_id: str, file_date: str, outcomes: dict, status: str
    ):
        """Update an existing event in the JSONL file with backfill results."""
        filepath = self.data_dir / f"missed_{file_date}.jsonl"
        if not filepath.exists():
            return

        try:
            lines = filepath.read_text(encoding="utf-8").strip().split("\n")
            updated = False
            new_lines: List[str] = []

            for line in lines:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    if event.get("event_metadata", {}).get("event_id") == event_id:
                        event.update(outcomes)
                        event["backfill_status"] = status
                        updated = True
                    new_lines.append(json.dumps(event, default=str))
                except json.JSONDecodeError:
                    new_lines.append(line)  # preserve unparseable lines

            if updated:
                filepath.write_text(
                    "\n".join(new_lines) + "\n", encoding="utf-8"
                )
        except Exception as e:
            self._write_error("update_event", event_id, e)

    def _write_event(self, event: MissedOpportunityEvent):
        """Append a missed opportunity event to today's JSONL file."""
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = self.data_dir / f"missed_{today}.jsonl"
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), default=str) + "\n")
        except Exception as e:
            self._write_error("write_event", "write_failure", e)

    def _write_error(self, method: str, context: str, error: Exception):
        """Write an error record to the errors JSONL file. Never raises."""
        try:
            error_dir = self.data_dir.parent / "errors"
            error_dir.mkdir(parents=True, exist_ok=True)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = error_dir / f"instrumentation_errors_{today}.jsonl"
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "component": "missed_opportunity",
                "method": method,
                "context": str(context),
                "error": str(error),
            }
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            # Last resort -- log to stderr via loguru, but never propagate
            try:
                logger.error(
                    f"MissedOpportunityLogger._write_error failed: "
                    f"method={method} context={context} error={error}"
                )
            except Exception:
                pass

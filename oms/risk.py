"""
Risk Gateway: Pre-trade risk checks.

Check order: Global -> Daily -> Exposure -> Strategy Budget -> Microstructure
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
import time
from loguru import logger

from .intent import Intent, IntentType
from .state import StateStore
from kis_core import SectorExposure, SectorExposureConfig


class RiskDecision(Enum):
    APPROVE = auto()
    MODIFY = auto()
    REJECT = auto()
    DEFER = auto()


@dataclass
class RiskResult:
    """Result of risk check."""
    decision: RiskDecision
    reason: str = ""
    modified_qty: Optional[int] = None
    cooldown_sec: Optional[float] = None
    blocking_positions: Optional[List[Dict[str, Any]]] = None
    resource_conflict_type: Optional[str] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RiskConfig:
    """Risk limits configuration."""
    # Daily circuit breakers
    daily_loss_warn_pct: float = 0.02
    daily_loss_halt_pct: float = 0.03

    # Exposure limits
    max_gross_exposure_pct: float = 0.80
    max_net_exposure_pct: float = 0.60
    max_position_pct: float = 0.15
    max_positions_count: int = 10
    max_sector_pct: float = 0.30
    unknown_sector_policy: str = "allow"

    # Strategy budgets (% of equity for risk)
    strategy_budgets: dict = None

    # Microstructure
    max_spread_bps: float = 50.0
    vi_cooldown_sec: float = 600.0

    # Regime-based exposure caps (set by PCIM at 08:30)
    regime_exposure_caps: Dict[str, float] = None
    current_regime: str = "NORMAL"

    # Durable protective-stop policy. These are enforced by OMS core in
    # paper/live when durable persistence is required.
    require_durable_stops: bool = True
    default_stop_protection_mode: str = "oms_watcher"
    allow_synthetic_stop_only: bool = False
    stop_price_stale_after_sec: float = 30.0
    stop_watcher_interval_sec: float = 5.0
    stop_exit_order_type: str = "MARKET"
    stop_protection_emergency_override: bool = False

    def __post_init__(self):
        # Set default regime exposure caps if not provided
        if self.regime_exposure_caps is None:
            self.regime_exposure_caps = {
                "CRISIS": 0.20, "WEAK": 0.50, "NORMAL": 0.80, "STRONG": 1.00,
            }

        # Set default strategy budgets if not provided
        if self.strategy_budgets is None:
            self.strategy_budgets = {
                "PCIM": {"max_positions": 8, "max_risk_pct": 0.10, "capital_allocation_pct": 1.0},
            }
        policy = str(self.unknown_sector_policy or "allow").lower().strip()
        self.unknown_sector_policy = "block" if policy in {"block", "reject"} else "allow"


class RiskGateway:
    """
    Pre-trade risk gateway.

    Checks intents against risk limits before approval.
    """

    def __init__(
        self,
        state: StateStore,
        config: RiskConfig,
        price_getter: Optional[Callable[[str], float]] = None,
        sector_map: Optional[Dict[str, str]] = None,
    ):
        self.state = state
        self.config = config
        self._price_getter = price_getter

        # Control flags
        self.safe_mode: bool = False
        self.halt_new_entries: bool = False
        self.flatten_in_progress: bool = False
        self._paused_strategies: set = set()

        # Sector exposure tracking
        sector_config = SectorExposureConfig(
            mode="pct",
            max_sector_pct=config.max_sector_pct,
            unknown_sector_policy=config.unknown_sector_policy,
        )
        self._sector_exposure = SectorExposure(sector_map or {}, sector_config)
        self.last_trace: List[Dict[str, Any]] = []

    def _get_price(self, symbol: str, fallback: Optional[float] = None) -> Optional[float]:
        """Get live price via injected getter, with fallback."""
        if self._price_getter:
            try:
                px = self._price_getter(symbol)
                if px and px > 0:
                    return px
            except Exception:
                pass
        return fallback

    def _build_blocking_positions(
        self,
        positions: Dict,
        equity: float,
        filter_fn: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Build list of positions contributing to a portfolio-level rejection.

        Args:
            positions: Dict of symbol -> PositionState from state store.
            equity: Current account equity for exposure_pct calculation.
            filter_fn: Optional filter; if given, only include positions where
                       filter_fn(symbol, position) returns True.

        Returns:
            List of dicts sorted by exposure_pct descending, each with keys:
            strategy, symbol, qty, exposure_pct, side.
        """
        result = []
        eq = max(equity, 1.0)
        for symbol, pos in positions.items():
            total_qty = pos.real_qty + pos.working_qty(side="BUY")
            if total_qty <= 0:
                continue
            if filter_fn and not filter_fn(symbol, pos):
                continue
            px = pos.avg_price or self._get_price(symbol) or 0.0
            notional = total_qty * px
            exposure_pct = round(notional / eq, 4)
            # Decompose by strategy allocation
            if pos.allocations:
                for strat_id, alloc in pos.allocations.items():
                    alloc_qty = alloc.qty
                    if alloc_qty <= 0:
                        continue
                    alloc_notional = alloc_qty * px
                    result.append({
                        "strategy": strat_id,
                        "symbol": symbol,
                        "qty": alloc_qty,
                        "exposure_pct": round(alloc_notional / eq, 4),
                        "side": "LONG",
                    })
                # Include working-order qty not yet allocated
                allocated_total = sum(a.qty for a in pos.allocations.values())
                unallocated = total_qty - allocated_total
                if unallocated > 0:
                    un_notional = unallocated * px
                    result.append({
                        "strategy": "_PENDING_",
                        "symbol": symbol,
                        "qty": unallocated,
                        "exposure_pct": round(un_notional / eq, 4),
                        "side": "LONG",
                    })
            else:
                result.append({
                    "strategy": "_UNKNOWN_",
                    "symbol": symbol,
                    "qty": total_qty,
                    "exposure_pct": exposure_pct,
                    "side": "LONG",
                })
        result.sort(key=lambda x: x["exposure_pct"], reverse=True)
        return result

    def check(self, intent: Intent) -> RiskResult:
        """
        Run all risk checks on intent.

        Returns RiskResult with decision and any modifications.
        """
        trace: List[Dict[str, Any]] = []

        # Block entries if equity not yet loaded
        if intent.intent_type == IntentType.ENTER and self.state.equity <= 0:
            result = RiskResult(
                RiskDecision.DEFER,
                "Equity not yet loaded — reconciliation pending"
            )
            trace.append(
                self._trace_row(
                    "equity_loaded",
                    intent,
                    result,
                    thresholds={"minimum_equity": 0.0},
                    observed={"equity": self.state.equity},
                )
            )
            return self._with_trace(result, trace)

        # 1. Global hard blocks
        result = self._check_global_blocks(intent)
        trace.append(self._trace_row("global_blocks", intent, result))
        if result.decision != RiskDecision.APPROVE:
            return self._with_trace(result, trace)

        # 2. Daily circuit breakers
        result = self._check_daily_limits(intent)
        trace.append(
            self._trace_row(
                "daily_limits",
                intent,
                result,
                thresholds={
                    "daily_loss_warn_pct": self.config.daily_loss_warn_pct,
                    "daily_loss_halt_pct": self.config.daily_loss_halt_pct,
                },
                observed={"daily_pnl_pct": self.state.daily_pnl_pct},
            )
        )
        if result.decision != RiskDecision.APPROVE:
            return self._with_trace(result, trace)

        # 3. Exposure limits (only for entries)
        modified_qty = None
        if intent.intent_type == IntentType.ENTER:
            result = self._check_exposure_limits(intent)
            trace.append(
                self._trace_row(
                    "exposure_limits",
                    intent,
                    result,
                    thresholds={
                        "max_gross_exposure_pct": self.config.max_gross_exposure_pct,
                        "max_position_pct": self.config.max_position_pct,
                        "max_positions_count": self.config.max_positions_count,
                    },
                    observed={"equity": self.state.equity},
                )
            )
            if result.decision == RiskDecision.MODIFY:
                # Apply modification but continue checking with modified qty
                modified_qty = result.modified_qty
                intent.desired_qty = modified_qty
            elif result.decision != RiskDecision.APPROVE:
                return self._with_trace(result, trace)

        # 3b. Sector limits (only for entries)
        if intent.intent_type == IntentType.ENTER:
            result = self._check_sector_limits(intent)
            trace.append(
                self._trace_row(
                    "sector_limits",
                    intent,
                    result,
                    thresholds={"max_sector_pct": self.config.max_sector_pct},
                )
            )
            if result.decision != RiskDecision.APPROVE:
                return self._with_trace(result, trace)

        # 4. Strategy budget (only for entries)
        if intent.intent_type == IntentType.ENTER:
            result = self._check_strategy_budget(intent)
            trace.append(
                self._trace_row(
                    "strategy_budget",
                    intent,
                    result,
                    thresholds=dict(self.config.strategy_budgets.get(intent.strategy_id) or {}),
                    observed={"strategy_id": intent.strategy_id},
                )
            )
            if result.decision == RiskDecision.MODIFY:
                # Take the more restrictive of the two modifications
                if modified_qty is None or result.modified_qty < modified_qty:
                    modified_qty = result.modified_qty
                    intent.desired_qty = modified_qty
            elif result.decision != RiskDecision.APPROVE:
                return self._with_trace(result, trace)

        # 5. Microstructure gates
        result = self._check_microstructure(intent)
        trace.append(
            self._trace_row(
                "microstructure",
                intent,
                result,
                thresholds={
                    "vi_cooldown_sec": self.config.vi_cooldown_sec,
                    "max_spread_bps": self.config.max_spread_bps,
                },
            )
        )
        if result.decision != RiskDecision.APPROVE:
            return self._with_trace(result, trace)

        # Return MODIFY if qty was scaled down by any check
        if modified_qty is not None:
            return self._with_trace(
                RiskResult(
                    decision=RiskDecision.MODIFY,
                    reason=f"Qty scaled to {modified_qty}",
                    modified_qty=modified_qty,
                ),
                trace,
            )

        return self._with_trace(RiskResult(decision=RiskDecision.APPROVE), trace)

    def _with_trace(self, result: RiskResult, trace: List[Dict[str, Any]]) -> RiskResult:
        result.trace = list(trace)
        self.last_trace = list(trace)
        return result

    def _trace_row(
        self,
        rule: str,
        intent: Intent,
        result: RiskResult,
        *,
        thresholds: Optional[Dict[str, Any]] = None,
        observed: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "rule": rule,
            "strategy_id": intent.strategy_id,
            "symbol": intent.symbol,
            "intent_type": intent.intent_type.name,
            "decision": result.decision.name,
            "reason": result.reason,
            "modified_qty": result.modified_qty,
            "thresholds": thresholds or {},
            "observed": observed or {},
            "blocking_positions": result.blocking_positions,
            "resource_conflict_type": result.resource_conflict_type,
        }

    def _check_global_blocks(self, intent: Intent) -> RiskResult:
        """Check system-level blocks."""
        if self.safe_mode:
            return RiskResult(RiskDecision.DEFER, "OMS in safe mode")

        if self.flatten_in_progress and intent.intent_type == IntentType.ENTER:
            return RiskResult(RiskDecision.REJECT, "Flatten in progress")

        # Halt new entries flag (set when daily loss exceeds warn threshold)
        if self.halt_new_entries and intent.intent_type == IntentType.ENTER:
            return RiskResult(RiskDecision.REJECT, "New entries halted (daily loss)")

        # Paused strategy blocks entries
        if intent.intent_type == IntentType.ENTER and intent.strategy_id in self._paused_strategies:
            return RiskResult(RiskDecision.REJECT, f"Strategy {intent.strategy_id} is paused")

        # Frozen symbol blocks entries
        if intent.intent_type == IntentType.ENTER:
            pos = self.state.get_position(intent.symbol)
            if pos.frozen:
                return RiskResult(RiskDecision.REJECT, "Symbol frozen: allocation drift unresolved")

        return RiskResult(RiskDecision.APPROVE)

    def _check_daily_limits(self, intent: Intent) -> RiskResult:
        """Check daily PnL circuit breakers."""
        pnl_pct = self.state.daily_pnl_pct

        if pnl_pct <= -self.config.daily_loss_halt_pct:
            if intent.intent_type == IntentType.ENTER:
                return RiskResult(RiskDecision.REJECT, f"Daily loss {pnl_pct:.1%} exceeds halt limit")

        if pnl_pct <= -self.config.daily_loss_warn_pct:
            if intent.intent_type == IntentType.ENTER:
                self.halt_new_entries = True
                return RiskResult(RiskDecision.REJECT, f"Daily loss {pnl_pct:.1%} exceeds warn limit")

        return RiskResult(RiskDecision.APPROVE)

    def _check_exposure_limits(self, intent: Intent) -> RiskResult:
        """Check portfolio exposure limits (gross, net, per-symbol)."""
        equity = max(self.state.equity, 1.0)
        positions = self.state.get_all_positions()

        # Count active positions (real + committed via working orders)
        active_count = sum(
            1 for p in positions.values()
            if p.real_qty > 0 or p.working_qty(side="BUY") > 0
        )
        if active_count >= self.config.max_positions_count:
            return RiskResult(
                RiskDecision.REJECT,
                f"Max positions ({self.config.max_positions_count}) reached",
                blocking_positions=self._build_blocking_positions(positions, equity),
                resource_conflict_type="max_positions",
            )

        # Gross exposure: existing + committed + new
        gross = 0.0
        for p in positions.values():
            px = p.avg_price or self._get_price(p.symbol) or 0.0
            gross += p.real_qty * px
            gross += p.working_qty(side="BUY") * px
        entry_px = intent.risk_payload.entry_px or self._get_price(intent.symbol)
        if not entry_px:
            return RiskResult(RiskDecision.DEFER, "Price unavailable for risk check")
        qty = intent.desired_qty or intent.target_qty or 0
        new_notional = entry_px * qty

        total_exposure_pct = (gross + new_notional) / equity

        if total_exposure_pct > self.config.max_gross_exposure_pct:
            return RiskResult(
                RiskDecision.REJECT,
                f"Gross exposure would exceed {self.config.max_gross_exposure_pct:.0%}",
                blocking_positions=self._build_blocking_positions(positions, equity),
                resource_conflict_type="gross_exposure",
            )

        # Regime cap (tighter than static limit in CRISIS/WEAK)
        regime_cap = self.config.regime_exposure_caps.get(self.config.current_regime, 1.0)
        if total_exposure_pct > regime_cap:
            return RiskResult(
                RiskDecision.REJECT,
                f"Regime {self.config.current_regime} cap {regime_cap:.0%} exceeded",
                blocking_positions=self._build_blocking_positions(positions, equity),
                resource_conflict_type="regime_cap",
            )

        # Per-symbol limit (existing + committed + new)
        existing_pos = self.state.get_position(intent.symbol)
        existing_px = existing_pos.avg_price or entry_px
        existing_notional = (
            existing_pos.real_qty + existing_pos.working_qty(side="BUY")
        ) * existing_px
        total_position_notional = existing_notional + new_notional
        position_pct = total_position_notional / equity
        if position_pct > self.config.max_position_pct:
            max_total = equity * self.config.max_position_pct
            max_new = max_total - existing_notional
            max_qty = int(max_new / max(entry_px, 1))
            if max_qty <= 0:
                return RiskResult(
                    RiskDecision.REJECT,
                    f"Position too large ({position_pct:.1%})",
                    blocking_positions=self._build_blocking_positions(
                        positions, equity,
                        filter_fn=lambda sym, _pos: sym == intent.symbol,
                    ),
                    resource_conflict_type="per_symbol",
                )
            return RiskResult(
                RiskDecision.MODIFY,
                f"Scaled from {qty} to {max_qty} for position limit",
                modified_qty=max_qty
            )

        return RiskResult(RiskDecision.APPROVE)

    def _check_sector_limits(self, intent: Intent) -> RiskResult:
        """Check sector exposure limits (max_sector_pct from config)."""
        equity = max(self.state.equity, 1.0)
        entry_px = intent.risk_payload.entry_px or self._get_price(intent.symbol)
        qty = intent.desired_qty or intent.target_qty or 0

        if not entry_px or entry_px <= 0 or qty <= 0:
            return RiskResult(RiskDecision.APPROVE)

        sector = self._sector_exposure.get_sector(intent.symbol)
        if sector == "UNKNOWN" and self.config.unknown_sector_policy == "block":
            return RiskResult(
                RiskDecision.REJECT,
                f"Unknown sector for {intent.symbol}; approved sector map required",
                resource_conflict_type="unknown_sector",
            )

        if not self._sector_exposure.can_enter(intent.symbol, qty, entry_px, equity):
            current_pct = self._sector_exposure.sector_pct(sector, equity)
            all_positions = self.state.get_all_positions()
            return RiskResult(
                RiskDecision.REJECT,
                f"Sector {sector} exposure {current_pct:.1%} would exceed {self.config.max_sector_pct:.0%}",
                blocking_positions=self._build_blocking_positions(
                    all_positions, equity,
                    filter_fn=lambda sym, _pos: self._sector_exposure.get_sector(sym) == sector,
                ),
                resource_conflict_type="sector_cap",
            )

        return RiskResult(RiskDecision.APPROVE)

    def _check_strategy_budget(self, intent: Intent) -> RiskResult:
        """Check strategy-specific position count and risk budget."""
        budget = self.config.strategy_budgets.get(intent.strategy_id)
        if not budget:
            return RiskResult(RiskDecision.APPROVE)

        # Count strategy positions (allocated + committed via working orders)
        positions = self.state.get_all_positions()
        strategy_positions = sum(
            1 for p in positions.values()
            if p.get_allocation(intent.strategy_id) > 0
            or p.working_qty(strategy_id=intent.strategy_id, side="BUY") > 0
        )
        if strategy_positions >= budget["max_positions"]:
            return RiskResult(
                RiskDecision.REJECT,
                f"{intent.strategy_id} max positions ({budget['max_positions']}) reached",
                blocking_positions=self._build_blocking_positions(
                    positions, max(self.state.equity, 1.0),
                    filter_fn=lambda sym, p: p.get_allocation(intent.strategy_id) > 0,
                ),
                resource_conflict_type="strategy_budget_positions",
            )

        # Risk-by-stop: incremental risk = qty * (entry - stop)
        max_risk_pct = budget.get("max_risk_pct")
        stop_px = intent.risk_payload.stop_px
        entry_px = intent.risk_payload.entry_px
        if max_risk_pct and stop_px and entry_px:
            qty = intent.desired_qty or intent.target_qty or 0
            risk_per_share = max(entry_px - stop_px, 0.0)
            trade_risk = qty * risk_per_share
            max_risk_krw = max_risk_pct * max(self.state.equity, 1.0)
            if trade_risk > max_risk_krw:
                scaled_qty = int(max_risk_krw / max(risk_per_share, 1.0))
                if scaled_qty <= 0:
                    return RiskResult(
                        RiskDecision.REJECT,
                        f"{intent.strategy_id} risk budget exceeded",
                        blocking_positions=self._build_blocking_positions(
                            positions, max(self.state.equity, 1.0),
                            filter_fn=lambda sym, p: p.get_allocation(intent.strategy_id) > 0,
                        ),
                        resource_conflict_type="strategy_budget_risk",
                    )
                return RiskResult(
                    RiskDecision.MODIFY,
                    f"Scaled from {qty} to {scaled_qty} for risk budget",
                    modified_qty=scaled_qty
                )

        return RiskResult(RiskDecision.APPROVE)

    def _check_microstructure(self, intent: Intent) -> RiskResult:
        """Check microstructure conditions."""
        pos = self.state.get_position(intent.symbol)
        now = time.time()

        if pos.vi_cooldown_until and now < pos.vi_cooldown_until:
            remaining = pos.vi_cooldown_until - now
            return RiskResult(RiskDecision.DEFER, f"VI cooldown ({remaining:.0f}s remaining)")

        return RiskResult(RiskDecision.APPROVE)

    def set_regime(self, regime: str) -> None:
        """Update current market regime (called by PCIM at 08:30)."""
        self.config.current_regime = regime
        cap = self.config.regime_exposure_caps.get(regime, 1.0)
        logger.info(f"Regime set to {regime}: max_exposure={cap:.0%}")

    def set_safe_mode(self, enabled: bool) -> None:
        """Enable/disable safe mode."""
        self.safe_mode = enabled

    def trigger_flatten(self) -> None:
        """Trigger emergency flatten."""
        self.flatten_in_progress = True
        self.halt_new_entries = True

    def set_vi_cooldown(self, symbol: str, duration_sec: Optional[float] = None) -> None:
        """Set VI cooldown on a symbol (called when VI trigger detected)."""
        dur = duration_sec or self.config.vi_cooldown_sec
        self.state.update_position(symbol, vi_cooldown_until=time.time() + dur)

    # --- Sector exposure lifecycle methods ---

    def reserve_sector(self, symbol: str, qty: int, price: float) -> None:
        """Reserve sector slot before order submission."""
        self._sector_exposure.reserve(symbol, qty, price)

    def unreserve_sector(self, symbol: str, qty: int, price: float) -> None:
        """Release sector reservation on order failure/cancel."""
        self._sector_exposure.unreserve(symbol, qty, price)

    def on_sector_fill(self, symbol: str, qty: int, price: float) -> None:
        """Update sector tracking on fill."""
        self._sector_exposure.on_fill(symbol, qty, price)

    def on_sector_close(self, symbol: str, qty: int, price: float) -> None:
        """Update sector tracking on position close."""
        self._sector_exposure.on_close(symbol, qty, price)

    def reconcile_sector_exposure(
        self,
        positions: Dict[str, tuple],
        working_orders: Optional[List[tuple[str, int, float]]] = None,
    ) -> None:
        """Rebuild sector exposure from OMS truth."""
        self._sector_exposure.reconcile(positions, working_orders)

    def update_sector_map(self, sector_map: Dict[str, str]) -> None:
        """Update symbol-to-sector mapping."""
        self._sector_exposure.sym_to_sector = sector_map

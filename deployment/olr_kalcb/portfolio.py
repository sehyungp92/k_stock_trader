from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable

from .hashing import canonical_json_hash

SUPPORTED_SIDES = {"BUY", "SELL"}


@dataclass(frozen=True, slots=True)
class PortfolioPolicyConfig:
    max_gross_notional: float = 10_000_000.0
    max_symbol_notional: float = 2_000_000.0
    max_sector_notional: float = 4_000_000.0
    strategy_priority: tuple[str, ...] = ("KALCB", "OLR")


@dataclass(frozen=True, slots=True)
class PortfolioArbitrationInput:
    action_ref: str
    strategy_id: str
    symbol: str
    side: str
    intended_qty: int
    intended_notional: float
    timestamp: datetime
    sector: str = "UNKNOWN"
    candidate_rank: int = 0
    candidate_score_band: str = ""
    route_family: str = ""
    current_strategy_exposure: float = 0.0
    current_portfolio_exposure: float = 0.0
    current_symbol_exposure: float = 0.0
    current_sector_exposure: float = 0.0
    current_strategy_symbol_qty: int = 0
    current_strategy_symbol_notional: float = 0.0
    current_symbol_qty: int = 0
    current_symbol_notional: float = 0.0
    cash: float = 0.0
    equity: float = 0.0
    source_artifact_hashes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.intended_qty < 0:
            raise ValueError("intended_qty cannot be negative")
        object.__setattr__(self, "strategy_id", self.strategy_id.upper().strip())
        object.__setattr__(self, "symbol", str(self.symbol).zfill(6))
        side = self.side.upper().strip()
        if side not in SUPPORTED_SIDES:
            raise ValueError(f"unsupported side {self.side!r}; expected one of {sorted(SUPPORTED_SIDES)}")
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "sector", str(self.sector or "UNKNOWN").upper().strip() or "UNKNOWN")
        object.__setattr__(self, "source_artifact_hashes", tuple(self.source_artifact_hashes))
        if self.current_strategy_exposure <= 0.0 and self.current_strategy_symbol_notional > 0.0:
            object.__setattr__(self, "current_strategy_exposure", float(self.current_strategy_symbol_notional))
        if self.current_symbol_exposure <= 0.0 and self.current_symbol_notional > 0.0:
            object.__setattr__(self, "current_symbol_exposure", float(self.current_symbol_notional))


@dataclass(frozen=True, slots=True)
class PortfolioArbitrationDecision:
    action_ref: str
    strategy_id: str
    symbol: str
    decision: str
    final_qty: int
    final_notional: float
    reason_code: str
    policy_hash: str
    source_artifact_hashes: tuple[str, ...]
    timestamp: datetime

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        payload["source_artifact_hashes"] = list(self.source_artifact_hashes)
        return payload


class PortfolioArbitrationPolicy:
    """Conservative live-safe admission policy over already-formed actions."""

    def __init__(self, config: PortfolioPolicyConfig | None = None):
        self.config = config or PortfolioPolicyConfig()
        self.policy_hash = canonical_json_hash(asdict(self.config))

    def decide_many(self, inputs: Iterable[PortfolioArbitrationInput]) -> list[PortfolioArbitrationDecision]:
        admitted_gross = 0.0
        admitted_buy_symbol: dict[str, float] = {}
        admitted_exit_strategy_symbol: dict[tuple[str, str], float] = {}
        admitted_exit_qty: dict[tuple[str, str], int] = {}
        admitted_sector: dict[str, float] = {}
        decisions: list[PortfolioArbitrationDecision] = []
        for item in sorted(inputs, key=self._sort_key):
            exit_key = (item.strategy_id, item.symbol)
            decision = self.decide_one(
                item,
                admitted_gross=admitted_gross,
                admitted_symbol=admitted_buy_symbol.get(item.symbol, 0.0),
                admitted_sector=admitted_sector.get(item.sector, 0.0),
                admitted_exit_notional=admitted_exit_strategy_symbol.get(exit_key, 0.0),
                admitted_exit_qty=admitted_exit_qty.get(exit_key, 0),
            )
            decisions.append(decision)
            if decision.decision not in {"accepted", "resized"} or decision.final_notional <= 0:
                continue
            if item.side == "BUY":
                admitted_gross += decision.final_notional
                admitted_buy_symbol[item.symbol] = admitted_buy_symbol.get(item.symbol, 0.0) + decision.final_notional
                admitted_sector[item.sector] = admitted_sector.get(item.sector, 0.0) + decision.final_notional
            elif item.side == "SELL":
                admitted_exit_strategy_symbol[exit_key] = admitted_exit_strategy_symbol.get(exit_key, 0.0) + decision.final_notional
                admitted_exit_qty[exit_key] = admitted_exit_qty.get(exit_key, 0) + int(decision.final_qty)
        return decisions

    def decide_one(
        self,
        item: PortfolioArbitrationInput,
        *,
        admitted_gross: float = 0.0,
        admitted_symbol: float = 0.0,
        admitted_sector: float = 0.0,
        admitted_exit_notional: float = 0.0,
        admitted_exit_qty: int = 0,
    ) -> PortfolioArbitrationDecision:
        if item.intended_qty == 0 or item.intended_notional <= 0:
            return self._decision(item, "blocked", 0, 0.0, "zero_quantity_or_notional")
        if item.side == "SELL":
            return self._sell_decision(item, admitted_exit_notional=admitted_exit_notional, admitted_exit_qty=admitted_exit_qty)
        if item.cash <= 0.0 or item.equity <= 0.0:
            return self._decision(item, "blocked", 0, 0.0, "missing_or_zero_account_state")
        if item.current_symbol_exposure + admitted_symbol > 0:
            return self._decision(item, "blocked", 0, 0.0, "duplicate_symbol_conflict")
        cash_capacity = item.cash - admitted_gross
        capacity = min(
            self.config.max_gross_notional - item.current_portfolio_exposure - admitted_gross,
            self.config.max_symbol_notional - item.current_symbol_exposure - admitted_symbol,
            self.config.max_sector_notional - item.current_sector_exposure - admitted_sector,
            cash_capacity,
        )
        if capacity <= 0:
            return self._decision(item, "blocked", 0, 0.0, "capital_or_exposure_limit")
        if item.intended_notional <= capacity:
            return self._decision(item, "accepted", item.intended_qty, item.intended_notional, "accepted")
        resized_qty = int(item.intended_qty * (capacity / item.intended_notional))
        if resized_qty <= 0:
            return self._decision(item, "blocked", 0, 0.0, "capacity_below_min_quantity")
        final_notional = item.intended_notional * (resized_qty / item.intended_qty)
        return self._decision(item, "resized", resized_qty, final_notional, "resized_to_capacity")

    def _sell_decision(
        self,
        item: PortfolioArbitrationInput,
        *,
        admitted_exit_notional: float = 0.0,
        admitted_exit_qty: int = 0,
    ) -> PortfolioArbitrationDecision:
        owned_qty = min(max(int(item.current_strategy_symbol_qty or 0), 0), max(int(item.current_symbol_qty or 0), 0))
        if owned_qty > 0:
            already_admitted_qty = max(int(admitted_exit_qty or 0), 0)
            if already_admitted_qty <= 0 and admitted_exit_notional > 0.0 and item.intended_qty > 0 and item.intended_notional > 0.0:
                already_admitted_qty = int(item.intended_qty * min(max(float(admitted_exit_notional), 0.0) / item.intended_notional, 1.0))
            reducible_qty = max(owned_qty - already_admitted_qty, 0)
            if reducible_qty <= 0:
                return self._decision(item, "blocked", 0, 0.0, "exit_capacity_below_min_quantity")
            if item.intended_qty > reducible_qty:
                final_notional = _notional_for_qty(item, reducible_qty)
                return self._decision(item, "resized", reducible_qty, final_notional, "resized_to_existing_exposure")
            return self._decision(item, "accepted", item.intended_qty, item.intended_notional, "accepted_exit_reduces_exposure")
        owned_notional = min(max(float(item.current_strategy_exposure), 0.0), max(float(item.current_symbol_exposure), 0.0))
        reducible_notional = max(owned_notional - max(float(admitted_exit_notional), 0.0), 0.0)
        if reducible_notional <= 0:
            return self._decision(item, "blocked", 0, 0.0, "unsupported_short_or_unmatched_exit")
        if item.intended_notional <= reducible_notional:
            return self._decision(item, "accepted", item.intended_qty, item.intended_notional, "accepted_exit_reduces_exposure")
        resized_qty = int(item.intended_qty * (reducible_notional / item.intended_notional))
        if resized_qty <= 0:
            return self._decision(item, "blocked", 0, 0.0, "exit_capacity_below_min_quantity")
        final_notional = _notional_for_qty(item, resized_qty)
        return self._decision(item, "resized", resized_qty, final_notional, "resized_to_existing_exposure")

    def metrics(self, decisions: Iterable[PortfolioArbitrationDecision]) -> dict[str, Any]:
        rows = list(decisions)
        reason_counts: dict[str, int] = {}
        notional_by_strategy: dict[str, float] = {}
        for row in rows:
            reason_counts[row.reason_code] = reason_counts.get(row.reason_code, 0) + 1
            if row.decision in {"accepted", "resized"}:
                notional_by_strategy[row.strategy_id] = notional_by_strategy.get(row.strategy_id, 0.0) + row.final_notional
        return {
            "accepted_count": sum(1 for row in rows if row.decision == "accepted"),
            "blocked_count": sum(1 for row in rows if row.decision == "blocked"),
            "resized_count": sum(1 for row in rows if row.decision == "resized"),
            "deferred_count": sum(1 for row in rows if row.decision == "deferred"),
            "reason_counts": reason_counts,
            "notional_by_strategy": notional_by_strategy,
            "gross_admitted_notional": sum(row.final_notional for row in rows if row.decision in {"accepted", "resized"}),
        }

    def _sort_key(self, item: PortfolioArbitrationInput) -> tuple[Any, ...]:
        try:
            priority = self.config.strategy_priority.index(item.strategy_id)
        except ValueError:
            priority = len(self.config.strategy_priority)
        return (item.timestamp, priority, item.symbol, item.action_ref)

    def _decision(self, item: PortfolioArbitrationInput, decision: str, qty: int, notional: float, reason: str) -> PortfolioArbitrationDecision:
        return PortfolioArbitrationDecision(
            action_ref=item.action_ref,
            strategy_id=item.strategy_id,
            symbol=item.symbol,
            decision=decision,
            final_qty=qty,
            final_notional=notional,
            reason_code=reason,
            policy_hash=self.policy_hash,
            source_artifact_hashes=item.source_artifact_hashes,
            timestamp=item.timestamp,
        )


def _notional_for_qty(item: PortfolioArbitrationInput, qty: int) -> float:
    if item.intended_qty <= 0:
        return 0.0
    return float(item.intended_notional) * (max(int(qty), 0) / item.intended_qty)

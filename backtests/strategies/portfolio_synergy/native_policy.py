"""Causal exposure caps for native replay; no completed-trade return inputs."""
from dataclasses import asdict, fields
import math

from deployment.olr_kalcb.portfolio import PortfolioPolicyConfig, PortfolioArbitrationPolicy, PortfolioArbitrationInput


def native_policy_config(mutations):
    allowed = {"portfolio." + f.name for f in fields(PortfolioPolicyConfig)}
    unknown = set(mutations) - allowed
    if unknown:
        raise ValueError(f"Unsupported native portfolio mutations: {sorted(unknown)}")
    values = {key.removeprefix("portfolio."): value for key, value in mutations.items()}
    if "strategy_priority" in values:
        priority = tuple(values["strategy_priority"])
        if set(priority) != {"KALCB", "OLR"} or len(priority) != 2:
            raise ValueError("Native strategy priority must contain KALCB and OLR exactly once")
        values["strategy_priority"] = priority
    return PortfolioPolicyConfig(**values)


class NativePolicyMixin:
    def configure_portfolio_policy(self, policy=None, sectors=None):
        if policy is not None and not isinstance(policy, PortfolioPolicyConfig):
            raise TypeError("Native replay requires a validated PortfolioPolicyConfig")
        self.portfolio_policy = PortfolioArbitrationPolicy(policy) if policy is not None else None
        self.policy_sectors = {symbol: str(sector or "UNKNOWN").strip().upper() for symbol, sector in (sectors or {}).items()}

    def _phase_order_key(self, order):
        key = super()._phase_order_key(order)
        policy = getattr(self, "portfolio_policy", None)
        if policy is None:
            return key
        priority = policy.config.strategy_priority
        rank = priority.index(order.strategy_id) if order.strategy_id in priority else len(priority)
        candidate_rank = (int(order.metadata.get("candidate_rank", 0) or 0)
            if getattr(self, "decision_rank", False) and order.strategy_id == "OLR" else 0) or 1_000_000
        model = getattr(self, "quality_model", None)
        if model is not None and order.side == "BUY":
            score = model.score(order.strategy_id, order.metadata, order.submitted_at)
            if score is not None:
                return (key[0], key[1], -score, rank, candidate_rank, *key[3:])
            return (key[0], key[1], 0.0, rank, candidate_rank, *key[3:])
        return (key[0], key[1], rank, candidate_rank, *key[3:])

    def _policy_execution_quantity(self, order, price, quantity, timestamp):
        policy = getattr(self, "portfolio_policy", None)
        if policy is None or order.side != "BUY":
            return quantity
        sector = self.policy_sectors.get(order.symbol, "UNKNOWN")
        exposure = {"gross": 0.0, "symbol": 0.0, "sector": 0.0}
        for (_, symbol), position in self.positions.items():
            notional = position.qty * self.last_prices.get(symbol, position.avg_price)
            exposure["gross"] += notional
            if symbol == order.symbol:
                exposure["symbol"] += notional
            if self.policy_sectors.get(symbol, "UNKNOWN") == sector:
                exposure["sector"] += notional
        equity = self.cash + exposure["gross"]
        unit_cost = price * (1 + self.strategy_costs[order.strategy_id].commission_bps / 10000)
        continuation = self.order_filled.get(order.order_id, 0) > 0
        funding = (self.strategy_buying_power(order.strategy_id)
                   if policy.config.funding_mode == "native_buying_power" else self.cash)
        if continuation:
            caps = policy.effective_notional_caps(equity)
            available = min(funding, caps["max_gross_notional"] - exposure["gross"],
                caps["max_symbol_notional"] - exposure["symbol"], caps["max_sector_notional"] - exposure["sector"])
            final = min(quantity, max(0, math.floor((available + 1e-9) / unit_cost)))
            reason = "remaining_original_order_capacity"
        else:
            item = PortfolioArbitrationInput(order.order_id, order.strategy_id, order.symbol, "BUY", quantity,
                quantity * unit_cost, timestamp, sector=sector, current_portfolio_exposure=exposure["gross"],
                current_symbol_exposure=exposure["symbol"], current_sector_exposure=exposure["sector"], cash=self.cash, equity=equity,
                available_buying_power=funding)
            decision = policy.decide_one(item)
            final, reason = decision.final_qty, decision.reason_code
        self.journal.append({"event": "portfolio_admission", "order": asdict(order), "timestamp": timestamp.isoformat(),
            "phase": self._phase, "policy": asdict(policy.config), "sector": sector, "sectors": self.policy_sectors,
            "exposure": exposure, "cash": self.cash, "equity": equity, "unit_cost": unit_cost,
            "offered_qty": quantity, "admitted_qty": final, "continuation": continuation, "reason": reason,
            "available_funding": funding,
            "quality_prediction": (self.quality_model.prediction(order.strategy_id, order.metadata, order.submitted_at)
                if getattr(self, "quality_model", None) else None),
            "reservation_policy": "admit_at_execution_phase_from_" + policy.config.funding_mode})
        return final

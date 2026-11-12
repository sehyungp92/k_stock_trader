"""Small, causal research ranker; scores are comparable net basis points.

Archived executed trades supply labels, never portfolio returns. These labels
describe the supported, selected population and cannot value unseen candidates.
One full session of label delay prevents partial exits and intraday ordering
from contaminating a decision. Hyperparameters are fixed before policy search.
"""
from collections import defaultdict
from datetime import datetime
import math
from statistics import mean, stdev

from backtests.auto.shared.cache_keys import stable_signature


def cohort(strategy, metadata):
    if strategy == "KALCB":
        return str(metadata.get("entry_route") or "unknown")
    return str(metadata.get("afternoon_score_band_rule") or "unknown")


class CausalQualityModel:
    def __init__(self, trades, *, minimum=20, prior_weight=20, uncertainty_penalty=.25):
        self.minimum = minimum
        self.prior_weight = prior_weight
        self.uncertainty_penalty = uncertainty_penalty
        self.labels = []
        for trade in trades:
            strategy = trade["strategy_id"]
            if strategy not in {"KALCB", "OLR"} or not trade.get("realized", True):
                continue
            entry = datetime.fromisoformat(trade["entry_fill_time"])
            exit_at = datetime.fromisoformat(trade["exit_fill_time"])
            notional = float(trade["qty"]) * float(trade["entry_price"])
            value = 10000 * float(trade["net_pnl"]) / notional if notional > 0 else float("nan")
            if entry.tzinfo is None or exit_at.tzinfo is None or exit_at < entry or not math.isfinite(value):
                raise ValueError("Invalid quality label")
            # Conservative 20bp tax for both sleeves, plus 10bp per-side stress.
            value -= 22 if strategy == "KALCB" else 20
            self.labels.append((strategy, cohort(strategy, trade.get("route_metadata", {})), exit_at, value))
        self.label_hash = stable_signature([(s, c, t.isoformat(), v) for s, c, t, v in self.labels])
        self.cache = {}

    def estimates(self, at):
        if at.tzinfo is None:
            raise ValueError("Decision timestamp must be timezone aware")
        day = at.date()
        if day in self.cache:
            return self.cache[day]
        groups, by_strategy, latest = defaultdict(list), defaultdict(list), {}
        for strategy, group, closed, value in self.labels:
            if closed.date() >= day:
                continue
            groups[(strategy, group)].append(value)
            by_strategy[strategy].append(value)
            latest[strategy] = max(latest.get(strategy, closed), closed)
        # Entire decision cohort uses fixed priority until BOTH priors exist.
        if any(len(by_strategy[s]) < self.minimum for s in ("KALCB", "OLR")):
            self.cache[day] = {}
            return {}
        result = {}
        for strategy, values in by_strategy.items():
            prior = mean(values)
            for group in [None] + [c for s, c in groups if s == strategy]:
                sample = groups.get((strategy, group), [])
                n = len(sample)
                estimate = (sum(sample) + self.prior_weight * prior) / (n + self.prior_weight)
                se = stdev(sample if n >= self.minimum else values) / math.sqrt(max(n, self.minimum))
                result[(strategy, group)] = {
                    "score_bps": estimate - self.uncertainty_penalty * se,
                    "mean_net_bps": estimate, "standard_error_bps": se,
                    "cohort_support": n, "strategy_support": len(values),
                    "latest_label_exit": latest[strategy].isoformat(), "label_hash": self.label_hash,
                }
        self.cache[day] = result
        return result

    def prediction(self, strategy, metadata, at):
        estimates = self.estimates(at)
        value = estimates.get((strategy, cohort(strategy, metadata)), estimates.get((strategy, None)))
        return None if value is None else dict(value)

    def score(self, strategy, metadata, at):
        prediction = self.prediction(strategy, metadata, at)
        return None if prediction is None else prediction["score_bps"]

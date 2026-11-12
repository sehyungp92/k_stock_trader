"""Action-independent, causal next-session auction forecasts for research.

Inputs are source-qualified daily observations, not the strategy's routed bars.
Unknown liquidity stays unknown. This module neither submits orders nor changes
the broker's independent executed-volume constraint.
"""
from datetime import date, timedelta
from functools import lru_cache
from math import exp, isfinite, log1p
from statistics import median

VERSION = 'olr-causal-liquidity-v6'
MODELS = ('broad_median', 'normalized_shrinkage')
PRIOR_STRENGTH = 5.0
LOOKBACK_DAYS = 60


def check_row(row):
    date.fromisoformat(row['day'])
    if not row.get('source_hash'):
        raise ValueError('Missing observation source')
    for key in ('prefix_volume', 'auction_volume'):
        value = row[key]
        if value is not None and (not isfinite(value) or value < 0):
            raise ValueError('Invalid observed volume')
    if row['auction_volume'] is not None and row['auction_available_at'][:10] != row['day']:
        raise ValueError('Invalid auction availability')


class CausalLiquidityPanel:
    """Predictions depend on dated market facts and a fixed universe only.

    A normalized pair uses a decision-day completed prefix and the *immediate
    next market session's* observed auction. Missing targets never bridge to a
    later available auction. A label enters history only on a later session.
    """
    def __init__(self, rows, sessions, universe):
        self.sessions = tuple(sessions)
        self.universe = tuple(sorted(universe))
        if list(self.sessions) != sorted(set(self.sessions)):
            raise ValueError('Unordered/duplicate sessions')
        if len(set(universe)) != len(universe):
            raise ValueError('Duplicate universe')
        self.rows = {}
        self.by_symbol = {s: [] for s in self.universe}
        for raw in rows:
            row = dict(raw); check_row(row)
            key = (row['day'], row['symbol'])
            if key in self.rows or row['day'] not in self.sessions or row['symbol'] not in self.universe:
                raise ValueError('Duplicate or out-of-scope observation')
            self.rows[key] = row
            self.by_symbol[row['symbol']].append(row)
        for values in self.by_symbol.values():
            values.sort(key=lambda r: r['day'])
        # The published daily panel is allowed to be sparse. No fabricated bar.
        self.pairs = []
        for prior, target in zip(self.sessions, self.sessions[1:]):
            for symbol in self.universe:
                a, z = self.rows.get((prior, symbol)), self.rows.get((target, symbol))
                if a and z and a['prefix_volume'] and z['auction_volume'] is not None:
                    self.pairs.append((target, symbol, log1p(z['auction_volume']) - log1p(a['prefix_volume'])))

    @lru_cache(maxsize=512)
    def _prior(self, day):
        cutoff = (date.fromisoformat(day)-timedelta(days=LOOKBACK_DAYS)).isoformat()
        pairs = [(d, s, v) for d, s, v in self.pairs if cutoff <= d < day]
        by_day, local = {}, {s: [] for s in self.universe}
        for d, s, v in pairs:
            by_day.setdefault(d, []).append(v)
            local[s].append(v)
        pooled = [median(v) for v in by_day.values()]
        return (median(pooled) if pooled else None, len(pooled), max(by_day, default=None), local)

    def forecast(self, day, symbol, model):
        if model not in MODELS or day not in self.sessions or symbol not in self.universe:
            raise ValueError('Unknown forecast contract')
        row = self.rows.get((day, symbol))
        cutoff = (date.fromisoformat(day)-timedelta(days=LOOKBACK_DAYS)).isoformat()
        own = [r for r in self.by_symbol[symbol]
               if cutoff <= r['day'] < day and r['auction_volume'] is not None]
        own = sorted(own, key=lambda r: r['day'])[-20:]
        evidence = dict(version=VERSION, model=model, day=day, symbol=symbol,
                        available_at=row['prefix_available_at'] if row else None,
                        own_observations=len(own), action_independent=True,
                        target='immediate_next_session_executable_auction_volume',
                        latest_own_day=max((r['day'] for r in own), default=None))
        if model == 'broad_median':
            value = median(r['auction_volume'] for r in own) if own else None
        else:
            prior, count, latest, local_by_symbol = self._prior(day)
            local = local_by_symbol[symbol][-20:]
            # One median per source session prevents a cross-section with more
            # observations from becoming an implicit date weight.
            weight = len(local)/(len(local)+PRIOR_STRENGTH)
            evidence.update(paired_observations=len(local), pooled_sessions=count,
                            prior_strength=PRIOR_STRENGTH, weight=weight,
                            latest_label_day=latest)
            if not row or not row['prefix_volume'] or prior is None:
                value = None
            else:
                local_center = median(local) if local else prior
                value = max(0.0, exp(log1p(row['prefix_volume']) + weight*local_center + (1-weight)*prior)-1)
        evidence.update(predicted_volume=value, status='available' if value is not None else 'unknown_fail_closed')
        return evidence


def request_bound(evidence, participation, available_at):
    """No under-three-observation bypass. Unknown yields zero new request."""
    from datetime import datetime
    from math import floor
    if not isfinite(participation) or not 0 < participation <= 1:
        raise ValueError('Invalid participation')
    if available_at.tzinfo is None:
        raise ValueError('Aware request clock required')
    stamp = evidence.get('available_at')
    if stamp is None:
        return 0
    observed = datetime.fromisoformat(stamp)
    if observed.tzinfo is None or observed > available_at or observed.date() != available_at.date():
        raise ValueError('Forecast not yet available or stale')
    for key in ('latest_own_day', 'latest_label_day'):
        if evidence.get(key) is not None and evidence[key] >= available_at.date().isoformat():
            raise ValueError('Future liquidity label')
    volume = evidence['predicted_volume']
    if volume is None:
        return 0
    if not isfinite(volume) or volume < 0:
        raise ValueError('Invalid forecast')
    return floor(volume*participation)

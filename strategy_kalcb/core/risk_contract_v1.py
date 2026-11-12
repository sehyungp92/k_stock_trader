"""Explicit, observational financial-risk contract; does not emit trading actions.

Reference stop risk is not a loss guarantee. Completed-close marks are not fills.
This module has no configuration side effects and is not enabled in live policy.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite


def timestamp(value):
    value = datetime.fromisoformat(str(value))
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('Risk evidence requires an aware timestamp')
    return value


def positive(value, name):
    value = float(value)
    if not isfinite(value) or value <= 0:
        raise ValueError(name)
    return value


@dataclass(frozen=True)
class EntryRiskReference:
    order_id: str
    symbol: str
    available_at: str
    planned_quantity: int
    reference_price: float
    reference_risk_per_share: float
    reference_stop: float
    equity: float
    daily_atr: float
    hard_stop_enabled: bool

    @classmethod
    def from_exposure(cls, exposure):
        action = exposure['action']; meta = action['metadata']
        q = action['qty']
        if isinstance(q, bool) or int(q) != q or q <= 0:
            raise ValueError('Invalid planned integer quantity')
        reference = positive(meta['entry_price_ref'], 'Reference price')
        risk = positive(meta['risk_per_share'], 'Reference risk')
        stop = positive(meta['stop_price'], 'Reference stop')
        if abs(max(reference-stop, 1.)-risk) > 1e-7:
            raise ValueError('Conflicting sizing risk and stop metadata')
        when = timestamp(exposure['decision_available_at']).isoformat()
        if not exposure['order_id'] or not action['symbol']:
            raise ValueError('Missing action identity')
        return cls(exposure['order_id'], action['symbol'], when, int(q), reference,
                   risk, stop, positive(exposure['account_before']['actual_equity'], 'Equity'),
                   positive(meta['daily_atr'], 'ATR'), bool(meta['hard_stop_enabled']))

    def evidence(self):
        return {'planned_reference_risk_krw': self.planned_quantity*self.reference_risk_per_share,
                'planned_reference_risk_equity': self.planned_quantity*self.reference_risk_per_share/self.equity,
                'reference_risk_price_fraction': self.reference_risk_per_share/self.reference_price,
                'atr_to_reference_risk': self.daily_atr/self.reference_risk_per_share,
                'hard_stop_enabled': self.hard_stop_enabled,
                'guaranteed_loss_bound': False,
                'scope': 'Sizing reference only; physical risk and financial loss are different.'}


def mark_at_completed_bar(reference, fills, raw_bar_at, close):
    """Native OPEN fills through this raw bar; exclude next-OPEN boundary fills."""
    at = timestamp(raw_bar_at); available = at+timedelta(minutes=5)
    if available <= timestamp(reference.available_at):
        raise ValueError('No holding evidence from the entry signal bar')
    price = positive(close, 'Completed close')
    cash, fees, qty, bought, cursor = 0., 0., 0, 0, None
    seen = set()
    for fill in fills:
        when = timestamp(fill['timestamp'])
        if cursor is not None and when < cursor:
            raise ValueError('Unordered fills')
        cursor = when
        eid = fill['metadata']['execution_id']
        if eid in seen:
            raise ValueError('Duplicate execution')
        seen.add(eid)
        if fill['symbol'] != reference.symbol:
            raise ValueError('Different holding')
        if when > at:
            continue
        q = fill['qty']
        if isinstance(q, bool) or q <= 0 or int(q) != q or fill['side'] not in ('BUY', 'SELL'):
            raise ValueError('Invalid fill')
        if when < timestamp(reference.available_at):
            raise ValueError('Fill precedes decision')
        sign = 1 if fill['side'] == 'BUY' else -1
        qty += sign*q; bought += q if sign == 1 else 0
        cash -= sign*q*positive(fill['price'], 'Fill price')
        fee = float(fill['metadata']['fee'])
        if not isfinite(fee) or fee < 0 or qty < 0:
            raise ValueError('Invalid fee/inventory')
        fees += fee
    if qty <= 0:
        return None
    gross = cash+qty*price
    risk = bought*reference.reference_risk_per_share
    return {'raw_bar_at': at.isoformat(), 'available_at': available.isoformat(),
            'close': price, 'held_qty': qty, 'bought_qty': bought,
            'pending_planned_qty': max(0, reference.planned_quantity-bought),
            'gross_marked_pnl_krw': gross, 'fees_paid_krw': fees,
            'net_marked_pnl_krw': gross-fees,
            'filled_reference_risk_krw': risk,
            'marked_net_loss_reference_r': max(0, fees-gross)/risk,
            'close_below_original_reference_stop': price <= reference.reference_stop,
            'scope': 'Observed mark, not an attainable exit price or counterfactual PnL.'}


def analyze_lifetime(reference, position, fills, bars, first_parent_exit=None):
    if reference.order_id not in position['entry_order_ids']:
        raise ValueError('Wrong lifetime identity')
    if len(position['entry_order_ids']) != 1:
        raise ValueError('Multi-order lifetime needs separate reference contract')
    first = timestamp(position['entry_fill_time']); last = timestamp(position['exit_fill_time'])
    points = [mark_at_completed_bar(reference, fills, at, close) for at, close in bars
              if first <= timestamp(at) < last]
    points = [p for p in points if p is not None]
    if not points:
        raise ValueError('Missing open-position completed bars')
    if any(timestamp(b['raw_bar_at']) <= timestamp(a['raw_bar_at']) for a,b in zip(points,points[1:])):
        raise ValueError('Duplicate/unordered raw bars')
    exit_at = timestamp(first_parent_exit) if first_parent_exit else None
    breaches = [p for p in points if p['close_below_original_reference_stop']]
    independent = [p for p in breaches if exit_at is None or timestamp(p['available_at']) < exit_at]
    first_breach = independent[0] if independent else None
    result = {**position, **reference.evidence(), 'original_reference_stop': reference.reference_stop,
              'raw_marks': len(points), 'reference_stop_breach': bool(breaches),
              'before_parent_exit_breach': first_breach,
              'worst_marked_net_pnl_krw': min(p['net_marked_pnl_krw'] for p in points),
              'max_marked_net_loss_reference_r': max(p['marked_net_loss_reference_r'] for p in points)}
    if first_breach:
        result['observed_after_breach_net_movement_krw'] = position['net_pnl_krw']-first_breach['net_marked_pnl_krw']
        result['post_breach_scope'] = 'Signed observed subsequent movement/fees, NOT an executable saving or exit label.'
    return result

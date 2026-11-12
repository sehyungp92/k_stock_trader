"""Research-only linked entry commitment. No live adapter enables this module.

The original order identity and one position lifetime survive a completion
tranche. Fill identities are never aliased. All decisions use completed bars.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from hashlib import sha256
from math import floor, isfinite

from strategy_common.actions import SubmitEntry, SubmitExit, SubmitPartialExit, FlattenPosition, CancelOrders
from strategy_common.events import DecisionEvent
from strategy_common.fill_lifecycle import entry_order_key, is_duplicate_fill

KEY = 'staged_commitments_v1'
REASON = 'staged_original_commitment_completion'


def contracts(state):
    return state.meta.setdefault(KEY, {})


def for_order(state, order_id):
    return next((c for c in state.meta.get(KEY, {}).values()
                 if order_id in (c.get('initial_id'), c.get('completion_id'))), None)


def linked_allowed(state, position, fill):
    c = for_order(state, fill.order_id)
    return bool(c and fill.side.upper() == 'BUY'
                and fill.symbol == c['symbol'] == position.symbol
                and fill.order_id == c.get('completion_id')
                and position.entry_order_id == c.get('initial_id')
                and position.metadata.get('staged_commitment_id') == c['id']
                and c['completion_filled'] + fill.qty <= c['completion_qty']
                and c['bought'] + fill.qty <= c['planned_qty'])


def history_for_fill(state, fill):
    ss = state.symbol_state(fill.symbol)
    direct = ss.entry_history.get(entry_order_key(fill))
    if direct is not None:
        return deepcopy(direct)
    c = for_order(state, fill.order_id)
    if c and fill.symbol == c['symbol'] and c.get('lifecycle_key'):
        return deepcopy(ss.entry_history.get(c['lifecycle_key']))
    return None


def stage_initial(state, result, available):
    if not state.meta.get('staged_commitment_enabled', False):
        return result
    actions = []
    for a in result.actions:
        if not isinstance(a, SubmitEntry) or a.metadata.get('entry_route') != 'first30_open_anchor' or a.qty < 2:
            actions.append(a); continue
        reference = float(a.metadata['entry_price_ref'])
        if not isfinite(reference) or reference <= 0:
            raise ValueError('Invalid initial reference')
        cid = sha256(f'{a.symbol}|{available.isoformat()}|{a.qty}|{reference}'.encode()).hexdigest()
        if cid in contracts(state):
            raise ValueError('Duplicate initial commitment')
        c = {'id': cid, 'symbol': a.symbol, 'planned_qty': a.qty, 'initial_qty': a.qty // 2,
             'reference': reference, 'available_at': available.isoformat(),
             'completion_at': (available + timedelta(minutes=5)).isoformat(),
             'initial_id': None, 'completion_id': None, 'bought': 0, 'bought_notional': 0.,
             'initial_filled': 0, 'completion_qty': 0, 'completion_filled': 0,
             'attempted': False, 'sale_or_exit': False, 'metadata': dict(a.metadata), 'events': []}
        contracts(state)[cid] = c
        meta = {**a.metadata, 'staged_commitment_id': cid, 'staged_leg': 'initial',
                'staged_planned_quantity': a.qty, 'risk_reserved_entry_qty': c['initial_qty']}
        changed = replace(a, qty=c['initial_qty'], metadata=meta)
        actions.append(changed)
        ss = state.symbol_state(a.symbol)
        ss.pending_entry_metadata = dict(meta)
        c['events'].append({'at': available.isoformat(), 'event': 'initial_split',
                            'original_qty': a.qty, 'initial_qty': changed.qty})
    result.actions = actions
    return result


def protect_exit(state, result, at):
    """A pending linked fill after any sale intent must be liquidated normally."""
    extra = []
    for a in result.actions:
        if not isinstance(a, (SubmitExit, SubmitPartialExit, FlattenPosition)):
            continue
        ss = state.symbol_state(a.symbol)
        if not ss.position:
            continue
        c = for_order(state, ss.position.entry_order_id)
        if not c:
            continue
        c['sale_or_exit'] = True
        if ss.pending_entry_order_id:
            oid = ss.pending_entry_order_id
            ss.pending_entry_metadata['cancel_late_fill_reason'] = 'staged_completion_after_exit_intent'
            state.order_roles.get(oid, {})['cancel_late_fill_reason'] = 'staged_completion_after_exit_intent'
            if not any(isinstance(x, CancelOrders) and x.symbol == a.symbol for x in result.actions):
                extra.append(CancelOrders(a.strategy_id, a.symbol, 'cancel_staged_completion_before_exit'))
        c['events'].append({'at': at.isoformat(), 'event': 'completion_closed_by_parent_exit'})
    result.actions = extra + result.actions
    return result


def completion_action(state, symbol, clock, config, portfolio):
    ss = state.symbol_state(symbol)
    candidates = [c for c in state.meta.get(KEY, {}).values()
                  if c['symbol'] == symbol and not c['attempted'] and clock >= datetime.fromisoformat(c['completion_at'])]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError('Ambiguous commitment')
    c = candidates[0]; c['attempted'] = True
    pos = ss.position
    reason = None
    prefix = ss.bars_today
    if clock != datetime.fromisoformat(c['completion_at']): reason = 'missed_completion_clock'
    elif not pos or pos.entry_order_id != c['initial_id']: reason = 'initial_position_unavailable'
    elif c['sale_or_exit'] or pos.exit_in_flight or pos.partial_taken or pos.qty_open != pos.qty_entry: reason = 'parent_exit_or_sale'
    elif ss.pending_entry_order_id or c['initial_filled'] != c['initial_qty']: reason = 'initial_fill_incomplete'
    elif len(prefix) != 7 or any(not b.is_completed or b.volume <= 0 for b in prefix): reason = 'incomplete_positive_volume_prefix'
    elif any(b.timestamp != prefix[0].timestamp + timedelta(minutes=5*i) for i,b in enumerate(prefix)) or prefix[-1].timestamp + timedelta(minutes=5) != clock: reason = 'nonconsecutive_prefix'
    elif prefix[-1].close < c['reference'] or prefix[-1].close < ss.vwap: reason = 'confirmation_not_met'
    elif portfolio.open_risk / max(portfolio.cash * config.risk_per_trade_pct, 1.) >= config.heat_cap_r: reason = 'native_heat_gate'
    if reason:
        c['events'].append({'at': clock.isoformat(), 'event': 'completion_abstention', 'reason': reason})
        return None
    close = prefix[-1].close
    # Never increase the original integer dose or its completed-price notional.
    # Native next-OPEN gaps/costs/funding/capacity still apply; no fill guarantee.
    qty = min(c['planned_qty'] - c['bought'], floor(max(0., c['planned_qty'] * c['reference'] - c['bought_notional']) / close))
    if qty <= 0:
        c['events'].append({'at': clock.isoformat(), 'event': 'completion_abstention', 'reason': 'no_remaining_original_commitment'})
        return None
    c['completion_qty'] = qty
    meta = {**c['metadata'], 'staged_commitment_id': c['id'], 'staged_leg': 'completion',
            'staged_parent_order_id': c['initial_id'], 'staged_planned_quantity': c['planned_qty'],
            'decision_available_at': clock.isoformat(), 'source_bar_timestamp': prefix[-1].timestamp.isoformat(),
            'staged_completion_reference_price': close, 'risk_reserved_entry_qty': qty,
            'expiry_timestamp': (clock + timedelta(minutes=5)).isoformat(), 'order_role': 'ENTRY'}
    c['events'].append({'at': clock.isoformat(), 'event': 'completion_intent', 'qty': qty,
                        'completed_close': close, 'vwap': ss.vwap, 'actual_equity': portfolio.equity,
                        'actual_cash': portfolio.cash, 'actual_open_risk': portfolio.open_risk})
    return SubmitEntry('KALCB', symbol, qty, 'MARKET', None, None, REASON, meta)


def install(namespace):
    """Install only in an authenticated research derivative of core.logic."""
    if namespace.get('_staged_installed'):
        raise ValueError('Duplicate staged core installation')
    namespace['_staged_installed'] = True
    step, timer, fill, remember, update = (namespace[k] for k in
        ('step_kalcb_core', 'on_kalcb_timer', 'on_kalcb_fill', 'remember_submitted_order', 'on_kalcb_order_update'))
    namespace['_staged_parent_api'] = dict(zip(
        ('step_kalcb_core','on_kalcb_timer','on_kalcb_fill','remember_submitted_order','on_kalcb_order_update'),
        (step,timer,fill,remember,update)))

    def new_step(state, bar, config, candidate_snapshot, portfolio):
        result = step(state, bar, config, candidate_snapshot, portfolio)
        clock = bar.timestamp + timedelta(minutes=5)
        stage_initial(state, result, clock)
        protect_exit(state, result, clock)
        # Store the actual current account view for the shared timer, after all
        # bar decisions. The adapter continues to submit normal native actions.
        if state.meta.get('staged_commitment_enabled', False):
            state.meta['staged_account_at'] = clock.isoformat()
            state.meta['staged_account'] = {k:getattr(portfolio,k) for k in portfolio.__dataclass_fields__}
        return result

    def new_timer(state, timestamp, config):
        result = protect_exit(state, timer(state, timestamp, config), timestamp)
        if not state.meta.get('staged_commitment_enabled', False): return result
        if state.meta.get('staged_account_at') != timestamp.isoformat(): return result
        from .core_models import KALCBPortfolioView
        portfolio = KALCBPortfolioView(**state.meta['staged_account'])
        for symbol in sorted(state.symbols, key=lambda s: state.symbols[s].candidate_rank):
            action = completion_action(state, symbol, timestamp, config, portfolio)
            if action:
                result.actions.append(action)
                result.decisions.append(DecisionEvent(timestamp, 'KALCB', symbol,
                    'staged_completion', REASON, actions=(action,), metadata=dict(action.metadata)))
        return result

    def new_remember(state, order_id, action):
        cid = action.metadata.get('staged_commitment_id') if isinstance(action, SubmitEntry) else None
        if cid and order_id:
            c = contracts(state).get(cid)
            if not c or c['symbol'] != action.symbol: raise ValueError('Unregistered staged action')
            key = 'initial_id' if action.metadata.get('staged_leg') == 'initial' else 'completion_id'
            qty = c['initial_qty'] if key == 'initial_id' else c['completion_qty']
            if c[key] is not None or qty != action.qty or not qty: raise ValueError('Duplicate/altered commitment order')
            c[key] = order_id
        remember(state, order_id, action)

    def new_fill(state, event, config):
        if is_duplicate_fill(state.meta, event): return fill(state, event, config)
        c = for_order(state, event.order_id)
        if c and event.side.upper() == 'BUY':
            if c['symbol'] != event.symbol or event.qty <= 0: raise ValueError('Staged fill symbol/quantity mismatch')
            key = 'initial_filled' if event.order_id == c['initial_id'] else 'completion_filled'
            limit = c['initial_qty'] if key == 'initial_filled' else c['completion_qty']
            if c[key]+event.qty > limit or c['bought']+event.qty > c['planned_qty']:
                raise ValueError('Staged fill exceeds declared commitment')
            if c['sale_or_exit']:
                event = replace(event, metadata={**event.metadata, 'cancel_late_fill_reason':'staged_completion_after_exit_intent'})
        elif event.side.upper() == 'SELL':
            pos = state.symbol_state(event.symbol).position
            c = for_order(state, pos.entry_order_id) if pos else None
        result = fill(state, event, config)
        if c:
            if event.side.upper() == 'BUY':
                c[key] += event.qty; c['bought'] += event.qty; c['bought_notional'] += event.qty*event.price
                c.setdefault('lifecycle_key', state.symbol_state(event.symbol).position.metadata['entry_lifecycle_key'])
            else:
                c['sale_or_exit'] = True
        return protect_exit(state, result, event.timestamp)

    def new_update(state, event):
        result = update(state, event)
        c = for_order(state, event.order_id)
        if c and event.status.upper() in {'BLOCKED','REJECTED','CANCELLED','DEFERRED','EXPIRED'}:
            c['events'].append({'at':event.timestamp.isoformat(), 'event':'native_order_terminal',
                                'order_id':event.order_id, 'status':event.status})
            if event.order_id == c['completion_id']:
                state.order_history.get(event.order_id, {})['cancel_late_fill_reason'] = 'staged_completion_terminal_late_fill'
        return result

    namespace.update(step_kalcb_core=new_step, on_kalcb_timer=new_timer,
        on_kalcb_fill=new_fill, remember_submitted_order=new_remember, on_kalcb_order_update=new_update)


def derive_logic(text):
    """Exact source transformation; no mutation of a registered original file."""
    replacements = {
        'existing = deepcopy(symbol_state.entry_history.get(entry_order_key(fill)))':
            'existing = _staged_history_for_fill(state, fill)',
        'if fill.side.upper() == "BUY" and existing.entry_order_id != fill.order_id:':
            'if fill.side.upper() == "BUY" and existing.entry_order_id != fill.order_id and not _staged_linked_allowed(state, existing, fill):',
        'if position.entry_order_id != fill.order_id:':
            'if position.entry_order_id != fill.order_id and not _staged_linked_allowed(state, position, fill):',
    }
    for original, changed in replacements.items():
        if text.count(original) != 1: raise ValueError('Unexpected original core contract: '+original)
        text = text.replace(original, changed)
    return text + '\nfrom .staged_commitment_v1 import (linked_allowed as _staged_linked_allowed, history_for_fill as _staged_history_for_fill, install as _install_staged)\n_install_staged(globals())\n'

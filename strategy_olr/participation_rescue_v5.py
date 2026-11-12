"""Bounded, causal empty-day participation exception and admission envelope.

Research opt-in only. Native selectors must construct both snapshots. This
module never reads future returns, changes a nonempty selection, or creates a
signal without the original downstream entry rule.
"""
from dataclasses import replace
from math import floor, isfinite

TAG = 'empty_participation_rescue_v5'
NAME_FRACTION = .05
COHORT_FRACTION = .10


def confirmed(candidate):
    f = candidate.metadata['afternoon_features']
    values = tuple(float(f[k]) for k in ('lagged_flow_5d', 'sector_intraday_ret',
        'sector_intraday_breadth', 'sector_intraday_effective_count'))
    return all(isfinite(v) for v in values) and values[0] > 0 and values[1] > 0 and values[2] >= .5 and values[3] >= 3


def select(baseline, alternative, config):
    from .research import final_candidate_config_fingerprint
    if baseline.trade_date != alternative.trade_date or baseline.generated_at != alternative.generated_at:
        raise ValueError('Participation exception requires matching causal selection times')
    sole = {s for s, reasons in baseline.metadata['rejected_symbols'].items()
            if reasons == ['sector_participation_below_floor']}
    # Preserve the exact previously screened order: native top six FIRST,
    # then sole-rejection membership and current sector confirmation.
    additions = tuple(replace(c, metadata={**c.metadata, TAG: True,
        'participation_original_reasons': ['sector_participation_below_floor']})
        for c in alternative.candidates[:6] if c.symbol in sole and confirmed(c)) if not baseline.candidates else ()
    chosen = baseline.candidates if baseline.candidates else additions
    fingerprint = final_candidate_config_fingerprint(config)
    metadata = {**baseline.metadata, 'candidate_config_hash': fingerprint,
        'final_candidate_config_hash': fingerprint, TAG: {
            'baseline_empty': not bool(baseline.candidates), 'sole_rejects': sorted(sole),
            'rescued_symbols': [c.symbol for c in additions],
            'alternate_top_six': [c.symbol for c in alternative.candidates[:6]] if not baseline.candidates else [],
            'rule': 'sole daily participation rejection; original native top-six and score gates; positive own flow and sector return, breadth >= .5 and >=3 names; original six-bar entry'}}
    return replace(baseline, candidates=chosen, metadata=metadata)


def envelope(state, bars, portfolio):
    """Current marked family holdings plus unfilled native commitments.

    Limits apply at admission. Appreciation can subsequently exceed a limit;
    it cannot authorize another acquisition. A pending sale releases nothing.
    """
    from .core.logic import pending_entry_reservation
    marks = {b.symbol: float(b.close) for b in bars}
    held = pending = 0.
    for symbol, item in state.symbols.items():
        pos = item.position
        if pos is not None and pos.metadata.get(TAG):
            if symbol not in marks:
                if not item.session_bars:
                    raise ValueError('Missing current/prior mark for rescued holding')
                marks[symbol] = float(item.session_bars[-1].close)
            held += pos.qty_open * marks[symbol]
        if item.pending_entry_order_id and item.pending_entry_metadata.get(TAG):
            pending += pending_entry_reservation(item)[1]
    equity = max(0., float(portfolio.equity))
    return {'equity': equity, 'name_limit': equity * NAME_FRACTION,
            'cohort_limit': equity * COHORT_FRACTION, 'held_marked': held,
            'pending_reserved': pending, 'remaining': max(0., equity * COHORT_FRACTION - held - pending)}


def constrain(proposals, account, state, bars, portfolio):
    """Clip new-family quantities and the shared budget before native allocation."""
    flagged = [a for a in proposals if a.metadata.get(TAG)]
    if not flagged:
        return proposals, account
    if len(flagged) != len(proposals):
        raise ValueError('Empty-day exception cannot mix with incumbent entry proposals')
    proof = envelope(state, bars, portfolio)
    result = []
    for action in proposals:
        m = dict(action.metadata)
        unit = float(m['estimated_entry_price']) * (1 + float(m['entry_cost_buffer_pct']))
        if not isfinite(unit) or unit <= 0:
            raise ValueError('Invalid rescued-entry cost')
        qty = min(action.qty, max(0, floor(proof['name_limit'] / unit)))
        if qty == 0:
            continue
        m.update(participation_envelope_v5={**proof, 'unit_cost': unit,
            'native_proposed_qty': action.qty, 'clipped_proposed_qty': qty},
            target_notional=min(float(m['target_notional']), proof['name_limit']))
        result.append(replace(action, qty=qty, metadata=m))
    return result, replace(account, funding_limit=min(account.funding_limit, proof['remaining']))

"""Opt-in research rule copied into an immutable engine, never live config."""
from dataclasses import asdict
from datetime import timedelta
from math import floor
from strategy_common.actions import SubmitExit, SubmitPartialExit
from strategy_common.events import DecisionEvent
from .core_models import OLRCoreResult

KEY = 'relative_context_v1'


def maybe_reduce(state, ss, position, bar, config, plan, risk):
    bundle = state.meta.get(KEY)
    if not bundle or bundle['fraction'] == 0:
        return None
    available = bar.timestamp + timedelta(minutes=5)
    if available.strftime('%H:%M') != '09:30':
        return None
    if (ss.pending_entry_order_id or ss.pending_exit_order_id
            or bar.symbol in state.meta.get('priority_exit_pending', {})
            or position.metadata.get(KEY + '_attempted')
            or position.entry_time.date() >= bar.timestamp.date()):
        return None
    evidence = bundle['rows'].get(bar.symbol)
    if evidence is None:
        return None
    if (evidence['available_at'] != available.isoformat()
            or evidence['source_last_label'] != bar.timestamp.isoformat()
            or evidence['close'] != bar.close):
        raise ValueError('Relative context not aligned with the actual completed decision bar')
    if (evidence['peer_count'] < 68 or evidence['own_return'] >= 0
            or evidence['market_median'] < 0):
        return None
    fraction = bundle['fraction']
    if fraction not in (.5, 1.):
        raise ValueError('Unregistered relative-context action')
    quantity = floor(position.qty_open * fraction)
    if quantity < 1:
        return None
    position.metadata[KEY + '_attempted'] = True
    metadata = dict(order_role='RELATIVE_CONTEXT', order_type='MARKET', risk_per_share=risk,
        trade_exit_plan=asdict(plan), source_artifact_hash=position.source_artifact_hash,
        candidate_rank=position.candidate_rank, candidate_score=position.candidate_score,
        relative_context_evidence=dict(**evidence, fraction=fraction, quantity_before=position.qty_open,
            original_risk=risk, feature_sha256=bundle['feature_sha256']))
    partial = fraction == .5
    reason = 'relative_context_reduce' if partial else 'relative_context_exit'
    action = (SubmitPartialExit if partial else SubmitExit)(
        'OLR', bar.symbol, quantity, 'MARKET', None, reason, metadata)
    decision = DecisionEvent(bar.timestamp, 'OLR', bar.symbol, 'EXIT_SUBMITTED', reason,
        metadata=dict(**metadata, qty=quantity))
    return OLRCoreResult(state=state, actions=[action], decisions=[decision])

"""One bounded reduction for an observed next-session failure to follow through."""
from dataclasses import asdict
from math import floor
from strategy_common.actions import SubmitPartialExit
from strategy_common.events import DecisionEvent
from .core_models import OLRCoreResult

REASON='weak_followthrough_partial'


def maybe_reduce(state, symbol_state, position, bar, config, plan, risk):
    fraction=config.capacity_weak_partial_fraction
    if (not fraction or symbol_state.pending_exit_order_id or symbol_state.pending_entry_order_id
            or position.metadata.get('weak_partial_attempted') or position.qty_open<2
            or bar.timestamp.date()<=position.entry_time.date() or risk<=0):
        return None
    session={b.timestamp for b in symbol_state.session_bars
             if b.timestamp.date()==bar.timestamp.date() and position.entry_time<b.timestamp<=bar.timestamp}
    close_r=(bar.close-position.entry_price)/risk
    mfe_r=(position.max_favorable_price-position.entry_price)/risk
    if len(session)<3 or close_r>-.4 or mfe_r>=.5:
        return None
    qty=min(position.qty_open-1,max(1,floor(position.qty_open*fraction)))
    # At most one attempt per position lifecycle; rejection must not produce a
    # stream of repeated sells. The original full-risk guard remains active.
    position.metadata['weak_partial_attempted']=True
    metadata={'order_role':'WEAK_PARTIAL','order_type':'MARKET','risk_per_share':risk,
        'trade_exit_plan':asdict(plan),'source_artifact_hash':position.source_artifact_hash,
        'candidate_rank':position.candidate_rank,'candidate_score':position.candidate_score,
        'weak_partial_evidence':{'session_bars':len(session),'close_r':close_r,'mfe_r':mfe_r,
            'fraction':fraction,'remaining_before':position.qty_open}}
    action=SubmitPartialExit('OLR',bar.symbol,qty,'MARKET',None,REASON,metadata)
    decision=DecisionEvent(bar.timestamp,'OLR',bar.symbol,'EXIT_SUBMITTED',REASON,
                          metadata={**metadata,'qty':qty})
    return OLRCoreResult(state=state,actions=[action],decisions=[decision])

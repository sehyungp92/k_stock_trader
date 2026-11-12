"""Bounded, checkpointable uncommitted opportunities for research allocation.

Waiting reserves nothing. Submission consumes the attempt; a cancellation
request never releases cash or authorizes an unrelated new entry lifecycle.
"""
from dataclasses import dataclass, asdict, replace
from datetime import datetime, timedelta
from math import floor

from strategy_common.actions import action_to_json_dict, action_from_json_dict

KEY = 'economic_opportunities_v1'


@dataclass(frozen=True)
class Opportunity:
    symbol: str
    origin: str
    expires: str
    status: str
    action: dict


def proposals(state, bars, config, at, fresh, *, wait_minutes=15):
    """Reconsider the same signal until its original bounded deadline.

    Original entry confirmation is not re-run as a continuation filter. Current
    price, costs and capacity are refreshed; the economic provider values it.
    """
    from .capacity import forecast
    from .logic import _market_entry_sizing_price
    ledger=state.meta.setdefault(KEY,{})
    current={b.symbol:b for b in bars}
    for a in fresh:
        old=ledger.get(a.symbol)
        if old and datetime.fromisoformat(old['origin']).date()==at.date():
            continue
        end=min(datetime.fromisoformat(a.metadata['expiry_ts']),at+timedelta(minutes=wait_minutes))
        ledger[a.symbol]=asdict(Opportunity(a.symbol,at.isoformat(),end.isoformat(),'watching',action_to_json_dict(a)))
    result=[]
    for symbol,row in ledger.items():
        ss=state.symbol_state(symbol)
        if row['status']!='watching':continue
        if ss.position is not None or ss.pending_entry_order_id:
            row['status']='committed';continue
        if at>datetime.fromisoformat(row['expires']):
            row['status']='expired';ss.entry_attempted=True;continue
        if symbol not in current:continue
        original=action_from_json_dict(row['action']);meta=dict(original.metadata)
        price=_market_entry_sizing_price(current[symbol],config)
        unit=price*(1+meta['entry_cost_buffer_pct'])
        q=floor(meta['target_notional']/unit)
        cap,_,evidence=forecast(state,current[symbol],config)
        if cap is not None:q=min(q,cap)
        if q<=0:continue
        meta.update(estimated_entry_price=price,reserved_notional=q*unit,
            opportunity_origin=row['origin'],opportunity_expires=row['expires'],
            capacity_forecast=evidence,capacity_approved_qty=q)
        # The deadline to elect an opportunity is not the lifetime of an
        # elected native order. Shortening acquisition here silently changes
        # the quantity response being priced and recreates premature cancels.
        result.append(replace(original,qty=q,metadata=meta))
    return result


def commit(state,symbol):
    state.symbol_state(symbol).entry_attempted=True
    row=state.meta.get(KEY,{}).get(symbol)
    if row:row['status']='committed'

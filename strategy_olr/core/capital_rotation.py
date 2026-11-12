"""Conditional late-session capital recycling, with acknowledged order changes.

An incoming confirmed native signal must lack cash. Incumbents are from prior
sessions; a weak current VWAP condition is optional for its declared ablation.
No expected sale proceeds enter the allocator. Entry validity never refreshes.
"""
from datetime import datetime, timedelta
from math import ceil, floor

from deployment.olr_kalcb.intent_allocator import allocate_entry_intents, IntentAllocationConfig
from strategy_common.actions import CancelOrders, SubmitExit, SubmitPartialExit
from strategy_common.events import DecisionEvent


def rotation_actions(state, bars, config, intents, account, actions, at):
    from .logic import _running_vwap, _position_risk, _maybe_submit_next_close_exit
    actions=list(actions); events=[]
    plans=state.meta.setdefault('capacity_rotation_pending',{})
    by_symbol={b.symbol:b for b in bars}
    decisions=allocate_entry_intents(intents,account,at,IntentAllocationConfig())
    by_id={d.intent_id:d for d in decisions}
    wanted=[i for i in intents if i.native_rank<=config.capacity_rotation_max_rank
            and by_id[i.intent_id].approved_qty<i.desired_qty]
    need=sum((i.desired_qty-by_id[i.intent_id].approved_qty)*i.unit_cost for i in wanted)
    waiting=False

    def discard_auction(symbol):
        nonlocal actions
        actions=[a for a in actions if not (a.symbol==symbol and
            ((isinstance(a,SubmitExit) and a.reason=='next_close_exit') or
             (isinstance(a,CancelOrders) and a.reason=='replace_stop_with_next_close_exit')))]

    def submit(symbol, limit_qty):
        nonlocal need,waiting
        s=state.symbols[symbol];position=s.position;bar=by_symbol[symbol]
        net_price=bar.close*(1-(config.commission_bps+config.tax_bps_on_sell+config.slippage_bps)/10000)
        qty=min(position.qty_open,limit_qty,ceil(need/max(net_price,1e-9)))
        if qty<=0:return
        discard_auction(symbol)
        position.metadata['capacity_rotation_date']=at.date().isoformat()
        metadata={'strategy_core_version':position.metadata.get('strategy_core_version',''),
            'source_artifact_hash':position.source_artifact_hash,
            'candidate_rank':position.candidate_rank,'candidate_score':position.candidate_score,
            'order_role':'CAPACITY_SLICE','risk_per_share':_position_risk(position),
            'rotation_requested_at':at.isoformat(),'rotation_required_cash':need,
            'rotation_entry_symbols':sorted(i.symbol for i in wanted),
            'rotation_filter':config.capacity_rotation_filter,'trade_exit_plan':vars_plan(config)}
        action=(SubmitExit('OLR',symbol,qty,'MARKET',None,'opportunity_capital_rotation',metadata)
            if qty==position.qty_open else
            SubmitPartialExit('OLR',symbol,qty,'MARKET',None,'opportunity_capital_rotation',metadata))
        actions.extend([CancelOrders('OLR',symbol,'opportunity_capital_rotation'),action])
        # If only a slice is sold, native next-close liquidation resumes after
        # its confirmed completion; no overlapping auction and market sell.
        s.exit_attempted_dates.discard(at.date())
        events.append(DecisionEvent(bar.timestamp,'OLR',symbol,'CAPITAL_ROTATION_SUBMITTED',
            'opportunity_capital_rotation',metadata={**metadata,'qty':qty}))
        need=max(0,need-qty*net_price);waiting=True

    def restore(symbol):
        s=state.symbols[symbol]
        if not s.pending_exit_order_id and symbol in by_symbol:
            s.exit_attempted_dates.discard(at.date())
            if not any(a.symbol==symbol and isinstance(a,(SubmitExit,SubmitPartialExit)) for a in actions):
                result=_maybe_submit_next_close_exit(state,by_symbol[symbol],config)
                actions.extend(result.actions);events.extend(result.decisions)
            return True
        return False

    for symbol,plan in list(plans.items()):
        s=state.symbols.get(symbol)
        if s is None or s.position is None:
            plans.pop(symbol);continue
        if at>datetime.fromisoformat(plan['end']) or not wanted or plan.get('restore'):
            plan['restore']=True
            if restore(symbol):plans.pop(symbol)
            continue
        if s.pending_exit_order_id:
            role=state.order_roles.get(s.pending_exit_order_id,{})
            if role.get('reason')!='next_close_exit':plans.pop(symbol)
            else:
                waiting=True
                if symbol in by_symbol:
                    net_price=by_symbol[symbol].close*(1-(config.commission_bps+config.tax_bps_on_sell+config.slippage_bps)/10000)
                    need=max(0,need-min(plan['qty'],s.position.qty_open)*net_price)
            continue
        if symbol not in by_symbol:
            continue
        if any(a.symbol==symbol and isinstance(a,(SubmitExit,SubmitPartialExit)) and a.order_type=='MARKET' for a in actions):
            plans.pop(symbol);waiting=True;continue
        submit(symbol,plan['qty']);plans.pop(symbol)

    if need>0 and wanted:
        choices=[]
        for symbol,s in state.symbols.items():
            pos=s.position;bar=by_symbol.get(symbol)
            if (pos is None or bar is None or pos.entry_time.date()>=at.date() or symbol in plans
                    or pos.metadata.get('capacity_rotation_date')==at.date().isoformat()):continue
            vwap=_running_vwap(tuple(s.session_bars))
            if config.capacity_rotation_filter=='below_vwap' and (vwap<=0 or bar.close>=vwap):continue
            if s.pending_exit_order_id and state.order_roles.get(s.pending_exit_order_id,{}).get('reason')!='next_close_exit':continue
            if any(a.symbol==symbol and isinstance(a,(SubmitExit,SubmitPartialExit)) and a.order_type=='MARKET' for a in actions):continue
            choices.append((bar.close/max(vwap,1e-9)-1,symbol))
        for _,symbol in sorted(choices):
            if need<=0:break
            s=state.symbols[symbol];bar=by_symbol[symbol]
            max_qty=floor(s.position.qty_open*config.capacity_rotation_fraction)
            if max_qty<=0:continue
            if s.pending_exit_order_id:
                net_price=bar.close*(1-(config.commission_bps+config.tax_bps_on_sell+config.slippage_bps)/10000)
                qty=min(max_qty,ceil(need/max(net_price,1e-9)))
                end=min(at+timedelta(minutes=10),*(i.expires_at for i in wanted))
                plans[symbol]={'qty':qty,'end':end.isoformat(),'cancel_order_id':s.pending_exit_order_id}
                actions.append(CancelOrders('OLR',symbol,'rotation_cancel_auction'))
                events.append(DecisionEvent(bar.timestamp,'OLR',symbol,'CAPITAL_ROTATION_CANCEL_REQUESTED',
                    'rotation_cancel_auction',metadata={'qty':qty,'expires_at':end.isoformat(),'cash_released':0}))
                need=max(0,need-qty*net_price);waiting=True
            else:submit(symbol,max_qty)
    return actions,events,waiting


def vars_plan(config):
    from dataclasses import asdict
    from .logic import _exit_plan
    return asdict(_exit_plan(config))

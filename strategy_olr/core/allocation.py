"""One completed timestamp cohort, native signals and shared cash allocation.

Optional rotation requests native cancellations; pending sales never fund a
proposal. The caller submits actions and remembers actual acknowledgements.
"""
from dataclasses import replace
from datetime import datetime, timedelta
from math import ceil, floor

from deployment.olr_kalcb.intent_allocator import (
    AllocationAccount, EntryIntent, IntentAllocationConfig, allocate_entry_intents)
from strategy_common.actions import SubmitEntry, SubmitExit, SubmitPartialExit, action_to_json_dict, action_from_json_dict
from strategy_common.events import DecisionEvent
from .core_models import OLRCoreResult
from .logic import step_olr_core, _pending_entry_reserved_notional
from .capacity import forecast


def step_olr_batch(state, bars, config, snapshot, portfolio, *, decision_context_provider=None, economic_policy=None):
    bars = tuple(bars)
    if not bars:
        return OLRCoreResult(state=state)
    stamp = bars[0].timestamp
    if any(b.timestamp != stamp or b.timeframe != '5m' or not b.is_completed for b in bars):
        raise ValueError('OLR allocation requires one completed five-minute timestamp')
    if len({b.symbol for b in bars}) != len(bars):
        raise ValueError('Duplicate symbol in OLR allocation cohort')
    at = stamp + timedelta(minutes=5)
    session_budget = None
    if config.capacity_session_budget_fraction:
        # Freeze the envelope before the first session decision. Restarts keep
        # it in the ordinary strategy checkpoint; intraday PnL cannot refill it.
        session_budget = state.meta.get('capacity_session_budget_v1')
        if session_budget is None or session_budget['day'] != stamp.date().isoformat():
            session_budget={'day':stamp.date().isoformat(),'equity':max(0,float(portfolio.equity))}
            state.meta['capacity_session_budget_v1']=session_budget
    actions, events, proposals = [], [], []
    view = replace(portfolio, external_entry_allocation=True,
        entry_proposal_only=economic_policy is not None,
        economic_size_envelope=bool(economic_policy is not None and economic_policy.size_envelope(at)))
    for bar in sorted(bars, key=lambda b:b.symbol):
        result = step_olr_core(state, bar, config, snapshot, view)
        for action in result.actions:
            (proposals if isinstance(action, SubmitEntry) else actions).append(action)
        events.extend(e for e in result.decisions if e.decision_code != 'ENTRY_SUBMITTED')
    if economic_policy is not None:
        from .opportunity import proposals as opportunity_proposals
        proposals=opportunity_proposals(state,bars,config,at,proposals)
        actions,management_events=economic_policy.on_book(state,bars,config,snapshot,portfolio,actions)
        events.extend(management_events)
    deferred = state.meta.setdefault('capacity_deferred_intents', {}) if config.capacity_defer_minutes else {}
    by_symbol = {b.symbol:b for b in bars}
    # Proposals have no broker reservation. Recheck current signal quality and
    # cash after real execution feedback; the original TTL never refreshes.
    for symbol, pending in list(deferred.items()):
        end = datetime.fromisoformat(pending['end'])
        s = state.symbol_state(symbol)
        if at > end or s.position is not None or s.pending_entry_order_id:
            deferred.pop(symbol)
            events.append(DecisionEvent(stamp,'OLR',symbol,'ENTRY_SKIPPED','deferred_signal_expired'))
            continue
        if symbol not in by_symbol:
            continue
        from .logic import _entry_plan, _market_entry_sizing_price
        from strategy_olr.execution import _common_entry_bar_passes, _running_vwap, _close_location, DECISION_CUTOFF
        current = by_symbol[symbol]
        plan = _entry_plan(config)
        if plan.mode not in ('confirm_next_bar', 'decision_next_open'):
            raise ValueError('Funding deferral only supports revalidated confirmation/next-open signals')
        prefix = tuple(s.session_bars)
        # Native entry evidence is retained, but the current completed bar must
        # still pass the same price/VWAP/close-location acceptance conditions.
        from strategy_common.execution_session import session_cutoff
        cutoff = session_cutoff(stamp.date(), DECISION_CUTOFF)
        prior = [b for b in prefix if b.timestamp.time().replace(tzinfo=None)<cutoff]
        valid = bool(prefix and prior) and _common_entry_bar_passes(current,_running_vwap(prefix),
            _close_location(current),float(prior[-1].close),plan)
        if not valid:
            deferred.pop(symbol)
            events.append(DecisionEvent(stamp,'OLR',symbol,'ENTRY_SKIPPED','deferred_signal_invalidated'))
            continue
        action = action_from_json_dict(pending['action'])
        meta = dict(action.metadata)
        price = max(float(meta['estimated_entry_price']),_market_entry_sizing_price(current,config))
        quantity = min(action.qty,int(float(meta['target_notional'])/(price*(1+meta['entry_cost_buffer_pct']))))
        cap,_,evidence = forecast(state,current,config)
        if cap is not None:quantity=min(quantity,cap)
        if quantity<=0:
            deferred.pop(symbol)
            continue
        meta.update(estimated_entry_price=price, capacity_forecast=evidence,
            allocation_deferred_origin=pending['origin'], allocation_deferred_until=pending['end'])
        proposals.append(replace(action,qty=quantity,metadata=meta))
    if not proposals and not config.capacity_rotation:
        return OLRCoreResult(state, actions, events)
    equity = max(float(portfolio.equity), 0)
    reserve = _pending_entry_reserved_notional(state)
    # Actual available cash is the funding bound; marked holdings are retained.
    occupied = frozenset(('OLR', sym) for sym, s in state.symbols.items()
        if s.position is not None or s.pending_entry_order_id)
    account = AllocationAccount(equity, equity, equity, equity,
        max(0, portfolio.cash-reserve), {'OLR':equity},
        gross_committed=max(0,portfolio.open_notional)+reserve,
        strategy_committed={'OLR':max(0,portfolio.open_notional)+reserve}, occupied=occupied)
    unrestricted_account = account
    budget_evidence = {}
    if session_budget is not None:
        consumed = session_acquisition_cost(state,stamp.date(),config)
        limit=session_budget['equity']*config.capacity_session_budget_fraction
        remaining=max(0,limit-consumed-reserve)
        account=replace(account,funding_limit=min(account.funding_limit,remaining))
        budget_evidence={'session_equity':session_budget['equity'],'limit':limit,
            'acquisition_cost':consumed,'pending_reserve':reserve,'remaining':remaining}
    intents = []
    by_id = {}
    for action in proposals:
        m = action.metadata
        unit = m['estimated_entry_price']*(1+m['entry_cost_buffer_pct'])
        key = action.symbol+':'+at.isoformat()
        end = datetime.fromisoformat(str(m['expiry_ts']))
        if end < at:
            continue
        intents.append(EntryIntent(key, 'OLR', action.symbol, m.get('sector','UNKNOWN'),
            action.qty, unit, at, end, native_rank=m['candidate_rank'],
            minimum_qty=ceil(equity*config.capacity_min_position_pct/unit)))
        by_id[key] = action
    rotation_waiting = False
    if config.capacity_rotation:
        from .capital_rotation import rotation_actions
        actions, rotation_events, rotation_waiting = rotation_actions(
            state,bars,config,intents,account,actions,at)
        auction_symbols={a.symbol for a in actions if isinstance(a,SubmitExit) and a.reason=='next_close_exit'}
        events=[e for e in events if not (e.decision_code=='EXIT_SUBMITTED'
            and e.reason=='next_close_exit' and e.symbol not in auction_symbols)]
        events.extend(rotation_events)
    # A closing-auction order cannot plausibly release cash in a five-minute
    # afternoon TTL. Only native market exits support this intervention.
    waiting_exit = rotation_waiting or any(isinstance(a,(SubmitExit,SubmitPartialExit)) and a.order_type=='MARKET' for a in actions) or any(
        s.pending_exit_order_id and state.order_roles.get(s.pending_exit_order_id,{}).get('reason')
        not in (None, '', 'next_close_exit') for s in state.symbols.values())
    # Reallocate the whole current cohort after removing an unfunded proposal;
    # deferred signals have neither physical nor phantom budget reservations.
    while True:
        from .loss_budget import allocate as allocate_with_loss_budget
        decisions = allocate_with_loss_budget(intents, account, at, config, state, by_id)
        remove = None
        if config.capacity_defer_minutes and waiting_exit:
            for decision in decisions:
                action = by_id[decision.intent_id]
                if config.capacity_rotation and action.metadata['candidate_rank']>config.capacity_rotation_max_rank:
                    continue
                pending = deferred.get(action.symbol)
                end = datetime.fromisoformat(pending['end']) if pending else min(
                    at+timedelta(minutes=config.capacity_defer_minutes),datetime.fromisoformat(str(action.metadata['expiry_ts'])))
                if (at<end and decision.binding_limit in ('funding','gross','strategy_budget')
                        and decision.approved_qty<action.qty*config.capacity_defer_min_fraction):
                    if pending is None:
                        deferred[action.symbol]={'action':action_to_json_dict(action), 'origin':at.isoformat(),'end':end.isoformat()}
                    events.append(DecisionEvent(stamp,'OLR',action.symbol,'ENTRY_DEFERRED','await_confirmed_exit_funding',
                        metadata={'desired_qty':action.qty,'currently_affordable_qty':decision.approved_qty,
                            'expires_at':end.isoformat(),'reserved_notional':0.0}))
                    remove=decision.intent_id
                    break
        if remove is None:
            break
        intents=[i for i in intents if i.intent_id!=remove]
    native_approved = {d.intent_id: d.approved_qty for d in decisions}
    unbudgeted = ({d.intent_id:d.approved_qty for d in allocate_entry_intents(
        intents,unrestricted_account,at,IntentAllocationConfig())} if session_budget is not None else {})
    unconditioned = {}
    broad_only=(paired_quantities(intents,by_id,'capacity_broad_only_qty',account,at)
        if config.capacity_history_scope in ('conditional_shrink','conditional_exit') else {})
    unvalued = (paired_quantities(intents,by_id,'capacity_without_value_qty',account,at)
        if config.capacity_value_mode!='off' else {})
    if config.capacity_volume_exponent:
        # Same proposals, account and timestamp; change ONLY the conditioned
        # quantity bound. This measures immediate feasible quantity support,
        # not a counterfactual return or an independent trading opportunity.
        unconditioned = paired_quantities(intents,by_id,'capacity_unconditioned_qty',account,at)
    recycling_before=None
    use_redistribution=config.capacity_redistribute and not view.economic_size_envelope
    if use_redistribution:
        used = sum(d.reserved_notional for d in decisions)
        residual = replace(account, funding_limit=max(0, account.funding_limit-used),
            gross_committed=account.gross_committed+used,
            strategy_committed={'OLR': account.strategy_committed['OLR']+used})
        recycling = None
        if config.capacity_recycle_liquidity_only:
            recycling=state.meta.get('capacity_recycling_v1')
            if recycling is None or recycling['day']!=stamp.date().isoformat():
                recycling={'day':stamp.date().isoformat(),'displaced_request_funding':0.,'assigned_extras':0.}
                state.meta['capacity_recycling_v1']=recycling
            uncapped=paired_quantities(intents,by_id,'capacity_native_qty',account,at)
            original_used=sum(uncapped[i.intent_id]*i.unit_cost for i in intents)
            # Risk-clipped quantities are not displaced liquidity demand.
            # Keep the existing liquidity-only top-up pool's meaning intact.
            from .loss_budget import enabled as loss_budget_enabled
            liquidity_used = (sum(d.reserved_notional for d in allocate_entry_intents(
                intents, account, at, IntentAllocationConfig())) if loss_budget_enabled(config) else used)
            # This is a request-sizing comparison, not proceeds from a sale or
            # a promise that the uncapped order could physically fill. Extra
            # orders still consume the actual account's remaining cash.
            recycling['displaced_request_funding']+=max(0,original_used-liquidity_used)
            residual=replace(residual,funding_limit=min(residual.funding_limit,
                max(0,recycling['displaced_request_funding']-recycling['assigned_extras'])))
        extras = []
        intents_by_id = {i.intent_id: i for i in intents}
        for decision in decisions:
            intent = intents_by_id[decision.intent_id]
            if not decision.approved_qty or intent.native_rank > config.capacity_redistribute_max_rank:
                continue
            action = by_id[decision.intent_id]
            cap, exit_cap, evidence = forecast(state, by_symbol[action.symbol], config)
            # Require the declared liquidity evidence, admit no new names,
            # and preserve every native approved allocation first.
            if config.capacity_redistribution_evidence=='auction' and exit_cap is None:
                continue
            if config.capacity_redistribution_evidence=='entry' and not evidence.get('entry',{}).get('samples'):
                continue
            upper = floor(equity*config.max_position_pct/intent.unit_cost)
            if exit_cap is not None:
                upper=min(upper,exit_cap)
            if cap is not None:
                upper = min(upper, cap)
            if config.capacity_redistribution_evidence=='entry':
                # Feasibility can constrain extras without changing the
                # incumbent median policy's funded native quantities.
                upper=min(upper,evidence['entry']['quantity'])
            if config.capacity_redistribution_max_multiplier:
                upper=min(upper,floor(action.qty*config.capacity_redistribution_max_multiplier))
            extra = max(0, upper-decision.approved_qty)
            if extra:
                extras.append(replace(intent, desired_qty=extra, minimum_qty=0))
        increments = {d.intent_id: d for d in allocate_with_loss_budget(
            extras, residual, at, config, state, by_id, prior=decisions)}
        if recycling is not None:
            # Conservatively retain this session's assigned envelope even if
            # an extra order later expires; no anticipated cancellation credit.
            recycling_before=recycling['assigned_extras']
            recycling['assigned_extras']+=sum(d.reserved_notional for d in increments.values())
        decisions = [replace(d, approved_qty=d.approved_qty+increments[d.intent_id].approved_qty,
            reserved_notional=d.reserved_notional+increments[d.intent_id].reserved_notional)
            if d.intent_id in increments else d for d in decisions]
    # Research-only, exact-account response context. With no installed context
    # this preserves the original native decisions and every execution path.
    pre_joint_quantities=None
    if decision_context_provider is not None and intents:
        from copy import deepcopy
        from dataclasses import asdict
        from .joint_allocation import KEY, account_identity
        from .serializers import snapshot_state
        if KEY in state.meta:
            raise ValueError('Two competing joint decision context providers')
        # The provider observes only this completed cohort and acknowledged
        # state. Copies prevent research code from changing the native ledger.
        request=deepcopy(dict(at=at.isoformat(), account=asdict(account),
            account_identity=account_identity(account,intents,at),
            intents=[asdict(i) for i in intents], native_decisions=[asdict(d) for d in decisions],
            proposals={k:action_to_json_dict(v) for k,v in by_id.items()},
            state=snapshot_state(state)))
        context=decision_context_provider(request)
        if context is not None:
            state.meta[KEY]=deepcopy(context)
    if 'joint_decision_context_v1' in state.meta:
        from .joint_allocation import allocate as allocate_joint_response
        pre_joint_quantities={d.intent_id:d.approved_qty for d in decisions}
        decisions = allocate_joint_response(intents, decisions, account, at, config, state, by_id)
        if recycling_before is not None:
            # Only the final selected increments consume the session envelope.
            units={i.intent_id:i.unit_cost for i in intents}
            recycling['assigned_extras']=recycling_before+sum(
                max(0,d.approved_qty-native_approved[d.intent_id])*units[d.intent_id] for d in decisions)
        if economic_policy is not None and hasattr(economic_policy,'record_allocation'):
            economic_policy.record_allocation(state.meta['joint_decision_receipt_v1'])
    for decision in decisions:
        action = by_id[decision.intent_id]
        deferred.pop(action.symbol,None)
        meta = {**dict(action.metadata), 'joint_allocation':True,
            'allocation_native_qty':action.qty, 'allocation_binding_limit':decision.binding_limit,
            'reserved_notional':decision.reserved_notional, 'allocation_reason':decision.reason}
        joint_receipt=state.meta.get('joint_decision_receipt_v1')
        if joint_receipt and joint_receipt['at']==at.isoformat():
            meta['joint_decision_response']={k:joint_receipt[k] for k in ('version','at','context_sha256','account_identity','changed')}
        if pre_joint_quantities is not None:
            meta.update(allocation_before_joint_qty=pre_joint_quantities[decision.intent_id],
                        allocation_joint_qty_delta=decision.approved_qty-pre_joint_quantities[decision.intent_id])
        from .loss_budget import enabled as loss_budget_enabled, unit_loss
        if loss_budget_enabled(config):
            meta['scenario_loss_budget'] = {'unit_loss': unit_loss(action.metadata, config),
                'approved_loss': decision.approved_qty * unit_loss(action.metadata, config),
                'equity': equity, 'position_pct': config.capacity_loss_position_pct,
                'sector_pct': config.capacity_loss_sector_pct, 'total_pct': config.capacity_loss_total_pct,
                'atr_multiple': config.capacity_loss_atr_multiple}
        if use_redistribution:
            meta.update(allocation_before_redistribution=native_approved[decision.intent_id],
                allocation_redistribution_evidence=config.capacity_redistribution_evidence,
                allocation_redistributed_qty=(pre_joint_quantities[decision.intent_id] if pre_joint_quantities is not None else decision.approved_qty)-native_approved[decision.intent_id])
            if config.capacity_recycle_liquidity_only:
                meta['capacity_recycling']=dict(state.meta['capacity_recycling_v1'])
        if unconditioned:
            meta.update(capacity_unconditioned_approved_qty=unconditioned[decision.intent_id],
                capacity_conditioned_qty_delta=native_approved[decision.intent_id]-unconditioned[decision.intent_id])
        if unbudgeted:
            meta.update(capacity_session_budget=budget_evidence,
                capacity_without_budget_approved_qty=unbudgeted[decision.intent_id],
                capacity_budget_qty_delta=native_approved[decision.intent_id]-unbudgeted[decision.intent_id])
        if unvalued:
            meta.update(capacity_without_value_approved_qty=unvalued[decision.intent_id],
                capacity_value_qty_delta=native_approved[decision.intent_id]-unvalued[decision.intent_id])
        if broad_only:
            meta.update(capacity_broad_only_approved_qty=broad_only[decision.intent_id],
                capacity_history_qty_delta=native_approved[decision.intent_id]-broad_only[decision.intent_id])
        if decision.approved_qty:
            if economic_policy is not None:
                from .opportunity import commit
                commit(state,action.symbol)
            actions.append(replace(action, qty=decision.approved_qty, metadata=meta))
        events.append(DecisionEvent(stamp, 'OLR', action.symbol,
            'ENTRY_SUBMITTED' if decision.approved_qty else 'ENTRY_SKIPPED',
            action.reason if decision.approved_qty else decision.reason,
            metadata={**meta, 'qty':decision.approved_qty}))
    return OLRCoreResult(state, actions, events)


def paired_quantities(intents, by_id, field, account, at):
    quantities={i.intent_id:by_id[i.intent_id].metadata.get(field,i.desired_qty) for i in intents}
    result={i.intent_id:0 for i in intents}
    result.update({d.intent_id:d.approved_qty for d in allocate_entry_intents([
        replace(i,desired_qty=quantities[i.intent_id]) for i in intents if quantities[i.intent_id]>0],
        account,at,IntentAllocationConfig())})
    return result


def session_acquisition_cost(state, day, config):
    """Filled acquisition cost for this vintage, including shares already sold."""
    consumed = 0.0
    for s in state.symbols.values():
        # A lifecycle may appear both in history and the active position
        # after a late fill. Count its most recent state exactly once.
        positions = dict(s.entry_history)
        if s.position is not None:
            p=s.position
            positions[p.metadata.get('entry_lifecycle_key',p.entry_order_id)]=p
        for p in positions.values():
            if p.entry_time.date()==day:
                if 'capacity_acquired_cost' not in p.metadata:
                    raise ValueError('Session budget requires restored native acquisition cash debits')
                consumed+=float(p.metadata['capacity_acquired_cost'])
    return consumed

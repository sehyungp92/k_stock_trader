"""Explicit research bridge from a causal response surface to native intents.

Disabled unless a complete, source-bound context for this exact account/cohort
is installed. Default behaviour returns the actual native decisions unchanged.
The bridge cannot release sale proceeds, alter fills, or waive native budgets.
"""
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from math import isclose, isfinite

from .joint_decision import Account, Action, Constraints, ScenarioAxis, decide

KEY = 'joint_decision_context_v1'
RECEIPT = 'joint_decision_receipt_v1'


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str,allow_nan=False).encode()).hexdigest()


def account_identity(account, intents, at):
    return signature(dict(account=asdict(account), intents=[asdict(i) for i in sorted(intents,key=lambda i:i.intent_id)],at=at.isoformat()))


def allocate(intents, native_decisions, account, at, config, state, by_id):
    context = state.meta.get(KEY)
    if context is None:
        return native_decisions
    if not intents:
        raise ValueError('Joint context without an allocation cohort')
    if not isfinite(context['meaningful_notional']) or context['meaningful_notional']<=0:
        raise ValueError('A positive meaningful acquisition threshold is required')
    if context['at'] != at.isoformat() or context['account_identity'] != account_identity(account,intents,at):
        raise ValueError('Stale or mismatched joint account/cohort context')
    if len({i.symbol for i in intents}) != len(intents):
        raise ValueError('One native proposal per symbol required')
    by_intent={i.intent_id:i for i in intents}
    if set(context['actions']) != set(by_intent):
        raise ValueError('Missing or additional action group')
    native={d.intent_id:d for d in native_decisions}
    if set(native) != set(by_intent):
        raise ValueError('Native comparison cohort mismatch')
    from .loss_budget import allocate as native_allocate, committed, enabled, unit_loss
    used_symbols,used_sectors=committed(state,config) if enabled(config) else ({},{})
    limits={'gross':max(0.,account.gross_limit-account.gross_committed)}
    groups={};action_quantities={}
    axis=ScenarioAxis(tuple(context['scenario_ids']),tuple(context['probabilities']),tuple(context['offsets']),
                      datetime.fromisoformat(context['known_at']),context['source_sha256'])
    for key,intent in by_intent.items():
        eligible=(intent.available_at<=at<=intent.expires_at and (intent.strategy_id,intent.symbol) not in account.occupied
                  and not (account.block_duplicates and account.symbol_committed.get(intent.symbol,0)>0))
        # This bridge has no authority to alter occupied-slot semantics. A
        # proposal rejected for structural eligibility remains ineligible.
        if native[key].reason in ('duplicate_symbol','own_position_or_order','expired_intent','not_yet_available'):
            eligible=False
        cap=max(intent.desired_qty,native[key].approved_qty)
        if intent.capacity_qty is not None:cap=min(cap,intent.capacity_qty)
        loss=unit_loss(by_id[key].metadata,config) if enabled(config) else 0.
        dimensions=[('gross',intent.unit_cost),('strategy:'+intent.strategy_id,intent.unit_cost),
                    ('symbol:'+intent.symbol,intent.unit_cost),('sector:'+intent.sector,intent.unit_cost)]
        limits['strategy:'+intent.strategy_id]=max(0.,account.strategy_limits[intent.strategy_id]-account.strategy_committed.get(intent.strategy_id,0))
        limits['symbol:'+intent.symbol]=max(0.,account.symbol_limit-account.symbol_committed.get(intent.symbol,0))
        limits['sector:'+intent.sector]=max(0.,account.sector_limit-account.sector_committed.get(intent.sector,0))
        if intent.shared_entry_capacity_qty is not None:
            name='physical:'+intent.symbol;limits[name]=intent.shared_entry_capacity_qty;dimensions.append((name,1.))
        slot_names=[]
        if intent.strategy_id in account.max_positions:
            name='slots:'+intent.strategy_id
            limits[name]=max(0,account.max_positions[intent.strategy_id]-sum(s==intent.strategy_id for s,_ in account.occupied));slot_names.append(name)
        if intent.strategy_id in account.max_per_sector:
            name='sector_slots:'+intent.strategy_id+':'+intent.sector
            limits[name]=max(0,account.max_per_sector[intent.strategy_id]-account.sector_slots.get((intent.strategy_id,intent.sector),0));slot_names.append(name)
        for name,fraction,used in [('position_risk:'+intent.symbol,config.capacity_loss_position_pct,used_symbols.get(intent.symbol,0)),
                                   ('sector_risk:'+intent.sector,config.capacity_loss_sector_pct,used_sectors.get(intent.sector,0)),
                                   ('account_risk',config.capacity_loss_total_pct,sum(used_symbols.values()))]:
            if fraction:
                limits[name]=max(0.,account.equity*fraction-used);dimensions.append((name,loss))
        choices=[]
        for spec in context['actions'][key]:
            q=spec['quantity']
            if type(q) is not int or q<0 or q>cap or q and (not eligible or q<intent.minimum_qty):
                raise ValueError('Response action violates native eligibility or quantity envelope')
            if not isclose(spec['reserve_cash'],q*intent.unit_cost,rel_tol=1e-12,abs_tol=1e-8):
                raise ValueError('Action cash reservation disagrees with native buffered price')
            probabilities=tuple(spec['completed_entry_probabilities'])
            if q*intent.unit_cost<context['meaningful_notional'] and any(probabilities):
                raise ValueError('An undersized request cannot predict a meaningful completed acquisition')
            a=Action(spec['action_id'],key,q,q*intent.unit_cost,tuple(tuple(x) for x in spec['net_paths']),axis.ids,
                     datetime.fromisoformat(spec['known_at']),spec['source_sha256'],
                     tuple((name,unit*q) for name,unit in dimensions)+tuple((name,int(q>0)) for name in slot_names),
                     probabilities,spec['supported'])
            choices.append(a);action_quantities[a.action_id]=q
        groups[key]=tuple(choices)
    cfg=Constraints(**context['constraints'])
    solver_account=Account(context.get('actual_equity',account.equity),context['actual_high_water'],
                           account.funding_limit,0,tuple(sorted(limits.items())))
    result=decide(solver_account,axis,context['held_net_paths'],groups,at,cfg)
    # A hash alone cannot explain which resource excluded a profitable dose.
    # Persist the exact finite decision problem, independently replayable after
    # the native process exits. No mutable strategy history is copied here.
    inputs=json.loads(json.dumps(dict(account=asdict(solver_account),axis=asdict(axis),
        held_net_paths=context['held_net_paths'],groups={k:[asdict(a) for a in v] for k,v in groups.items()},
        constraints=context['constraints']),default=lambda v:v.isoformat(),allow_nan=False))
    receipt=dict(version='joint-decision-v1',at=at.isoformat(),context_sha256=signature(context),account_identity=context['account_identity'],
                 outcome=asdict(result),native_quantities={k:d.approved_qty for k,d in native.items()},changed=False,
                 solver_inputs=inputs,solver_inputs_sha256=signature(inputs),
                 interpretation='Forecast scenario frontier; not realized portfolio performance or a fill promise.')
    # No context preserves native behavior. An installed but infeasible risk
    # context must never silently restore purchases that failed its constraints.
    if result.selected is None:
        receipt['fallback']='no_new_risk_infeasible';state.meta[RECEIPT]=receipt;state.meta.pop(KEY)
        return [replace(d,approved_qty=0,reserved_notional=0.,reason='joint_risk_infeasible',
                        binding_limit='joint_response_surface') for d in native_decisions]
    chosen={key:action_quantities[aid] for key,aid in zip(sorted(groups),result.selected.action_ids)}
    positive=[replace(i,desired_qty=chosen[i.intent_id]) for i in intents if chosen[i.intent_id]>0]
    checked={d.intent_id:d for d in native_allocate(positive,account,at,config,state,by_id)}
    if any(checked[key].approved_qty!=q for key,q in chosen.items() if q):
        raise ValueError('Joint choice failed independent native funding/risk allocation')
    decisions=[]
    for original in native_decisions:
        q=chosen[original.intent_id]
        decisions.append(replace(original,approved_qty=q,reserved_notional=q*by_intent[original.intent_id].unit_cost,
                         reason='joint_supported_choice' if q else 'joint_no_order',rank_basis='joint_net_value_and_tail',
                         binding_limit='joint_response_surface'))
    receipt.update(selected_quantities=chosen,changed=any(chosen[k]!=native[k].approved_qty for k in chosen))
    state.meta[RECEIPT]=receipt;state.meta.pop(KEY)
    return decisions


def replay_receipt(receipt):
    """Reproduce a saved forecast decision, never a counterfactual broker fill."""
    raw=receipt['solver_inputs']
    if signature(raw)!=receipt['solver_inputs_sha256']:
        raise ValueError('Changed solver receipt inputs')
    a=raw['axis'];axis=ScenarioAxis(tuple(a['ids']),tuple(a['weights']),tuple(a['offsets']),
        datetime.fromisoformat(a['known_at']),a['source_sha256'])
    groups={}
    for name,rows in raw['groups'].items():
        options=[]
        for row in rows:
            row=dict(row);row['available_at']=datetime.fromisoformat(row['available_at'])
            for key in ('net_paths','resources'):row[key]=tuple(tuple(x) for x in row[key])
            for key in ('scenario_ids','completed_entry_probabilities','management_completion_deltas'):
                row[key]=tuple(row[key])
            options.append(Action(**row))
        groups[name]=tuple(options)
    result=decide(Account(**raw['account']),axis,raw['held_net_paths'],groups,
                  datetime.fromisoformat(receipt['at']),Constraints(**raw['constraints']))
    if signature(asdict(result))!=signature(receipt['outcome']):
        raise ValueError('Saved solver result does not reproduce')
    return result

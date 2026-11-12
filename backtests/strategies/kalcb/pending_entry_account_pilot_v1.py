"""Finite same-account native cancellation pilot; no fitted policy or live use."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from itertools import groupby
from pathlib import Path
from collections.abc import Mapping
import gzip
import hashlib
import json

from strategy_kalcb.core import pending_entry_revalidation_v1 as action
from backtests.strategies.kalcb.mechanism_observer_v2 import clone_broker
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound


def neutral(value):
    """Include every account/adapter attribute, failing on unknown state types."""
    if value is None or isinstance(value, (str, int, float, bool)): return value
    if isinstance(value, (datetime, date, time)): return value.isoformat()
    if isinstance(value, timedelta): return {'timedelta':str(value)}
    if isinstance(value, Enum): return [type(value).__name__, value.value]
    if isinstance(value, Mapping):
        return sorted([[neutral(k), neutral(v)] for k,v in value.items()], key=lambda x: canonical(x[0]))
    if isinstance(value, (list, tuple)): return [neutral(x) for x in value]
    if isinstance(value, (set, frozenset)): return sorted([neutral(x) for x in value], key=canonical)
    if is_dataclass(value): return {f.name:neutral(getattr(value,f.name)) for f in fields(value)}
    if hasattr(value, 'hexdigest') and hasattr(value, 'copy'): return {'sha256_state':value.hexdigest()}
    if hasattr(value, '__dict__'): return {'type':type(value).__qualname__, 'attributes':neutral(vars(value))}
    raise TypeError('Unbound state type '+str(type(value)))


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',',':'), allow_nan=False)


def state_receipt(adapter, broker):
    value={'adapter':neutral(vars(adapter)), 'broker':neutral(vars(broker))}
    return {'sha256':hashlib.sha256(canonical(value).encode()).hexdigest(), 'state':value}


def clone_account(adapter, broker):
    # Copy ALL adapter attributes: no hand-maintained mutable-field whitelist.
    if adapter.frontier_shadow is not None: raise ValueError('Unsupported frontier shadow')
    return deepcopy(adapter), clone_broker(broker)


def pending_symbols(broker):
    return list(dict.fromkeys(o.symbol for o in broker.orders if o.strategy_id=='KALCB' and o.side=='BUY'))


def eligibility(adapter, broker, timestamp):
    # Only predecision state is consulted, never future fills or outcomes.
    state=deepcopy(adapter.state); state.meta[action.KEY+'_enabled']=True
    orders=[asdict(o) for o in broker.orders]; results=[]
    for symbol in pending_symbols(broker):
        actions, reason=action.revalidate(state, symbol, timestamp+timedelta(minutes=5), orders)
        results.append({'symbol':symbol, 'reason':reason, 'actions':len(actions)})
    return results


def fork_pair(adapter, broker, timestamp):
    before=state_receipt(adapter,broker)
    arms={name:clone_account(adapter,broker) for name in ('hold','cancel')}
    for aa,bb in arms.values():
        if state_receipt(aa,bb)['sha256']!=before['sha256']: raise ValueError('Clone lost account state')
    selected=[]; records=[]
    for name,(aa,bb) in arms.items():
        old_clock=bb._decision_available_at
        bb._decision_available_at=timestamp+timedelta(minutes=5)
        if name=='cancel': aa.state.meta[action.KEY+'_enabled']=True
        for symbol in pending_symbols(bb):
            actions,reason=action.revalidate(aa.state,symbol,timestamp+timedelta(minutes=5),[asdict(o) for o in bb.orders])
            if actions:
                selected.extend(a.metadata['entry_order_id'] for a in actions)
                aa._submit_actions(actions,bb,timestamp)
            if name=='cancel':records.append({'symbol':symbol,'reason':reason,'actions':len(actions)})
        bb._decision_available_at=old_clock
    if not selected: raise ValueError('Selected state produced no cancellation')
    if state_receipt(adapter,broker)['sha256']!=before['sha256']: raise ValueError('Branch mutated parent')
    if state_receipt(*arms['hold'])['sha256']!=before['sha256']: raise ValueError('Disabled helper changed state')
    remaining={o.order_id for o in arms['cancel'][1].orders}
    if remaining!={o.order_id for o in broker.orders}-set(selected): raise ValueError('Cancellation affected another order')
    if arms['cancel'][1].positions!=broker.positions or arms['cancel'][1].cash!=broker.cash:
        raise ValueError('Cancellation liquidated or changed filled inventory')
    return {'arms':arms,'before':before,'orders':selected,'records':records,
            'clock':(timestamp+timedelta(minutes=5)).isoformat(),'day':str(timestamp.date())}


def account_detail(adapter,broker,indexes):
    from backtests.engine.integrity import audit_broker
    from backtests.strategies.kalcb.checkpoint_contrast_labels_v1 import independent_account
    from scripts.kalcb_executed_value_replay import equity_records
    adapter._sync_new_fills(broker)
    audit=audit_broker(broker)
    if not audit['execution_validity_pass']:raise ValueError('Native execution audit failed')
    checked=0
    for batch in broker.expected_batches:
        for bar in batch:
            expected=indexes[bar['timestamp'][:10]].get((bar['symbol'],bar['timestamp'],bar['timeframe']))
            if expected!=tuple(bar[k] for k in ('open','high','low','close','volume','is_completed')):
                raise ValueError('Consumed raw source mismatch')
            checked+=1
    independent=independent_account(broker,indexes)
    return {'fills':[asdict(x) for x in broker.fills], 'journal':broker.journal,
            'equity':equity_records(broker.timestamps,broker.equity_curve),
            'cash':broker.cash,'positions':[asdict(x) for x in broker.positions.values()],
            'pending':[asdict(x) for x in broker.orders], 'costs':asdict(broker.costs),
            'engine_audit':audit,'independent_account':independent,'raw_bars':checked,
            'state_sha256':state_receipt(adapter,broker)['sha256']}


def save_pair(root,scenario,ordinal,pair,parent,indexes):
    # Disabled continuation must reproduce the untouched concurrent parent,
    # including non-economic cursors, risk state and complete order history.
    if state_receipt(*pair['arms']['hold'])['sha256']!=state_receipt(*parent)['sha256']:
        raise ValueError('Disabled native continuation differs from parent')
    folder=root/'pairs'/scenario/f'{ordinal:02d}_{pair["day"]}';folder.mkdir(parents=True,exist_ok=False)
    with gzip.open(folder/'prestate.json.gz','xt',encoding='utf-8',compresslevel=1) as f:
        json.dump(pair['before'],f,separators=(',',':'),allow_nan=False)
    details={}
    for name,(aa,bb) in pair['arms'].items():
        details[name]=account_detail(aa,bb,indexes)
        with gzip.open(folder/(name+'.json.gz'),'xt',encoding='utf-8',compresslevel=1) as f:
            json.dump(details[name],f,default=str,separators=(',',':'),allow_nan=False)
    clock=datetime.fromisoformat(pair['clock']);ids=set(pair['orders'])
    later={name:[f for f in d['fills'] if f['order_id'] in ids and f['side']=='BUY' and f['timestamp']>=clock]
           for name,d in details.items()}
    if later['cancel']:raise ValueError('Acknowledged cancellation filled again in native simulation')
    result={'day':pair['day'],'scenario':scenario,'ordinal':ordinal,'clock':pair['clock'],
        'prestate_sha256':pair['before']['sha256'],'exact_disabled_continuation':True,
        'cancellation_records':pair['records'],'cancelled_orders':pair['orders'],
        'original_later_buy_fills':len(later['hold']), 'original_later_buy_qty':sum(f['qty'] for f in later['hold']),
        'original_later_buy_notional':sum(f['qty']*f['price'] for f in later['hold']),
        'account_cash_difference_krw':details['cancel']['cash']-details['hold']['cash'],
        'scope':'Same actual funded prestate; whole-account session suffix difference. Includes other-order/funding/management descendants. These dependent contrasts are not a sequential policy return or independent observations.',
        'full_portfolio_audit_status':{n:d['engine_audit']['audit_status'] for n,d in details.items()},
        'bindings':[bound(folder/f) for f in ('prestate.json.gz','hold.json.gz','cancel.json.gz')], 'study_complete':False}
    write(folder/'pair.json',result)
    return result


def run(root,scenario):
    from backtests.engine.capacity import BarParticipationScenario
    from backtests.engine.replay import drive_timestamp_batch
    from backtests.engine.sim_broker import SimBroker,BrokerCosts
    from backtests.strategies.kalcb import executed_value as ev
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter
    from backtests.strategies.kalcb.checkpoint_contrast_labels_v1 import SourceIndexes
    from scripts.kalcb_executed_value_study import load_plan,config_for,connect,load_day
    from scripts.kalcb_executed_value_replay import verify_policy,timed_policy,observation_prefix,validate_watch_universe,equity_records
    import yaml
    p=read(root/'protocol.json'); profile=read(p['profile']['path']); plan=load_plan(Path(profile['plan']['path']))
    policies=Path(profile['policy']['path']).parent
    manifest,snapshots=verify_policy(policies,'interaction_cost_margin');updates=timed_policy(policies,manifest,snapshots)
    declared=tuple(str(s).zfill(6) for s in yaml.safe_load(Path(plan['universe']).read_text())['symbols'])
    validate_watch_universe(manifest.get('observation_symbols',()),declared)
    cfg=config_for(plan,scenario);participation,slippage,scenario_name=ev.SCENARIOS[scenario]
    costs=BrokerCosts(cfg.commission_bps,cfg.tax_bps_on_sell,slippage)
    broker=SimBroker(ev.REFERENCE_EQUITY,costs,buying_power_leverage=cfg.intraday_leverage,
                     capacity_policy=BarParticipationScenario(participation,scenario_name))
    adapter=KALCBReplayAdapter(cfg,snapshots,initial_equity=ev.REFERENCE_EQUITY,costs=costs,
        timed_snapshots=updates,observation_symbols=tuple(manifest.get('observation_symbols',())))
    conn=connect(plan);source=plan['input_sha256'][plan['database']]
    days=[r[0] for r in conn.execute('SELECT DISTINCT day FROM bars WHERE day BETWEEN ? AND ? ORDER BY day',
                                  (manifest['evaluation_from'],manifest['evaluation_through']))]
    if {d.isoformat() for d in snapshots}!=set(days) or len(days)!=156:raise ValueError('Changed parent calendar')
    indexes=SourceIndexes(p['databases']);out=root/'parents'/scenario;out.mkdir(parents=True,exist_ok=False)
    old=Path(p['parent_accounts'][scenario]);oldfills=read(old/'fills.json');oldequity=read(old/'equity.json');olddaily=read(old/'daily.json')
    results=[];census=[];daily=[]
    try:
        for day in days:
            d=date.fromisoformat(day);symbols={c.symbol for c in snapshots[d].candidates}
            for update in updates.get(d,()):symbols.update(c.symbol for c in update.candidates)
            symbols.update(s for _,s in broker.positions);symbols.update(o.symbol for o in broker.orders)
            if not symbols:symbols={'000100'}
            execution_symbols=set(symbols);symbols.update(manifest.get('observation_symbols',()))
            bars=load_day(conn,day,source,symbols)
            if manifest.get('observation_minutes'):bars=observation_prefix(bars,d,execution_symbols,manifest['observation_minutes'])
            active=None
            for timestamp,group in groupby(bars,key=lambda b:b.timestamp):
                batch=tuple(group)
                drive_timestamp_batch(broker,batch,(adapter,))
                if active:
                    for aa,bb in active['arms'].values():drive_timestamp_batch(bb,batch,(aa,))
                if (timestamp.hour,timestamp.minute)==(9,30):
                    observations=eligibility(adapter,broker,timestamp)
                    census.append({'day':day,'observations':observations})
                    if any(x['reason']=='cancel_requested' for x in observations):
                        active=fork_pair(adapter,broker,timestamp)
            daily.append({'day':day,'equity':broker.equity_curve[-1],
                'inventory':{str(k):v.qty for k,v in broker.positions.items()},'pending':len(broker.orders)})
            # Exact original full-account source-prefix identity before scoring any pair.
            actual=json.loads(json.dumps([asdict(f) for f in broker.fills],default=str))
            if actual!=[f for f in oldfills if f['timestamp'][:10]<=day]:raise ValueError('Original parent fill prefix differs')
            if [list(r) for r in equity_records(broker.timestamps,broker.equity_curve)]!=[r for r in oldequity if r[0][:10]<=day]:
                raise ValueError('Original parent MTM prefix differs')
            if daily!=[r for r in olddaily if r['day']<=day]:raise ValueError('Original parent balance prefix differs')
            if active:results.append(save_pair(root,scenario,len(results)+1,active,(adapter,broker),indexes))
            write(root/'progress.json',{'scenario':scenario,'through':day,'paired_states':len(results),'parent_sessions':len(daily)})
            print(json.dumps({'scenario':scenario,'day':day,'pairs':len(results),'parent_bridge':True}),flush=True)
            if len(results)==p['states_per_scenario']:break
        if len(results)!=p['states_per_scenario']:raise ValueError('Finite IS window lacks declared states; do not broaden')
        write(out/'census.json',census);write(out/'daily.json',daily)
        write(out/'account.json',account_detail(adapter,broker,indexes))
        write(out/'completion.json',{'complete':True,'scenario':scenario,'pairs':len(results),'source_prefix_sessions':len(daily),
            'original_parent_fill_balance_mtm_prefix_exact':True,'full_period_replay':False,'study_complete':False,
            'pairs_bound':[bound(root/'pairs'/scenario/f'{i+1:02d}_{r["day"]}'/'pair.json') for i,r in enumerate(results)]})
    finally:conn.close()

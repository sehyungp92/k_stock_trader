"""One completed-minute45 HOLD/LIQUIDATE contrast; isolated native labels only.

The parent owns every transition before the checkpoint. This registered research
hook emits ordinary shared-core actions after the parent bar and timer decisions.
No closed-parent exit is displaced; neither arm uses a hand-priced liquidation.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timedelta
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from unittest.mock import patch

from backtests.strategies.kalcb import executed_value as ev
from backtests.strategies.kalcb.late_entry_v1 import continuous_connection
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound, verify

REASON='research_completed45_liquidation_contrast_v1'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),default=str,allow_nan=False)


def checkpoint_actions(state,symbol):
    """Shared-core state/actions, usable by a future research policy unchanged."""
    from strategy_common.actions import CancelOrders, SubmitExit
    from strategy_kalcb.core.logic import _cancel_entry_remainder, _exit_metadata
    ss=state.symbols.get(symbol)
    pos=ss.position if ss else None
    if pos is None or pos.qty_open<=0:return [],'no_open_position'
    if pos.metadata.get('entry_route')!='first30_open_anchor':return [],'not_original_opening_route'
    if pos.exit_in_flight or pos.metadata.get('liquidation_reason'):
        return [],'parent_complete_exit'
    # Cancellation is symbol-wide in both native adapters. Keep the original
    # late-fill role before cancellation acknowledgement, then submit one exit.
    cancellations=_cancel_entry_remainder(state,ss,REASON)
    if not cancellations:cancellations=[CancelOrders('KALCB',symbol,REASON,{})]
    pos.metadata['liquidation_reason']=REASON
    pos.exit_in_flight=True
    return [*cancellations,SubmitExit('KALCB',symbol,pos.qty_open,'MARKET',None,
        REASON,_exit_metadata(pos,REASON,{'order_role':'EXIT','research_contrast_clock':45}))],'eligible'


def state_before(adapter,broker,symbol,clock):
    from strategy_kalcb.core.serializers import snapshot_state
    ss=adapter.state.symbols.get(symbol);pos=ss.position if ss else None
    positions=[asdict(p) for p in broker.positions.values()]
    state={'clock':clock.isoformat(),'core':snapshot_state(adapter.state),
        'cash':broker.cash,'positions':positions,'orders':[asdict(v) for v in broker.orders],
        'fills':[asdict(f) for f in broker.fills],'journal':broker.journal}
    # Serialize now: no mutable reference is allowed to observe post-action state.
    encoded=canonical(state)
    return {'sha256':hashlib.sha256(encoded.encode()).hexdigest(),'state':json.loads(encoded),
        'held_quantity':pos.qty_open if pos else 0,
        'entry_price':pos.entry_price if pos else None,
        'entry_route':pos.metadata.get('entry_route') if pos else None,
        'parent_complete_exit':bool(pos and (pos.exit_in_flight or pos.metadata.get('liquidation_reason')))}


def independent_account(broker,indexes):
    """Reconcile each physical fill, fee, cash, inventory and raw-close MTM."""
    from scripts.kalcb_executed_value_replay import equity_records
    fills=defaultdict(list)
    for fill in broker.fills:fills[fill.timestamp.isoformat()].append(fill)
    cash=ev.REFERENCE_EQUITY;qty=defaultdict(int);marks=0
    for timestamp,equity in equity_records(broker.timestamps,broker.equity_curve):
        timestamp=datetime.fromisoformat(timestamp).isoformat()
        day=datetime.fromisoformat(timestamp).date().isoformat()
        for f in fills.pop(timestamp,[]):
            notional=f.qty*f.price
            commission=notional*broker.costs.commission_bps/10000
            tax=notional*broker.costs.tax_bps_on_sell/10000 if f.side=='SELL' else 0
            if f.side=='BUY':cash-=notional+commission;qty[f.symbol]+=f.qty
            else:cash+=notional-commission-tax;qty[f.symbol]-=f.qty
            if qty[f.symbol]<0:raise ValueError('Independent negative inventory')
        marked=cash
        for symbol,n in qty.items():
            if n:marked+=n*indexes[day][symbol,timestamp,'5m'][3]
        if abs(marked-equity)>.011:raise ValueError('Independent raw-close MTM mismatch')
        marks+=1
    if fills or abs(cash-broker.cash)>.011 or any(qty.values()):
        raise ValueError('Independent terminal cash/inventory mismatch')
    return {'passed':True,'raw_close_marks':marks,'terminal_cash':cash,'terminal_inventory':dict(qty)}


def arm_episode(row,bars,cfg,scenario,conn,sessions,context,arm,indexes,observe=True):
    from scripts.kalcb_executed_value_study import run_episode
    from backtests.engine import sim_broker
    from backtests.engine.integrity import audit_broker
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter
    clock=datetime.fromisoformat(context['available_at']);symbol=row['symbol']
    brokers=[];captures=[];original_broker=sim_broker.SimBroker
    original_end=KALCBReplayAdapter.on_timestamp_end
    def new_broker(*args,**kwargs):
        b=original_broker(*args,**kwargs);brokers.append(b);return b
    def end(adapter,timestamp,batch,broker):
        decisions=original_end(adapter,timestamp,batch,broker)
        if timestamp+timedelta(minutes=5)!=clock:return decisions
        if captures:raise ValueError('Duplicate checkpoint')
        before=state_before(adapter,broker,symbol,clock)
        ss=adapter.state.symbols.get(symbol);pos=ss.position if ss else None
        eligibility=('no_open_position' if not pos or pos.qty_open<=0 else
            'not_original_opening_route' if pos.metadata.get('entry_route')!='first30_open_anchor' else
            'parent_complete_exit' if before['parent_complete_exit'] else 'eligible')
        captures.append({**before,'eligibility':eligibility,'action_submitted':False})
        if arm=='liquidate':
            actions,reason=checkpoint_actions(adapter.state,symbol)
            if reason!=eligibility:raise ValueError('Checkpoint classification differs')
            if actions:
                adapter._submit_actions(actions,broker,timestamp)
                captures[-1]['action_submitted']=True
        return decisions
    with patch.object(sim_broker,'SimBroker',new_broker):
        if observe:
            with patch.object(KALCBReplayAdapter,'on_timestamp_end',end):
                value,detail=run_episode(row,bars,cfg,scenario,'completed45-native-contrast-v1',conn,sessions)
        else:value,detail=run_episode(row,bars,cfg,scenario,'completed45-native-contrast-v1',conn,sessions)
    if len(brokers)!=1:raise ValueError('Unexpected native broker count')
    broker=brokers[0];audit=audit_broker(broker);checked=0
    for batch in broker.expected_batches:
        for b in batch:
            day=datetime.fromisoformat(b['timestamp']).date().isoformat()
            expected=indexes[day].get((b['symbol'],b['timestamp'],b['timeframe']))
            if expected!=tuple(b[k] for k in ('open','high','low','close','volume','is_completed')):
                raise ValueError('Native label consumed non-source bar')
            checked+=1
    account=independent_account(broker,indexes) if not value['unresolved'] else {'passed':False,'unresolved_retained':True}
    capture=captures[0] if captures else {'eligibility':'missing_native_clock','action_submitted':False,'held_quantity':0,'sha256':None}
    original_entries=[r for r in broker.journal if r['event']=='submit' and r['order']['side']=='BUY']
    if original_entries and original_entries[0]['order']['qty']!=row['action']['quantity']:
        raise ValueError('Original integer entry quantity differs from declared dose')
    requested=[r for r in broker.journal if r['event']=='submit' and r['order'].get('reason')==REASON]
    changed_ids={r['order']['order_id'] for r in requested}
    changed_fills=[asdict(f) for f in broker.fills if f.order_id in changed_ids]
    if capture['action_submitted'] and len(requested)!=1:raise ValueError('Missing or duplicate exit submission')
    if any(datetime.fromisoformat(str(f['timestamp']))<clock for f in changed_fills):raise ValueError('Same/incomplete bar exit fill')
    value.update(arm=arm,checkpoint={k:v for k,v in capture.items() if k!='state'},
        raw_bars_verified=checked,independent_execution_pass=audit['execution_validity_pass'],
        independent_account_pass=account['passed'],full_portfolio_audit_status=audit['audit_status'],
        actual_changed_exit_fills=len(changed_fills),actual_changed_exit_quantity=sum(f['qty'] for f in changed_fills))
    detail.update(arm=arm,checkpoint=capture,journal=broker.journal,equity_curve=broker.equity_curve,
        timestamps=broker.timestamps,terminal_cash=broker.cash,independent_account=account,
        independent_engine_audit=audit,costs={k:getattr(broker.costs,k) for k in ('commission_bps','tax_bps_on_sell','slippage_bps')})
    return value,detail


def paired_episode(row,bars,cfg,scenario,conn,sessions,context,indexes,bridge=False):
    hold,hd=arm_episode(row,bars,cfg,scenario,conn,sessions,context,'hold',indexes)
    if bridge:
        base,bd=arm_episode(row,bars,cfg,scenario,conn,sessions,context,'hold',indexes,observe=False)
        keys=('fills','orders','journal','equity_curve','timestamps','terminal_cash')
        if any(canonical(hd[k])!=canonical(bd[k]) for k in keys) or hold['net_value_krw']!=base['net_value_krw']:
            raise ValueError('Observed HOLD changed original native episode')
    liquid,ld=arm_episode(row,bars,cfg,scenario,conn,sessions,context,'liquidate',indexes)
    if hold['checkpoint']['sha256']!=liquid['checkpoint']['sha256']:
        raise ValueError('Arms differ before the action')
    eligibility=hold['checkpoint']['eligibility'];eligible=eligibility=='eligible'
    if not eligible and any(canonical(hd[k])!=canonical(ld[k]) for k in ('fills','orders','equity_curve')):
        raise ValueError('Ineligible arms should be exact economic aliases')
    valid=eligible and all(v['label_valid'] and v['independent_execution_pass'] and v['independent_account_pass'] for v in (hold,liquid))
    notional=hold['checkpoint']['held_quantity']*context['completed_close']
    target=(liquid['net_value_krw']-hold['net_value_krw'])/notional*10000 if valid and notional>0 else None
    pair={'opportunity_id':row['opportunity_id'],'day':row['trade_date'],'symbol':row['symbol'],
        'scenario':scenario,'quantity_scale':row['features']['action_quantity_scale'],
        'planned_quantity':row['action']['quantity'],'checkpoint_at':context['available_at'],
        'context_features':context['context_features'],'own_features':context['own_features'],
        'eligibility':eligibility,'paired_label_valid':valid,'held_notional_krw':notional,
        'liquidate_minus_hold_bps':target,'liquidate_minus_hold_krw':liquid['net_value_krw']-hold['net_value_krw'] if valid else None,
        'label_available_at':max(hold['label_available_at'],liquid['label_available_at']) if valid else None,
        'hold':hold,'liquidate':liquid,'exact_predecision_identity':True,
        'original_unhooked_hold_bridge':bridge,'isolated_not_funded_portfolio_pnl':True}
    return pair,{'pair':{k:pair[k] for k in ('opportunity_id','scenario','checkpoint_at','eligibility')},'hold':hd,'liquidate':ld}


class SourceIndexes(dict):
    def __init__(self,databases):super().__init__();self.databases=databases
    def __missing__(self,day):
        with sqlite3.connect('file:'+Path(self.databases[int(day>='2026-04-01')]).as_posix()+'?mode=ro',uri=True) as c:
            value={(s,t,'5m'):(o,h,l,cl,v,True) for s,t,o,h,l,cl,v in c.execute('SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day=?',(day,))}
        self[day]=value;return value


def run(root,smoke=False):
    from scripts.kalcb_executed_value_study import config_for,load_day
    from scripts.kalcb_action_quantity_labels import scaled_row
    p=read(root/'protocol.json');verify(p['bindings'])
    contexts={}
    with gzip.open(p['contexts']['path'],'rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row['context_supported']:contexts[row['day'],row['symbol']]=row
    days=sorted({day for day,_ in contexts});selected=days[:1] if smoke else days
    out=root/('smoke' if smoke else 'labels');out.mkdir(exist_ok=False)
    conn=continuous_connection(p['databases']);sessions=[r[0] for r in conn.execute('SELECT DISTINCT day FROM bars ORDER BY day')]
    if len(sessions)!=286 or sessions.index('2026-04-01')!=217:raise ValueError('Changed continuous source history')
    by_day=defaultdict(list)
    for i,folder in enumerate(p['opening_prepared']):
        with (Path(folder)/'opportunities.jsonl').open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line);day=row['trade_date'];key=(day,row['symbol'])
                if (i==0 and day>'2026-03-31') or (i==1 and day<'2026-04-01'):continue
                if day in selected and key in contexts:
                    if not row['action']['ready']:raise ValueError('Frozen source-ready population changed')
                    by_day[day].append(row)
    cfgs={s:replace(config_for(read(p['plan']['path']),s),entry_action_quantity_scale_enabled=True) for s in p['scenarios']}
    counts=Counter();bindings=[];began=time.perf_counter();pairs=0
    for day in selected:
        indexes=SourceIndexes(p['databases']);bars=defaultdict(list)
        for b in load_day(conn,day,'completed45-native-contrast-v1',{r['symbol'] for r in by_day[day]}):bars[b.symbol].append(b)
        for scenario in p['scenarios']:
            name=day+'_'+scenario
            source=root/'smoke'/(name+'.jsonl.gz')
            values=out/(name+'.jsonl.gz');events=out/(name+'.events.jsonl.gz')
            if not smoke and source.exists():
                receipt=read(root/'smoke/completion.json')
                artifacts=[b for b in receipt['artifacts'] if Path(b['path']).name in (values.name,events.name)]
                if len(artifacts)!=2:raise ValueError('Unauthenticated smoke artifacts')
                verify(artifacts)
                for dst in (values,events):os.link(source.parent/dst.name,dst)
                with gzip.open(values,'rt',encoding='utf-8') as f:rows=[json.loads(line) for line in f]
                for r in rows:counts[r['eligibility']]+=1;counts['valid_pairs']+=r['paired_label_valid']
                pairs+=len(rows)
            else:
                with gzip.open(values,'xt',encoding='utf-8',compresslevel=1) as vf,gzip.open(events,'xt',encoding='utf-8',compresslevel=1) as ef:
                    for row in sorted(by_day[day],key=lambda r:r['symbol']):
                        for q in p['quantities']:
                            action=scaled_row(row,q)
                            pair,detail=paired_episode(action,bars[row['symbol']],cfgs[scenario],scenario,conn,sessions,contexts[day,row['symbol']],indexes,bridge=smoke)
                            vf.write(canonical(pair)+'\n');ef.write(canonical(detail)+'\n')
                            pairs+=1;counts[pair['eligibility']]+=1;counts['valid_pairs']+=pair['paired_label_valid']
                            for arm in ('hold','liquidate'):
                                counts['execution_failures']+=not pair[arm]['independent_execution_pass']
                                counts['account_failures']+=not pair[arm]['independent_account_pass']
                                counts['deadline_violations']+=bool(pair[arm]['deadline_violation'])
                # All losses, no-order rows, already-exiting states and failures persist.
            bindings += [bound(values),bound(events)]
            elapsed=time.perf_counter()-began
            status={'pairs':pairs,'planned_pairs':p['planned_pairs'],'day':day,'scenario':scenario,'counts':dict(counts),'elapsed_seconds':elapsed,'pairs_per_second':pairs/elapsed}
            write(root/'progress.json',status);print(json.dumps(status),flush=True)
    conn.close();verify(p['bindings'])
    write(out/'completion.json',{'complete':True,'study_complete':False,'pairs':pairs,'counts':dict(counts),'artifacts':bindings,'elapsed_seconds':time.perf_counter()-began,
        'full_portfolio_audit_status':'unverified','models':0,'policies':0,'portfolio_cases':0})

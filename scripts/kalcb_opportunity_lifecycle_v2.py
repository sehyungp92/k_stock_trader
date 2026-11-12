"""C12 saved-account episode census. No model fits or native policy replays."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip, hashlib, json, math, sys, traceback
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
BASE=Path('E:/CodexResearch/k_stock_trader')
OLD=BASE/'kalcb_retained_funding_20261016'
ROOT=BASE/'kalcb_opportunity_lifecycle_20261016/census_v2'
DESIGN=REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0036.md'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def bound(p):return dict(path=str(p),sha256=digest(p))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def lines(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:
        for row in f:yield json.loads(row)
def key(day,symbol):return day[:10]+'/'+symbol
def stamp(x):return datetime.fromisoformat(x)
def selected_meta(m):
    names=('entry_route','entry_route_mode','entry_route_attempts','decision_available_at','entry_price_ref','stop_price',
           'risk_per_share','portfolio_equity','portfolio_drawdown_pct','portfolio_session_return_pct',
           'effective_max_position_notional_pct','entry_route_risk_mult','entry_route_notional_mult','candidate_rank',
           'frontier_rank','candidate_hash','source_artifact_hash','action_quantity_scale','entry_checkpoint_at',
           'first30_ret','first30_rel_volume','first30_signal_bar_cpr','avwap','entry_quality_votes','entry_path_quality',
           'reason','message','order_id','order_role','entry_order_expires_at','management_reference_price')
    return {n:m[n] for n in names if n in m}

def trade_record(t):
    m=t['route_metadata'];equity=float(m.get('portfolio_equity') or 0)
    notional=t['qty']*t['entry_price']
    return dict(symbol=t['symbol'],entry_decision_time=t['entry_decision_time'],entry_fill_time=t['entry_fill_time'],
        exit_fill_time=t['exit_fill_time'],qty=t['qty'],entry_price=t['entry_price'],exit_price=t['exit_price'],
        net_pnl=t['net_pnl'],exit_reason=t['exit_reason'],material=equity>0 and notional>=.01*equity,
        useful=equity>0 and notional>=.01*equity and t['net_pnl']>=.0001*equity,metadata=selected_meta(m))

def classify(e):
    tags=[];trades=sorted(e['trades'],key=lambda t:stamp(t['entry_fill_time']))
    if e['negative_opening_signal']:tags.append('negative_opening_signal')
    if e['veto_actions']:tags.append('observed_opening_veto')
    if e['reservation_omitted']:tags.append('reservation_omitted')
    if e['native_rejects']:tags.append('native_order_rejected')
    if any(r['reason']=='buying_power' for r in e['native_rejects']):tags.append('buying_power_rejected')
    if e['cancels']:tags.append('cancelled_or_expired_order')
    if any(t['metadata'].get('entry_route')!='first30_open_anchor' for t in trades):tags.append('later_route_filled')
    if 'later_route_filled' in tags and e['veto_actions']:tags.append('later_fill_after_veto')
    if 'later_route_filled' in tags and e['reservation_omitted']:tags.append('later_fill_after_reservation_omission')
    if any(stamp(t['entry_fill_time'])>stamp(r['timestamp']) for t in trades for r in e['native_rejects']):tags.append('fill_after_native_rejection')
    if any(stamp(t['entry_fill_time'])>stamp(r['timestamp']) for t in trades for r in e['cancels'] if r['side']=='BUY'):tags.append('fill_after_entry_cancel')
    if any(stamp(b['entry_fill_time'])>=stamp(a['exit_fill_time']) for a,b in zip(trades,trades[1:])):tags.append('reentry_after_exit')
    if len(e['buy_submits'])>1:tags.append('multiple_buy_proposals')
    if not e['signal_present']:tags.append('no_c9_opening_forecast')
    if not trades and e['buy_submits']:tags.append('submitted_unfilled')
    e['tags']=tags;e['net_pnl']=sum(t['net_pnl'] for t in trades)
    e['useful_campaigns']=sum(t['useful'] for t in trades)
    e['useful_issuer_day']=bool(trades) and any(t['material'] for t in trades) and e['net_pnl']>=.0001*float(trades[0]['metadata'].get('portfolio_equity') or 0)
    return e

def census(folder,signals):
    # R1: corroborate fresh decision-journal observations against bound outputs.
    observed=Counter(); proposed=[]; submitted=[]
    for d in lines(folder/'decisions.jsonl.gz'):
        observed[d['reason']]+=1
        for action in d.get('actions',[]):
            if action['action_type']=='SubmitEntry':
                proposed.append((d['timestamp'],action['symbol'],action['qty']))
    if dict(observed)!=read(folder/'result.json')['decision_counts']:
        raise ValueError('Decision histogram differs from bound result')
    for row in lines(folder/'orders.jsonl.gz'):
        o=row.get('order',{})
        if row['event']=='submit' and o.get('side')=='BUY':
            submitted.append((o['metadata']['signal_bar'],o['symbol'],o['qty']))
    if sorted(proposed)!=sorted(submitted):raise ValueError('Decision submissions differ from bound order journal')
    episodes={}
    def ep(day,symbol):
        k=key(day,symbol)
        if k not in episodes:
            s=signals.get(k)
            episodes[k]=dict(key=k,signal_present=s is not None,negative_opening_signal=bool(s and s['predictions']['baseline']<-23 and s['predictions']['combined']<-33),
                signal=s,decisions=[],buy_submits=[],native_rejects=[],cancels=[],fills=[],trades=[],veto_actions=[],reservation_omitted=False)
        return episodes[k]
    for i,x in enumerate(lines(folder/'decisions.jsonl.gz')):
        e=ep(x['timestamp'],x['symbol']);m=x.get('metadata',{})
        e['decisions'].append(dict(row=i,at=x['timestamp'],code=x['decision_code'],reason=x['reason'],metadata=selected_meta(m)))
    event_counts=Counter()
    for i,x in enumerate(lines(folder/'orders.jsonl.gz')):
        event_counts[x['event']]+=1;o=x.get('order',{});symbol=o.get('symbol')
        if not symbol:continue
        at=x.get('timestamp') or o['submitted_at'];e=ep(at,symbol);m=o.get('metadata',{})
        row=dict(row=i,timestamp=at,event=x['event'],side=o['side'],qty=o['qty'],order_id=o['order_id'],
                 reason=x.get('reason'),metadata=selected_meta(m))
        if x['event']=='submit' and o['side']=='BUY':e['buy_submits'].append(row)
        if x['event']=='reject' and o['side']=='BUY':
            row.update(cash=x.get('cash'),attempted_qty=x.get('attempted_qty'),execution_price=x.get('execution_price'))
            e['native_rejects'].append(row)
        if x['event'] in ('cancel','expire','cancelled','expired'):e['cancels'].append(row)
    for i,x in enumerate(read(folder/'fills.json')):
        e=ep(x['timestamp'],x['symbol'])
        e['fills'].append(dict(row=i,timestamp=x['timestamp'],side=x['side'],qty=x['qty'],price=x['price'],order_id=x['order_id'],reason=x['reason']))
    trades=read(folder/'trades.json')
    for t in trades:ep(t['entry_decision_time'],t['symbol'])['trades'].append(trade_record(t))
    if (folder/'horizon_actions.json').exists():
        for a in read(folder/'horizon_actions.json'):
            if a['kind']=='veto':
                day,symbol=a['key'].split('/');ep(day,symbol)['veto_actions'].append(a)
    if (folder/'reservation_actions.json').exists():
        for a in read(folder/'reservation_actions.json'):
            for s,v in a.get('book',{}).items():
                if v['qty']==0:ep(a['day'],s)['reservation_omitted']=True
    for e in episodes.values():classify(e)
    actual=sum(e['net_pnl'] for e in episodes.values());expected=sum(t['net_pnl'] for t in trades)
    if not math.isclose(actual,expected,abs_tol=.001):raise ValueError('Episode accounting mismatch')
    summary={};routes={}
    tags=sorted({x for e in episodes.values() for x in e['tags']})
    for tag in ['all',*tags]:
        rows=[e for e in episodes.values() if tag=='all' or tag in e['tags']]
        valued=sorted(rows,key=lambda e:(e['net_pnl'],e['key']))
        tracekeys=[]
        if rows:tracekeys=list(dict.fromkeys([valued[0]['key'],valued[-1]['key'],sorted(rows,key=lambda e:e['key'])[0]['key'],valued[len(valued)//2]['key']]))
        summary[tag]=dict(issuer_days=len(rows),dates=len({e['key'][:10] for e in rows}),
            buy_proposals=sum(len(e['buy_submits']) for e in rows),campaigns=sum(len(e['trades']) for e in rows),
            net_pnl=sum(e['net_pnl'] for e in rows),positive_wealth=sum(max(e['net_pnl'],0) for e in rows),
            negative_wealth=sum(min(e['net_pnl'],0) for e in rows),useful_issuer_days=sum(e['useful_issuer_day'] for e in rows),
            useful_campaigns=sum(e['useful_campaigns'] for e in rows),tracekeys=tracekeys)
    for route in sorted({t['metadata'].get('entry_route','unknown') for e in episodes.values() for t in e['trades']}):
        ts=[t for e in episodes.values() for t in e['trades'] if t['metadata'].get('entry_route','unknown')==route]
        routes[route]=dict(campaigns=len(ts),net_pnl=sum(t['net_pnl'] for t in ts),useful_campaigns=sum(t['useful'] for t in ts))
    return dict(episodes=episodes,summary=summary,routes=routes,journal_event_counts=dict(event_counts),reconciliation_error=actual-expected)

def run():
    ROOT.mkdir(parents=True,exist_ok=False)
    previous=read(OLD/'closeout_v1/registration.json')
    historical={str(Path(b['path']).resolve()).lower():b['sha256'] for b in previous['inputs']}
    completion=read(OLD/'closeout_v1/completion.json')
    if digest(OLD/'closeout_v1/registration.json')!=completion['registration']['sha256']:raise ValueError('Changed parent audit registration')
    p=read(OLD/'v1/protocol.json');reg=read(OLD/'v1/registration.json')
    for item in read(OLD/'v1/completed_cases.json'):
        b=item['completion'];historical[str(Path(b['path']).resolve()).lower()]=b['sha256']
    fresh_observation_bindings=[]

    cases={}
    for period in ('is','oos'):
        for scenario in ('baseline','combined'):
            cases[f'prior_{period}_{scenario}']=p['prior_controls'][period+'/'+scenario]
            cases[f'veto_{period}_{scenario}']=p['controls'][period+'/'+scenario]
            for arm in ('queue','forecast'):cases[f'{arm}_{period}_{scenario}']=str(OLD/'v1/cases'/f'{arm}_{period}_{scenario}')
    inputs=[]
    for folder in cases.values():
        for name in ('decisions.jsonl.gz','orders.jsonl.gz','fills.json','trades.json','result.json','completion.json','input_profile.json','horizon_actions.json','reservation_actions.json'):
            q=Path(folder)/name
            if not q.exists():
                if name in ('horizon_actions.json','reservation_actions.json'):continue
                raise ValueError('Missing account source '+str(q))
            b=bound(q);expected=historical.get(str(q.resolve()).lower())
            if expected is None and name=='decisions.jsonl.gz':
                fresh_observation_bindings.append(dict(**b,scope='Fresh diagnostic observation, histogram and submitted-action corroboration required'))
            elif expected is None and name=='input_profile.json':
                if read(q) not in list(reg['profiles'].values()):raise ValueError('Unbound profile differs from registered runtime')
            elif expected!=b['sha256']:raise ValueError('Unauthenticated account source '+str(q))

            inputs.append(b)
    sourcefiles={};configurations={}
    for period,item in reg['profiles'].items():
        code=Path(item['manifest']['path']).parent/'files';manifest=read(item['manifest']['path'])
        for rel in ('strategy_kalcb/core/logic.py','strategy_kalcb/core/state.py','strategy_kalcb/risk.py','scripts/kalcb_executed_value_study.py','backtests/strategies/kalcb/runner.py','backtests/strategies/kalcb/funded_horizon_v1.py'):
            q=code/rel;b=bound(q)
            if manifest['files'].get(rel)!=b['sha256']:raise ValueError('Changed executed source '+rel)
            inputs.append(b);sourcefiles[period+'/'+rel]=b
        plan=read(item['plan']['path']);cfg=Path(plan['config'])
        # Plan is authenticated by the original registration; its input map binds config.
        if digest(item['plan']['path'])!=item['plan']['sha256']:raise ValueError('Changed plan')
        if digest(cfg)!=plan['input_sha256'].get(str(cfg)):raise ValueError('Unbound configuration')
        inputs.extend([bound(cfg),bound(item['plan']['path'])]);configurations[period]=read(cfg)
    signal=Path(p['signal_map']['path'])
    if digest(signal)!=p['signal_map']['sha256']:raise ValueError('Changed signal map')
    inputs.extend([bound(signal),bound(DESIGN),bound(__file__),bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0036-repair-r1.md'),bound(OLD/'v1/completed_cases.json'),bound(OLD/'closeout_v1/registration.json')])
    write(ROOT/'registration.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),experiment='kalcb-exp-0036',stage='A',cases=cases,inputs=inputs,fresh_observation_bindings=fresh_observation_bindings,native_accounts=0,fits=0))
    signals=read(signal);summaries={}
    for name,folder in cases.items():
        result=census(Path(folder),signals)
        with gzip.open(ROOT/(name+'.episodes.jsonl.gz'),'xt',encoding='utf-8') as f:
            for e in sorted(result.pop('episodes').values(),key=lambda e:e['key']):f.write(json.dumps(e,ensure_ascii=False,allow_nan=False)+'\n')
        summaries[name]=result
        print(name,json.dumps({k:v for k,v in result['summary'].items() if k in ('later_route_filled','later_fill_after_veto','fill_after_native_rejection','reentry_after_exit')}),flush=True)
    write(ROOT/'assessment.json',dict(cases=summaries,configurations=configurations,sourcefiles=sourcefiles,
        scope='Observed episodes, not counterfactual returns. Tags overlap; useful metrics retain existing thresholds.'))
    for b in inputs:
        if digest(b['path'])!=b['sha256']:raise ValueError('Input changed during census '+b['path'])
    write(ROOT/'completion.json',dict(complete=True,at_utc=datetime.now(timezone.utc).isoformat(),registration=bound(ROOT/'registration.json'),assessment=bound(ROOT/'assessment.json'),native_accounts=0,fits=0,economic_objective_achieved=False))

if __name__=='__main__':
    try:run()
    except Exception as e:
        if ROOT.exists() and not (ROOT/'failure.json').exists():write(ROOT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise

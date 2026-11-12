"""Read-only funded-account attribution after registered population screening.

Conditional block intervals describe paired historical paths. They do not make
reused development independent or correct for selection across tested variants.
This program never generates a trading policy or selects a live configuration.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify


def compound(values):
    return math.prod(1+x for x in values)-1


def percentile(values,p):
    ordered=sorted(values);position=(len(ordered)-1)*p;left=int(position)
    return ordered[left]+(ordered[min(left+1,len(ordered)-1)]-ordered[left])*(position-left)


def paired_blocks(a,b,length,*,draws=2000,seed=631027):
    if len(a)!=len(b) or not a or length<1:raise ValueError('Unpaired calendar')
    rng=random.Random(seed);n=len(a);deltas=[]
    for _ in range(draws):
        indexes=[]
        while len(indexes)<n:
            start=rng.randrange(n)
            indexes.extend((start+j)%n for j in range(length))
        indexes=indexes[:n]
        deltas.append(100*(compound(a[i] for i in indexes)-compound(b[i] for i in indexes)))
    return {'block_sessions':length,'draws':draws,'seed':seed,
            'return_delta_pp_2_5':percentile(deltas,.025),'return_delta_pp_50':percentile(deltas,.5),
            'return_delta_pp_97_5':percentile(deltas,.975),
            'resample_share_positive':sum(x>0 for x in deltas)/draws,
            'scope':'Conditional paired circular-block resampling of observed daily percentage returns; not independent validation, familywise inference or funded constant-capital replay.'}


def account(folder):
    folder=Path(folder);daily=read(folder/'daily.json');trades=read(folder/'trades.json');result=read(folder/'result.json')
    if not read(folder/'independent_account_verification.json')['passed']:raise ValueError('Unverified account')
    dates=[d['day'] for d in daily]
    if dates!=sorted(set(dates)):raise ValueError('Duplicate/nonchronological session')
    balances=[100_000_000.]+[float(d['equity']) for d in daily]
    daily_returns=[balances[i+1]/balances[i]-1 for i in range(len(daily))]
    trade_net=sum(t['net_pnl'] for t in trades)
    if abs(trade_net-(balances[-1]-balances[0]))>.01:raise ValueError('Trade cash mismatch')
    if abs(compound(daily_returns)*100-result['return_pct'])>1e-8:raise ValueError('Daily return mismatch')
    monthly={};issuer=defaultdict(lambda:{'net_pnl_krw':0.,'trades':0});issuer_days=defaultdict(float)
    for i,day in enumerate(dates):
        month=day[:7]
        row=monthly.setdefault(month,{'start_equity_krw':balances[i],'end_equity_krw':balances[i],'sessions':0,'trades':0,'trade_net_pnl_krw':0.})
        row['end_equity_krw']=balances[i+1];row['sessions']+=1
    for t in trades:
        day=t['entry_decision_time'][:10]
        if day!=t['exit_fill_time'][:10]:raise ValueError('Overnight position requires different attribution')
        monthly[day[:7]]['trades']+=1;monthly[day[:7]]['trade_net_pnl_krw']+=t['net_pnl']
        issuer[t['symbol']]['net_pnl_krw']+=t['net_pnl'];issuer[t['symbol']]['trades']+=1
        issuer_days[day+'/'+t['symbol']]+=t['net_pnl']
    for row in monthly.values():
        row['return_pct']=100*(row['end_equity_krw']/row['start_equity_krw']-1)
        row['cash_change_krw']=row['end_equity_krw']-row['start_equity_krw']
        if abs(row['cash_change_krw']-row['trade_net_pnl_krw'])>.01:raise ValueError('Monthly cash mismatch')
    gains=sorted((v['net_pnl_krw'] for v in issuer.values() if v['net_pnl_krw']>0),reverse=True)
    issuer_day_values=sorted(issuer_days.values(),reverse=True)
    top_count=max(1,math.ceil(len(issuer_day_values)*.05))
    gross=sum(t['gross_pnl'] for t in trades);costs=sum(t['commission'] for t in trades)
    if abs(gross-costs-trade_net)>.01:raise ValueError('Cost reconciliation failed')
    metrics={k:result[k] for k in ('return_pct','drawdown_pct','trades','active_days','audit_status')}
    return {'metrics':metrics,'dates':dates,'daily_returns':daily_returns,'monthly':monthly,
            'issuers':dict(issuer),'issuer_days':dict(issuer_days),'net_pnl_krw':trade_net,
            'gross_pnl_krw':gross,'explicit_costs_krw':costs,
            'explicit_cost_note':'Fees/taxes recorded in commission; slippage is already in physical prices and gross PnL.',
            'positive_issuer_profit_top1_share':gains[0]/sum(gains) if gains else None,
            'positive_issuer_profit_top5_share':sum(gains[:5])/sum(gains) if gains else None,
            'top_5pct_issuer_days_count':top_count,'top_5pct_issuer_days_net_pnl_krw':sum(issuer_day_values[:top_count]),
            'traded_issuer_days':len(issuer_days),
            'bindings':[bound(folder/name) for name in ('result.json','daily.json','trades.json','equity.json','fills.json','independent_account_verification.json')]}


def paired_attribution(a,b):
    if a['dates']!=b['dates']:raise ValueError('Different calendars')
    oldkeys=set(b['issuer_days']);newkeys=set(a['issuer_days'])
    old_top=set(sorted(oldkeys,key=b['issuer_days'].get,reverse=True)[:b['top_5pct_issuer_days_count']])
    return {'metric_delta':{k:a['metrics'][k]-b['metrics'][k] for k in ('return_pct','drawdown_pct','trades','active_days')},
            'monthly':{m:{k:a['monthly'][m][k]-b['monthly'][m][k] for k in ('return_pct','trades','cash_change_krw')} for m in a['monthly']},
            'new_traded_issuer_days':len(newkeys-oldkeys),'parent_traded_issuer_days_absent':len(oldkeys-newkeys),
            'parent_top_5pct_issuer_days_count':len(old_top),
            'parent_top_5pct_issuer_days_retained':len(old_top&newkeys),
            'parent_top_5pct_issuer_days_original_pnl_krw':sum(b['issuer_days'][k] for k in old_top),
            'same_issuer_days_variant_pnl_krw':sum(a['issuer_days'].get(k,0) for k in old_top),
            'attribution_scope':'Observed funded portfolio paths; same issuer-day can have changed timing, quantity, retries and funding. These sums are not isolated causal PnL or recovered MFE.',
            'paired_blocks':[paired_blocks(a['daily_returns'],b['daily_returns'],length) for length in (5,10)]}


def assess(root,out):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Preserve prior assessment; use a versioned output')
    completion=read(root/'screening_completion.json')
    if not completion['complete']:raise ValueError('Screening incomplete')
    protocol=read(root/'protocol.json');study=read(root/'study.json');verify(protocol['bindings']+protocol['programs'])
    expected=set(study['recipes']);records={p.parent.name:read(p) for p in (root/'cases').glob('*/record.json')}
    if set(records)!=expected:raise ValueError('Missing or undeclared trials')
    controls={parent:name for name,parent in study['controls'].items()};accounts={};trials=[]
    for name in study['recipes']:
        if records[name].get('rejected'):continue
        for period in ('is','oos'):
            for scenario in ('baseline','combined'):
                accounts[name,period,scenario]=account(root/'cases'/name/period/scenario)
    for name in study['recipes']:
        row={'recipe':name,'specification':study['recipes'][name],'rejected':records[name].get('rejected',False),'cases':{}}
        if row['rejected']:
            row['rejection']=records[name];trials.append(row);continue
        parent=controls[study['recipes'][name]['parent']]
        for period in ('is','oos'):
            for scenario in ('baseline','combined'):
                key=period+'/'+scenario;a=accounts[name,period,scenario];b=accounts[parent,period,scenario]
                row['cases'][key]={'account':a,'vs_local_parent':paired_attribution(a,b)}
        row['global_IS_sacrifice_pp']={s:accounts['control','is',s]['metrics']['return_pct']-accounts[name,'is',s]['metrics']['return_pct'] for s in ('baseline','combined')}
        row['IS_reporting_bands']={str(b):all(x<=b for x in row['global_IS_sacrifice_pp'].values()) for b in (2,5,10)}
        row['local_joint_dominance_all_four_cells']=all(
            all(delta[k]>=0 for k in ('return_pct','trades','active_days')) and delta['drawdown_pct']<=0
            for delta in (v['vs_local_parent']['metric_delta'] for v in row['cases'].values())) and name!=parent
        trials.append(row)
    inputs=[bound(root/name) for name in ('protocol.json','study.json','screening_completion.json')]+[bound(p) for p in (root/'cases').glob('*/record.json')]
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'program':bound(__file__),'inputs':inputs,
            'scope':'All registered native trials; reused development. Descriptive paired resampling, no automatic winner or promotion. Full four-scenario stability/causal/funded-constant-capital qualification remains necessary.',
            'trials':trials}
    write(out/'assessment.json',result)
    print(json.dumps({'saved':str(out/'assessment.json'),'trials':len(trials),'promotion':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assess(args.root,args.output)

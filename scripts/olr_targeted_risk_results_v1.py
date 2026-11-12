"""Authenticate native trials and quantify return/risk/frequency trade-offs."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import bisect
import json
import numpy as np
from scripts import olr_targeted_risk_study_v1 as t
from scripts import olr_boundary_restart_results_v1 as history
n,s,b=t.n,t.s,t.b


def episode(rows):
    values=np.array([float(v[1]) for v in rows])
    peak=np.maximum.accumulate(np.r_[1e8,values])[1:]
    dd=1-values/peak
    trough=int(np.argmax(dd));high=int(np.argmax(values[:trough+1]))
    return {'drawdown_pct':float(dd[trough]*100),'peak_time':rows[high][0],
        'trough_time':rows[trough][0],'peak_nav':float(peak[trough]),'trough_nav':float(values[trough]),
        'recovered_after_trough':bool(any(values[trough+1:]>=peak[trough]))}


def describe(folder):
    result=n.read(folder/'result.json');trades=n.read(folder/'trades.json')
    daily=n.read(folder/'daily.json');equity=n.read(folder/'equity.json')
    bymonth={}
    previous=1e8
    daily_returns=[]
    for row in daily:
        month=row['day'][:7]
        bymonth.setdefault(month,{'start':previous,'end':previous,'sessions':0})
        bymonth[month]['end']=row['equity'];bymonth[month]['sessions']+=1
        daily_returns.append(row['equity']/previous-1);previous=row['equity']
    for row in bymonth.values():row['return_pct']=100*(row['end']/row['start']-1)
    counts=Counter(x['entry_fill_time'][:10] for x in trades)
    ranks={}
    for trade in trades:
        rank=str(trade['route_metadata'].get('candidate_rank','unknown'))
        bucket=ranks.setdefault(rank,{'trades':0,'winners':0,'pnl':0.})
        bucket['trades']+=1;bucket['winners']+=trade['net_pnl']>0;bucket['pnl']+=trade['net_pnl']
    stamps=[v[0] for v in equity];risks=[]
    for trade in trades:
        index=bisect.bisect_left(stamps,trade['entry_fill_time'])-1
        nav=equity[index][1] if index>=0 else 1e8
        risks.append(100*trade['qty']*float(trade['route_metadata'].get('risk_per_share',0))/nav)
    budget_overshoots=[]
    for trade in trades:
        budget=trade['route_metadata'].get('scenario_loss_budget',{})
        if budget.get('position_pct') and -trade['net_pnl']>budget['equity']*budget['position_pct']:
            budget_overshoots.append({'symbol':trade['symbol'],'entry':trade['entry_decision_time'],
                'realized_loss_pct_decision_nav':-100*trade['net_pnl']/budget['equity'],
                'position_scenario_budget_pct':100*budget['position_pct'],'net_pnl':trade['net_pnl']})
    return {**{k:result[k] for k in ('return_pct','drawdown_pct','trades','sessions','active_days','trades_per_21_sessions')},
        'entry_days':len(counts),'mean_trades_per_entry_day':len(trades)/len(counts),'rank_outcomes':ranks,
        'daily_close_drawdown':episode([(v['day'],v['equity']) for v in daily]),
        'intraday_drawdown':episode(equity),'months':bymonth,
        'win_rate':result['metrics']['win_rate'],'profit_factor':result['metrics']['profit_factor'],
        'net_gains':result['metrics']['net_gains'],'net_losses':result['metrics']['net_losses'],
        'risk_proxy_pct':{'median':float(np.median(risks)),'maximum':max(risks)},
        'realized_position_scenario_overshoots':budget_overshoots,
        'daily_returns':daily_returns,'daily_dates':[row['day'] for row in daily],
        'folder':str(folder),'completion_sha256':n.sha(folder/'completion.json')}


def matched(oldfolder,newfolder):
    key=lambda x:(x['symbol'],x['entry_decision_time'])
    old={key(x):x for x in n.read(oldfolder/'trades.json')}
    new={key(x):x for x in n.read(newfolder/'trades.json')}
    matches=[]
    for k in sorted(old.keys()&new.keys()):
        a,z=old[k],new[k];au=a['net_pnl']/a['qty'];zu=z['net_pnl']/z['qty']
        matches.append({'symbol':k[0],'entry_decision_time':k[1], 'old_qty':a['qty'],'new_qty':z['qty'],
            'old_pnl':a['net_pnl'],'new_pnl':z['net_pnl'],'pnl_delta':z['net_pnl']-a['net_pnl'],
            'per_unit_execution_effect':a['qty']*(zu-au),'quantity_effect':(z['qty']-a['qty'])*zu,
            'old_unit_pnl':au,'new_unit_pnl':zu,'old_exit':a['exit_fill_time'],'new_exit':z['exit_fill_time'],
            'old_exit_reason':a['exit_reason'],'new_exit_reason':z['exit_reason'],
            'old_entry_price':a['entry_price'],'new_entry_price':z['entry_price'],
            'old_exit_price':a['exit_price'],'new_exit_price':z['exit_price']})
    added=[new[k] for k in new.keys()-old.keys()];removed=[old[k] for k in old.keys()-new.keys()]
    unit=sum(x['per_unit_execution_effect'] for x in matches)
    quantity=sum(x['quantity_effect'] for x in matches)
    turnover=sum(x['net_pnl'] for x in added)-sum(x['net_pnl'] for x in removed)
    delta=sum(x['net_pnl'] for x in new.values())-sum(x['net_pnl'] for x in old.values())
    assert abs(unit+quantity+turnover-delta)<1e-5
    winners=[x for x in matches if x['old_pnl']>0]
    largest=sorted(old.values(),key=lambda x:x['net_pnl'],reverse=True)[:5]
    return {'matched':len(matches),'new_count':len(added),'removed_count':len(removed),
        'pnl_delta':delta,'per_unit_execution_effect':unit,'quantity_effect':quantity,'new_minus_removed_effect':turnover,
        'matched_winners':len(winners),'winner_to_loser':sum(x['new_pnl']<=0 for x in winners),
        'winner_exit_changes':sum(x['new_exit']!=x['old_exit'] or abs(x['new_exit_price']-x['old_exit_price'])>1e-6 for x in winners),
        'matched_winner_pnl_retention_pct':100*sum(x['new_pnl'] for x in winners)/sum(x['old_pnl'] for x in winners),
        'top_five_original_winner_pnl_retention_pct':100*sum(new.get(key(x),{}).get('net_pnl',0) for x in largest)/sum(x['net_pnl'] for x in largest),
        'all_matches':matches,'added':added,'removed':removed,
        'interpretation':'Exact accounting decomposition. Per-unit execution includes both entry and exit fill changes; quantity and opportunity paths respond to cash, capacity and compounding.'}


def bootstrap(old,new):
    rng=np.random.default_rng(1729);old=np.array(old);new=np.array(new);count=len(old)
    trials=[]
    for _ in range(4000):
        starts=rng.integers(0,count,size=(count+4)//5)
        indices=np.concatenate([(start+np.arange(5))%count for start in starts])[:count]
        trials.append(100*(np.prod(1+new[indices])-np.prod(1+old[indices])))
    return {'block_sessions':5,'replications':4000,'return_gain_pp_percentiles':{
        str(q):float(np.percentile(trials,q)) for q in (2.5,50,97.5)},
        'fraction_positive':float(np.mean(np.array(trials)>0)),
        'interpretation':'Paired circular-block resampling of development-period daily returns. Conditional sensitivity only; excludes strategy selection bias and does not prove future alpha.'}


def collect():
    t.activate();catalog=n.read(t.ROOT/'candidate_plan.json')['candidates'];control=catalog[0]
    data={'created_utc':s.utc(),'registration_sha256':n.sha(t.ROOT/'registration.json'),'candidates':[]}
    original={w:s.PARENT/'cases'/s.forensic.BASE/w/'100000000_baseline' for w in ('train','oos')}
    data['original_saved']={w:{**describe(original[w]),'authentication':s.forensic.authenticate(original[w])} for w in original}
    stress_original={w:s.PRIOR/'cases'/s.forensic.BASE/w/'100000000_combined' for w in ('train','oos')}
    data['original_stress']={w:{**describe(folder),'authentication':s.forensic.authenticate(folder)}
        for w,folder in stress_original.items()}
    for c in catalog:
        row={'id':c['id'],'label':c['label'],'mutations':c['mutations'],'windows':{},'stress':{},'capped':{}}
        for scenario,capped,target in [('baseline',False,'windows'),('combined',False,'stress'),('baseline',True,'capped')]:
            for w in ('train','oos'):
                folder=b.output(c,w,scenario,capped)
                if not (folder/'assessment.json').exists():continue
                proof=history.audit(c,w,scenario,capped)
                row[target][w]={**describe(folder),'authentication':proof}
        if len(row['windows'])==2:
            refs={w:describe(b.output(control,w)) for w in ('train','oos')}
            assert all(refs[w]['daily_dates']==row['windows'][w]['daily_dates'] for w in ('train','oos'))
            a,z=row['windows']['train'],row['windows']['oos']
            row['gates']={'is_guard_return':a['return_pct']>=refs['train']['return_pct']-3,
                'is_original_return':a['return_pct']>=data['original_saved']['train']['return_pct']-3,
                'is_drawdown':a['drawdown_pct']<=refs['train']['drawdown_pct']+2,
                'oos_return':z['return_pct']>refs['oos']['return_pct'],
                'oos_drawdown_target':z['drawdown_pct']<=refs['oos']['drawdown_pct']-2}
            row['nominal_qualified']=all(row['gates'].values())
            row['trade_comparison']={w:matched(b.output(control,w),b.output(c,w)) for w in ('train','oos')}
            row['monthly_comparison']={w:{month:v['return_pct']-refs[w]['months'][month]['return_pct']
                for month,v in row['windows'][w]['months'].items()} for w in ('train','oos')}
            row['bootstrap']={w:bootstrap(refs[w]['daily_returns'],row['windows'][w]['daily_returns']) for w in ('train','oos')}
        if len(row['stress'])==2 and all((b.output(control,w,'combined')/'assessment.json').exists() for w in ('train','oos')):
            ref={w:n.read(b.output(control,w,'combined')/'result.json') for w in ('train','oos')}
            orig=data['original_stress']['train']
            a,z=row['stress']['train'],row['stress']['oos']
            row['stress_gates']={'is_guard_return':a['return_pct']>=ref['train']['return_pct']-3,
                'is_original_return':a['return_pct']>=orig['return_pct']-3,
                'oos_positive':z['return_pct']>0,'oos_return':z['return_pct']>=ref['oos']['return_pct'],
                'is_drawdown':a['drawdown_pct']<=ref['train']['drawdown_pct']+2,
                'oos_drawdown':z['drawdown_pct']<=ref['oos']['drawdown_pct']}
            row['stress_qualified']=all(row['stress_gates'].values())
        data['candidates'].append(row)
    bylabel={row['label']:row for row in data['candidates']}
    priority=bylabel['auction_priority']
    for row in data['candidates']:
        if row['label'] in ('guard_control','auction_priority') or len(row['windows'])!=2:continue
        row['budget_increment_vs_priority']={w:{
            'return_gain_pp':row['windows'][w]['return_pct']-priority['windows'][w]['return_pct'],
            'dd_change_pp':row['windows'][w]['drawdown_pct']-priority['windows'][w]['drawdown_pct'],
            'trade_change':row['windows'][w]['trades']-priority['windows'][w]['trades']}
            for w in ('train','oos')}
    five,six=bylabel['priority_aggressive_budget'],bylabel['aggressive_budget_six_slots']
    if len(five['windows'])==len(six['windows'])==2:
        snapshots={}
        for w in ('train','oos'):
            def payload(cid):
                study=n.read(t.ROOT/'registrations'/w/(cid+'_100000000')/'study.json')
                return n.read(Path(study['code'])/study['olr_snapshots'])
            def economics(rows):
                return {row['trade_date']:[{k:v for k,v in c.items() if k not in (
                    'metadata','source_fingerprint','source_artifact_hash')} for c in row['candidates']] for row in rows}
            before,after=payload(five['id']),payload(six['id'])
            assert economics(before)==economics(after),'Slot experiment changed candidate economics'
            snapshots[w]={'all_candidate_economics_identical':True,'snapshot_dates':len(before),
                'five_slot_hash':n.sig(before),'six_slot_hash':n.sig(after)}
        n.write(t.OUT/'sixth_slot_snapshot_parity.json',snapshots)
        six['sixth_slot_increment']={w:{
            'return_gain_pp':six['windows'][w]['return_pct']-five['windows'][w]['return_pct'],
            'dd_change_pp':six['windows'][w]['drawdown_pct']-five['windows'][w]['drawdown_pct'],
            'trade_change':six['windows'][w]['trades']-five['windows'][w]['trades'],
            'decomposition':matched(Path(five['windows'][w]['folder']),Path(six['windows'][w]['folder']))}
            for w in ('train','oos')}
    n.write(t.OUT/'results.json',data)
    print([{ 'label':r['label'], **{k:{w:{f:v[f] for f in ('return_pct','drawdown_pct','trades')} for w,v in r[k].items()}
        for k in ('windows','stress','capped')}, 'gates':r.get('gates'),'stress_gates':r.get('stress_gates')}
        for r in data['candidates']],flush=True)
    return data


if __name__=='__main__':collect()

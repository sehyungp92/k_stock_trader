"""Post-freeze reproduction, conditional references and explicit coverage reporting."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
from datetime import datetime,timedelta
from pathlib import Path
import csv,json,shutil
import numpy as np
from backtests.auto.shared.cache_keys import stable_signature
from .round3_native_io_v1 import ROOT,read,write,bound
from .round3_native_tasks_v1 import execute
from .round3_analysis_v1 import account
from .round3_contract_v1 import verify_acceptance
from .round3_integrity_v1 import rows
from .round3_evidence_v1 import authenticate

def saved_profile(service,name,period,optimized):
    parent=service.profiles[period][name+'/100000000'];dest=service.out/'saved_config_reproduction/inputs'/period
    if (dest/'profile.json').exists():return read(dest/'profile.json')
    config=read(optimized)['mutations']
    if config!=service.arms[name]['config']:raise ValueError('Saved optimized settings changed')
    dest.mkdir(parents=True,exist_ok=False);write(dest/'config.json',config)
    plan=deepcopy(read(parent['plan']['path']));plan['input_sha256'].pop(plan['config']);plan['config']=str(dest/'config.json');plan['input_sha256'][plan['config']]=bound(plan['config'])['sha256']
    write(dest/'plan.json',plan);mp=Path(parent['manifest']['path']);payload=read(mp);payload.pop('manifest_hash');code=mp.parent/'files'
    for n in ('config.json','plan.json'):
        rel='research/c17/saved_config_reproduction/'+n;target=code/rel;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write((dest/n).read_bytes())
        payload['files'][rel]=bound(target)['sha256']
    payload['evidence']['saved_configuration_reproduction']=dict(optimized=bound(optimized),parent=name,registration=bound(service.registration))
    manifest=mp.parent/'saved_config_reproduction.manifest.json';write(manifest,{**payload,'manifest_hash':stable_signature(payload)})
    item={**parent,'plan':bound(dest/'plan.json'),'manifest':bound(manifest)};write(dest/'profile.json',item);return item

def reproduce(service,name,canonical):
    target=service.out/'saved_config_reproduction/completion.json'
    if target.exists():
        proof=read(target)
        for r in proof['receipts'].values():verify_acceptance(r)
        if bound(canonical/'optimized_config.json')['sha256']!=proof['optimized']['sha256']:raise ValueError('Saved config changed after reproduction')
        return proof
    jobs=[]
    for p,s in [('is','baseline'),('oos','combined')]:
        item=saved_profile(service,name,p,canonical/'optimized_config.json');folder=service.out/'saved_config_reproduction/accounts'/p
        jobs.append((p,s,100000000,item,folder))
    service._reserve(jobs);receipts={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(execute,item,s,folder,bridge=service.receipts(name)[p+'/'+s]['folder']):(p,s) for p,s,c,item,folder in jobs}
        for future in as_completed(futures):
            p,s=futures[future];r=future.result()
            if not r.get('passed'):raise ValueError('Saved configuration failed full-path reproduction')
            receipts[p+'/'+s]=r
    # All other reported scenarios used the exact same explicit settings plus their declared execution stress.
    cfg=read(canonical/'optimized_config.json')['mutations']
    for key,r in service.receipts(name).items():
        effective=read(Path(r['folder'])/'request.json')['effective_config']
        for k,v in cfg.items():
            if k in ('slippage_bps','fast_replay_suppress_rejections'):continue
            if effective[k]!=v:raise ValueError('Reported scenario did not use saved settings')
    result=dict(passed=True,optimized=bound(canonical/'optimized_config.json'),receipts=receipts,
        all_eight_reported_effective_settings_checked=True,full_ordered_path_reproduced=['is/baseline','oos/combined'],max_workers=2)
    write(target,result);return result

def conditional_reference(receipt):
    """Dependent date-block intervals of reused-history averages, never forward guarantees."""
    verify_acceptance(receipt);folder=Path(receipt['folder']);daily=read(folder/'daily.json');trades=read(folder/'trades.json');cells=account(receipt)
    groups=defaultdict(list)
    for t in trades:groups[t['entry_fill_time'][:10]].append(t)
    useful=defaultdict(int)
    for d,s in cells['useful_issuer_days']:useful[d]+=1
    vectors=np.array([[len(groups[d['day']]),sum(t['net_pnl']>0 for t in groups[d['day']]),sum(t['net_pnl'] for t in groups[d['day']]),useful[d['day']]] for d in daily],float)
    n=len(daily);point=vectors.sum(axis=0)
    def metrics(x):
        return dict(lifetimes_per20_sessions=20*x[0]/n,useful_issuer_days_per20_sessions=20*x[3]/n,
            lifetime_win_rate=x[1]/x[0] if x[0] else None,net_expectancy_krw_per_lifetime=x[2]/x[0] if x[0] else None)
    uncertainty={}
    for length in (5,10,20):
        rng=np.random.default_rng(41017+length+n);starts=rng.integers(0,n,(4000,int(np.ceil(n/length))))
        idx=((starts[:,:,None]+np.arange(length))%n).reshape(4000,-1)[:,:n];totals=vectors[idx].sum(axis=1)
        samples=[metrics(v) for v in totals]
        uncertainty[str(length)]={k:np.quantile([s[k] for s in samples if s[k] is not None],[.025,.975]).tolist() for k in metrics(point)}
    from kis_core.trading_calendar import get_trading_calendar
    clock=get_trading_calendar();gross_hours=0.;position_hours=0.;duration=0.;maxpositions=0;maxgross=0
    for st in rows(folder/'account_states.jsonl.gz'):
        at=datetime.fromisoformat(st['timestamp']);opening,closing=clock.session_hours(at.date())
        if not opening<=at.time().replace(tzinfo=None)<closing:continue
        hours=min(5,(datetime.combine(at.date(),closing,at.tzinfo)-at).total_seconds()/60)/60
        q=st['inventory'];gross=sum(v*st['marks'][s] for s,v in q.items());ratio=gross/st['equity']
        gross_hours+=ratio*hours;position_hours+=len(q)*hours;duration+=hours;maxpositions=max(maxpositions,len(q));maxgross=max(maxgross,ratio)
    return dict(point=metrics(point),conditional_average_CI95=uncertainty,observed_native_MTM_DD_pct=cells['mtm_dd_pp'],
        observed_mean_gross_equity=gross_hours/duration,observed_position_hours=position_hours,observed_session_hours=duration,
        observed_max_positions=maxpositions,observed_max_gross_equity=maxgross,
        scope='Native five-minute end-of-batch marks; time averages assign each completed batch to its five-minute interval. Conditional reused-history means, not a predictive paper-trading interval or safe forward DD bound. No financing charges, queue/latency or broker restart qualification.',
        source=bound(folder/'result.json'))

def lifetime_contrasts(base,candidate):
    def groups(folder):
        result=defaultdict(list)
        for t in read(Path(folder)/'trades.json'):
            result[(t['symbol'],t['entry_decision_time'],t['route_metadata'].get('entry_route'))].append(t)
        return result
    b,c=groups(base),groups(candidate);result=[]
    for key in set(b)&set(c):
        if len(b[key])!=1 or len(c[key])!=1:continue
        x,y=b[key][0],c[key][0];qb,qc=x['qty'],y['qty'];rb,rc=x['net_pnl']/qb,y['net_pnl']/qc
        quantity=(qc-qb)*rb;payoff=qb*(rc-rb);interaction=(qc-qb)*(rc-rb)
        result.append(dict(symbol=key[0],entry_decision=key[1],route=key[2],baseline_net=x['net_pnl'],candidate_net=y['net_pnl'],
            delta=y['net_pnl']-x['net_pnl'],baseline_qty=qb,candidate_qty=qc,quantity_effect=quantity,
            per_share_lifecycle_effect=payoff,interaction=interaction,
            residual=y['net_pnl']-x['net_pnl']-quantity-payoff-interaction,
            baseline_exit=x.get('exit_fill_time'),candidate_exit=y.get('exit_fill_time'),
            baseline_entry_price=x['entry_price'],candidate_entry_price=y['entry_price']))
    return sorted(result,key=lambda x:x['delta'])

def finish(service,state,canonical):
    canonical=Path(canonical);name=service.name_for_config(service.canonical_config if hasattr(service,'canonical_config') else state.cumulative_mutations)
    service.final_freeze(name);reproduction=reproduce(service,name,canonical)
    summaries=[];episodes={};paths={};evaluations={}
    for n in service.arms:
        path=service.out/'evaluations'/(n+'.json')
        if not path.exists():continue
        e=authenticate(path);evaluations[n]=e;stat=e.get('statistics',{}).get('5',{});score=e.get('score',{})
        summaries.append(dict(arm=n,native_valid=e['native_valid'],selected=n==name,selectable=service.arms[n]['selectable'],
            R=score.get('R'),D=score.get('D'),S=score.get('S'),gain_pp=e.get('qualification',{}).get('gain_pp'),
            lower95=stat.get('simultaneous_Bonferroni_CI95_pp',[None,None])[0],upper95=stat.get('simultaneous_Bonferroni_CI95_pp',[None,None])[1],MDE_pp=stat.get('planning_MDE_pp'),
            promoted=e.get('qualification',{}).get('promote',False),failures=';'.join(e.get('qualification',{}).get('failures',[])),evidence=str(path)))
        if e.get('native_valid'):
            episodes[n]={k:lifetime_contrasts(service.control_receipts[k]['folder'],r['folder']) for k,r in service.receipts(n).items()}
            paths[n]=[account(r)['economic_path_hash'] for _,r in sorted(service.receipts(n).items())]
    table=service.out/'trial_summary.csv'
    with table.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    write(service.out/'lifetime_contrasts.json',dict(scope='Matched native lifetimes. Per-share payoff includes entry timing, management, exits and costs; quantities and payoff interact. Not a standalone oracle.',arms=episodes))
    unique={};aliases={}
    for n,path in paths.items():
        identity=stable_signature(path)
        if identity in unique:aliases[n]=unique[identity]
        else:unique[identity]=n
    write(service.out/'economic_path_deduplication.json',dict(unique=unique,aliases=aliases,receipts_preserved=True,market_samples_not_increased=True))
    reference={k:conditional_reference(r) for k,r in service.receipts(name).items()};write(service.out/'conditional_paper_reference.json',reference)
    write(service.out/'holdout_access_log.json',dict(untouched_available=False,accesses=0,reason='All225 retained sessions consumed before C17. No new data acquired or inspected as a holdout.',final_freeze=bound(service.out/'final_freeze.json')))
    result=dict(process_completed=True,selected=name,new_economic_promotion=name!='v6_h0',implementation_valid=True,
        full_saved_config_reproduction=reproduction['passed'],completed_phases=state.completed_phases,
        central_coverage_incomplete=['All-rejected native labels','Globally funded S/E/SE oracles','Matched-entry placebo','Untouched holdout'],
        full_round_diagnostics=bound(canonical/'round_final_diagnostics.txt'),trial_summary=bound(table),conditional_reference=bound(service.out/'conditional_paper_reference.json'),
        paper_or_live_activated=False,remaining=['Human-readable interpretation and current/experiment/lesson/decision record closeout'])
    write(service.out/'closeout.json',result);return result

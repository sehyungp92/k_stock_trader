"""Registered allocation/risk challenge on frozen, reused portfolio data.

No learned trade filter, seed selection, strategy lifecycle change or live route.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.portfolio_native_budget_study import freeze, read, one, checked
from backtests.strategies.kalcb.executed_value import digest, write_json

OLD=Path.home()/'.codex/research/portfolio_native_budget_20261009'
BATTERY={'baseline':('baseline',{}),'cost':('cost',{}),'participation':('participation',{}),
    'combined':('combined',{}),'auction_price_baseline':('baseline',{'auction_adverse_bps':15}),
    'auction_price_combined':('combined',{'auction_adverse_bps':15}),
    'auction_nonfill_baseline':('baseline',{'auction_nonfill_rate':.5}),
    'auction_nonfill_combined':('combined',{'auction_nonfill_rate':.5})}
PROTOCOL={
    'version':'portfolio-allocation-challenge-v1', 'capital':25_000_000,
    'scope':'Reused development, not independent performance. Frozen native strategy cores and schedule.',
    'budget_grid':[[.75,.75],[.5,1.],[.5,.75]],
    'allocation_modes':['symbol','native_rank','proportional'], 'diagnostic_random_seeds':[11,29],
    'risk':{'baseline_max_drawdown_pct':11.,'all_adverse_max_drawdown_pct':12.5,
        'max_overnight_nav_fraction':.80,'max_mean_overnight_nav_fraction':.30,
        'worst_session_loss_pct':4.5,'frequency_floor':515},
    'risk_basis':'Retain original DD limits across the entire battery; at most 80% overnight capital (5pp headroom to the incumbent), mean 30%, single-session loss 4.5%; observed development gates, not future loss guarantees. Frequency floor retains at least 98% of incumbent activity.',
    'selection':'Exclude random seeds. Require every battery scenario to pass risk and frequency. Maximize minimum adverse-scenario net return; within 1pp prefer minimum entry count, then smaller maximum DD. Native-rank/proportional promotion requires >=1pp better minimum adverse return than same-budget symbol, no >0.25pp worse max DD, and no lower minimum frequency. Otherwise retain symbol. A risk-feasible budget may replace an incumbent that fails the same battery; disclose return sacrifice.',
    'allocation_first_stage':'Native-rank/proportional and two fixed seeds at incumbent budget, baseline/combined. Qualify both deterministic modes on full battery only if baseline and combined returns each exceed symbol by >=1pp, neither DD worsens >0.25pp, and minimum frequency does not decline. No rich EV model on five conflict sessions.',
    'local_refinement':{'trigger':'Incumbent .75/.75 fails full battery, .5/.75 passes, and no deterministic allocation rule qualifies in first stage.',
        'grid':[[.625,.75],[.6875,.75]],'mode':'symbol','rule':'Same eight-scenario gates; no further grid expansion.'},
    'qualification':'All eight scenarios of proposed winner require full source/account audit and identical trades/daily/equity to screens before research promotion.',
    'partition_benchmark':'Two separately funded 12.5m native accounts, KALCB and OLR, baseline/combined; sum synchronized marked NAV, not scaled standalone returns. Verify aggregate same-phase participation before treating as executable combined benchmark.',
    'inventory':'Report delayed-liquidation contributions and existing-position suppression. No multi-lot renewal or retrospective next-day demand oracle; lifecycle changes require separate evidence.',
    'chronology':'Report continuous-account calendar blocks and paired session-block uncertainty. Reused splits cannot establish untouched out-of-sample validity. No fitting on unmatured outcomes.',
    'no_live_deployment':True}

OVERNIGHT_PROTOCOL={**copy.deepcopy(PROTOCOL),
    'version':'portfolio-overnight-frontier-v1',
    'scope':'Adaptive, prospectively registered follow-up after the original finite challenge failed its unchanged overnight-exposure gates. All data remain reused development.',
    'budget_grid':[[.75,.5],[.75,.625],[.875,.5],[.875,.625]],
    'allocation_modes':['symbol'],'diagnostic_random_seeds':[],
    'allocation_first_stage':'The original native-rank and proportional challenges failed joint return/frequency promotion; retain symbol ordering for this funding-budget follow-up.',
    'local_refinement':{'trigger':'For each OLR fraction, K=.75 passes every battery gate and K=.875 fails at least one. Then test K=.8125 at that same OLR fraction.',
        'midpoint':.8125,'maximum_additional_candidates':2,'rule':'Same risk/frequency gates and eight scenarios; no further extension.'},
    'reason':'Both original O=.75 budgets exceed the mean overnight-capital limit under nonfills; O=1 exceeds maximum overnight exposure. Reduce the component actually tying up overnight capital, then test a bounded amount of intraday KALCB risk.'}


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    pointer=REPO/'config/research/portfolio_synergy_selected.json'
    profile=Path(read(pointer)['profile']['path'])
    protocol={**PROTOCOL,'registered_utc':datetime.now(timezone.utc).isoformat(),
        'incumbent_pointer':{'path':str(pointer),'sha256':digest(pointer)},
        'incumbent_profile':{'path':str(profile),'sha256':digest(profile)},
        'critical_review_sha256':digest(REPO/'docs/research/portfolio_synergy_critical_review_20261009.txt'),
        'driver_sha256':digest(Path(__file__))}
    write_json(root/'protocol.json',protocol)
    base=read(OLD/'priority_challenge/finalist.json')['recipe']
    recipes={};jobs=[]
    def add(name,k,o,mode='symbol',seed=0,battery=None):
        for label,(scenario,stress) in (battery or BATTERY).items():
            suffix='_price' if 'auction_price' in label else '_nonfill' if 'auction_nonfill' in label else ''
            recipe_name=name+suffix
            r=copy.deepcopy(base);r['native_budget'].update(kalcb_sizing_fraction=k,olr_fraction=o,
                allocation_mode=mode,allocation_seed=seed)
            if stress:r['olr_execution_stress']=stress
            recipes[recipe_name]=r
            jobs.append({'recipe':recipe_name,'scenario':scenario,'group':'risk' if mode=='symbol' else 'allocation',
                'candidate':name,'battery':label,'diagnostic_only':mode=='random'})
    for k,o in PROTOCOL['budget_grid']:add(f'k{int(k*100):03}_o{int(o*100):03}',k,o)
    for mode,seed in [('native_rank',0),('proportional',0),('random',11),('random',29)]:
        add(mode+(str(seed) if seed else ''),.75,.75,mode,seed,{k:BATTERY[k] for k in ('baseline','combined')})
    freeze(OLD/'priority_challenge/study.json',root/'mechanisms',recipes,jobs,registration={
        'protocol':{'path':str(root/'protocol.json'),'sha256':digest(root/'protocol.json')},
        'purpose':'Preregistered mechanical allocation and full adverse-execution budget challenge.'})


def run(root,group='all',full=False,workers=1):
    from concurrent.futures import ThreadPoolExecutor,as_completed
    registration=read(root/'registration.json')
    protocol=registration.get('protocol')
    if protocol and digest(Path(protocol['path']))!=protocol['sha256']:raise ValueError('Changed protocol')
    jobs=[j for j in registration['jobs'] if group=='all' or j['group']==group
          or (group.startswith('candidate:') and j.get('candidate')==group.split(':',1)[1])]
    if not jobs:raise ValueError('No registered jobs match '+group)
    errors={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks={pool.submit(one,root,j,full,'qualified' if full else 'cases'):j for j in jobs}
        for future in as_completed(tasks):
            try:future.result()
            except Exception as e:
                errors[tasks[future]['recipe']+'_'+tasks[future]['scenario']]=str(e)
                print(json.dumps({'error':str(e)}),flush=True)
    write_json(root/('run_'+group.replace(':','_')+('_full' if full else '')+'.json'),{'jobs':jobs,'errors':errors})
    if errors:raise RuntimeError('Research failures preserved; no selection permitted')


def partition(root):
    """Ring-fenced 12.5m books share one execution engine and its volume cap."""
    parent=root/'mechanisms/study.json';base=read(parent)['candidates']['k075_o075']
    output=root/('partition_cash_funded' if (root/'partition_v2').exists() else 'partition_v2' if (root/'partition').exists() else 'partition')
    recipes={};jobs=[]
    for mode in ('shared','partition','daily_rebalanced'):
        r=copy.deepcopy(base);r['native_budget'].update(capital_mode=mode,kalcb_capital_fraction=.5,
            kalcb_sizing_fraction=.5 if mode=='shared' else 1.,olr_fraction=.5 if mode=='shared' else 1.,
            kalcb_budget_fraction=1. if mode=='shared' else 2.)
        recipes[mode]=r
        jobs.extend({'recipe':mode,'scenario':s,'group':'partition'} for s in ('baseline','combined'))
    freeze(parent,output,recipes,jobs,registration={
        'protocol':{'path':str(root/'protocol.json'),'sha256':digest(root/'protocol.json')},
        'partition_implementation':{'registered_utc':datetime.now(timezone.utc).isoformat(),
            'starting_wealth':{'KALCB':12_500_000,'OLR':12_500_000},
            'execution':'One physical broker with separate sleeve cash/marked-wealth books; all orders share issuer, sector, gross and same-phase participation constraints. This avoids double use of market volume by separately replayed accounts.',
            'transfer':'partition retains each sleeves own profits/losses; daily_rebalanced moves toward 50/50 before the first opening mark, using previous observed prices and only positive actual donor cash. Invested capital and unrealized gains cannot fund the transfer.',
            'comparison':'shared .5/.5 has identical initial sizing but both later size from total portfolio NAV; no summation of separately funded 25m results.',
            'initial_holding_budgets':{'KALCB':25_000_000,'OLR':12_500_000},
            'pre_run_correction':'The initial unused partition freeze gave shared KALCB a 2*NAV holding budget versus the partitioned initial 1*NAV. This generation matches both initial holding budgets; no partition outcomes were observed before correction.',
            'cash_transfer_correction':{'path':str(root/'partition_cash_transfer_correction.json'),
                'sha256':digest(root/'partition_cash_transfer_correction.json')} if (root/'partition_cash_transfer_correction.json').exists() else None,
            'candidate_eligibility':'Mechanism diagnostic only; not eligible for this round selection.'}})
    write_json(root/'partition_study.json',{'path':str(output/'study.json'),'sha256':digest(output/'study.json')})


def risk_metrics(case):
    import numpy as np
    r=checked(case);daily=read(case/'daily.json')
    nav=np.array([25_000_000]+[d['equity'] for d in daily]);returns=100*(nav[1:]/nav[:-1]-1)
    overnight=[max(0.,(d['equity']-d['cash'])/d['equity']) for d in daily]
    return {'return_pct':r['return_pct'],'drawdown_pct':r['drawdown_pct'],'trades':r['trades'],
        'worst_session_pct':float(min(returns)), 'worst_decile_mean_pct':float(np.mean(np.sort(returns)[:max(1,int(np.ceil(len(returns)*.1)))])),
        'overnight_max':max(overnight),'overnight_mean':float(np.mean(overnight)),
        'completion':{'path':str(case/'completion.json'),'sha256':digest(case/'completion.json')}}


def passes(metrics,label,protocol):
    risk=protocol['risk'];m=metrics
    cap=risk['baseline_max_drawdown_pct'] if label=='baseline' else risk['all_adverse_max_drawdown_pct']
    tests={'drawdown':m['drawdown_pct']<=cap,'frequency':m['trades']>=risk['frequency_floor'],
        'overnight_max':m['overnight_max']<=risk['max_overnight_nav_fraction'],
        'overnight_mean':m['overnight_mean']<=risk['max_mean_overnight_nav_fraction'],
        'worst_session':m['worst_session_pct']>=-risk['worst_session_loss_pct']}
    return {'pass':all(tests.values()),'tests':tests}


def collect(root,partial=False):
    protocol=read(root/'protocol.json')
    if protocol['version']==OVERNIGHT_PROTOCOL['version']:
        studies=['coarse']
        if (root/'midpoint/registration.json').exists():studies.append('midpoint')
    else:
        studies=['mechanisms']
        if (root/'refinement/registration.json').exists():studies.append('refinement')
        if (root/'allocation_battery/registration.json').exists():studies.append('allocation_battery')
    candidates={}
    for child in studies:
        reg=read(root/child/'registration.json')
        study=read(root/child/'study.json')
        if digest(Path(reg['protocol']['path']))!=reg['protocol']['sha256']:raise ValueError('Changed protocol')
        for j in reg['jobs']:
            case=root/child/'cases'/(j['recipe']+'_'+j['scenario'])
            if partial and not (case/'completion.json').exists():continue
            m=risk_metrics(case);m['gate']=passes(m,j['battery'],protocol)
            m['case']=str(case);m['study']=str(root/child/'study.json');m['recipe']=j['recipe'];m['scenario']=j['scenario']
            m['allocation_mode']=study['candidates'][j['recipe']]['native_budget'].get('allocation_mode','symbol')
            m['diagnostic_only']=j.get('diagnostic_only',False)
            candidates.setdefault(j['candidate'],{})[j['battery']]=m
    return candidates


def refinement(root):
    candidates=collect(root,partial=True)
    for name,labels in [('k075_o075',set(BATTERY)),('k050_o075',set(BATTERY)),
                        *[(n,{'baseline','combined'}) for n in ('native_rank','proportional','random11','random29')]]:
        if not labels.issubset(candidates.get(name,{})):raise ValueError('Stage decision awaits registered cases for '+name)
    incumbent=candidates['k075_o075'];qualifying=[]
    for mode in ('native_rank','proportional'):
        a=candidates[mode]
        if (all(a[s]['return_pct']>=incumbent[s]['return_pct']+1 and a[s]['drawdown_pct']<=incumbent[s]['drawdown_pct']+.25 for s in ('baseline','combined'))
            and min(x['trades'] for x in a.values())>=min(incumbent[s]['trades'] for s in ('baseline','combined'))):
            qualifying.append(mode)
    trigger=(not all(x['gate']['pass'] for x in incumbent.values())
        and all(x['gate']['pass'] for x in candidates['k050_o075'].values()) and not qualifying)
    decision={'allocation_qualifiers':qualifying,'local_refinement_triggered':trigger,
        'inputs':{n:{s:m['completion'] for s,m in d.items()} for n,d in candidates.items()}}
    write_json(root/'stage_decision.json',decision)
    recipes={};jobs=[]
    for name,k,o,mode in ([(f'k{k*100:g}_o075',k,o,'symbol') for k,o in PROTOCOL['local_refinement']['grid']] if trigger else
                         [(m,.75,.75,m) for m in qualifying]):
        for label,(scenario,stress) in BATTERY.items():
            suffix='_price' if 'auction_price' in label else '_nonfill' if 'auction_nonfill' in label else ''
            r=copy.deepcopy(read(root/'mechanisms/study.json')['candidates']['k075_o075'])
            r['native_budget'].update(kalcb_sizing_fraction=k,olr_fraction=o,allocation_mode=mode)
            if stress:r['olr_execution_stress']=stress
            recipes[name+suffix]=r
            jobs.append({'recipe':name+suffix,'scenario':scenario,'candidate':name,'battery':label,'group':'refinement'})
    if recipes:
        freeze(root/'mechanisms/study.json',root/('refinement' if trigger else 'allocation_battery'),recipes,jobs,overlay=False,
            registration={'protocol':{'path':str(root/'protocol.json'),'sha256':digest(root/'protocol.json')},
                'stage_decision':{'path':str(root/'stage_decision.json'),'sha256':digest(root/'stage_decision.json')}})
    print(json.dumps({k:decision[k] for k in ('allocation_qualifiers','local_refinement_triggered')}),flush=True)


def choose(candidates):
    """Fixed rule: quality claims cannot excuse less activity or failed tails."""
    feasible={}
    for name,battery in candidates.items():
        if (name.startswith('random') or set(battery)!=set(BATTERY)
            or any(m.get('diagnostic_only') or m.get('allocation_mode')=='random' for m in battery.values())):continue
        if not all(m['gate']['pass'] for m in battery.values()):continue
        if name in ('native_rank','proportional') or any(m.get('allocation_mode') in ('native_rank','proportional') for m in battery.values()):
            old=candidates['k075_o075']
            if (min(m['return_pct'] for s,m in battery.items() if s!='baseline')<min(m['return_pct'] for s,m in old.items() if s!='baseline')+1
                or max(m['drawdown_pct'] for m in battery.values())>max(m['drawdown_pct'] for m in old.values())+.25
                or min(m['trades'] for m in battery.values())<min(m['trades'] for m in old.values())):continue
        feasible[name]=battery
    scores={n:min(m['return_pct'] for s,m in b.items() if s!='baseline') for n,b in feasible.items()}
    best=max(scores.values(),default=0)
    tied=[n for n,v in scores.items() if v>=best-1]
    winner=max(tied,key=lambda n:(min(m['trades'] for m in feasible[n].values()),
        -max(m['drawdown_pct'] for m in feasible[n].values()),scores[n],n)) if tied else None
    return winner,list(feasible),scores


def select(root,save=True):
    candidates=collect(root);winner,feasible,scores=choose(candidates)
    result={'version':'portfolio-allocation-selection-v1','winner':winner,'feasible':list(feasible),
        'adverse_return_floor':scores,'candidates':candidates,
        'protocol':{'path':str(root/'protocol.json'),'sha256':digest(root/'protocol.json')},
        'independent_performance_established':False,'live_deployment_enabled':False}
    if save:
        write_json(root/'selection.json',result)
        print(json.dumps({k:result[k] for k in ('winner','feasible','adverse_return_floor')}),flush=True)
    return result


def qualify(root,workers=1):
    selection=read(root/'selection.json');name=selection['winner']
    if not name:raise ValueError('No risk-feasible winner')
    from concurrent.futures import ThreadPoolExecutor
    def qualified(m):
        study_root=Path(m['study']).parent
        one(study_root,{'recipe':m['recipe'],'scenario':m['scenario']},True,'qualified')
        new=study_root/'qualified'/(m['recipe']+'_'+m['scenario'])
        for file in ('trades.json','daily.json','equity.json'):
            if digest(new/file)!=digest(Path(m['case'])/file):raise ValueError('Full qualification changed economic path')
        return str(new)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        paths=list(pool.map(qualified,selection['candidates'][name].values()))
    write_json(root/'qualification.json',{'winner':name,'cases':paths,'selection_sha256':digest(root/'selection.json')})


def pipeline(root,workers=1):
    """Resume the registered dependent stages while initial screen workers run."""
    import time
    def status(stage,**extra):
        value={'stage':stage,'utc':datetime.now(timezone.utc).isoformat(),**extra}
        write_json(root/'pipeline_progress.json',value);print(json.dumps(value),flush=True)
    try:
        required=[j for j in read(root/'mechanisms/registration.json')['jobs'] if j['candidate']!='k050_o100']
        status('awaiting_registered_stage_inputs',required=len(required))
        while any(not (root/'mechanisms/cases'/(j['recipe']+'_'+j['scenario'])/'completion.json').exists() for j in required):time.sleep(15)
        if not (root/'stage_decision.json').exists():refinement(root)
        for child in ('refinement','allocation_battery'):
            if (root/child/'registration.json').exists():
                status('running_'+child)
                run(root/child,workers=workers)
        jobs=read(root/'mechanisms/registration.json')['jobs']
        status('awaiting_remaining_initial_battery')
        while any(not (root/'mechanisms/cases'/(j['recipe']+'_'+j['scenario'])/'completion.json').exists() for j in jobs):time.sleep(15)
        selected=select(root)
        if selected['winner']:
            status('full_source_qualification',winner=selected['winner'])
            qualify(root,workers=workers)
            status('qualified_awaiting_publication_review',winner=selected['winner'])
        else:status('completed_without_risk_feasible_replacement')
    except Exception as e:
        status('failed_evidence_preserved',error=str(e));raise


def budget_jobs(base,grid):
    recipes={};jobs=[]
    for k,o in grid:
        name=f'k{k*100:g}_o{o*100:g}'
        for label,(scenario,stress) in BATTERY.items():
            suffix='_price' if 'auction_price' in label else '_nonfill' if 'auction_nonfill' in label else ''
            r=copy.deepcopy(base);r['native_budget'].update(kalcb_sizing_fraction=k,olr_fraction=o,allocation_mode='symbol',allocation_seed=0)
            if stress:r['olr_execution_stress']=stress
            recipes[name+suffix]=r
            jobs.append({'recipe':name+suffix,'scenario':scenario,'candidate':name,'battery':label,'group':'budget'})
    return recipes,jobs


def overnight(root,reference):
    initial=read(reference/'protocol.json')
    if any(initial.get(k)!=v for k,v in PROTOCOL.items()):raise ValueError('Initial protocol changed')
    observed=collect(reference,partial=True);failures={}
    for name,label in [('k075_o075','auction_nonfill_combined'),('k050_o075','auction_nonfill_baseline'),('k050_o100','baseline')]:
        evidence=observed[name][label]
        if evidence['gate']['pass']:raise ValueError('Follow-up trigger not supported')
        failures[name]=evidence
    root.mkdir(parents=True,exist_ok=False)
    protocol={**OVERNIGHT_PROTOCOL,'registered_utc':datetime.now(timezone.utc).isoformat(),
        'reference_challenge_root':str(reference.resolve()),
        'reference_protocol':{'path':str(reference/'protocol.json'),'sha256':digest(reference/'protocol.json')},
        'observed_failure_evidence':failures,
        **{k:initial[k] for k in ('incumbent_pointer','incumbent_profile','critical_review_sha256')},
        'driver_sha256':digest(Path(__file__))}
    write_json(root/'protocol.json',protocol)
    base=read(reference/'mechanisms/study.json')['candidates']['k075_o075']
    recipes,jobs=budget_jobs(base,OVERNIGHT_PROTOCOL['budget_grid'])
    freeze(reference/'mechanisms/study.json',root/'coarse',recipes,jobs,overlay=False,registration={
        'protocol':{'path':str(root/'protocol.json'),'sha256':digest(root/'protocol.json')},
        'purpose':'Unchanged gates; reduce OLR overnight budget and test bounded intraday KALCB risk after the original grid has a documented failure for every budget.'})


def overnight_pipeline(root,workers=1):
    def status(stage,**extra):
        v={'stage':stage,'utc':datetime.now(timezone.utc).isoformat(),**extra}
        write_json(root/'pipeline_progress.json',v);print(json.dumps(v),flush=True)
    try:
        status('running_registered_overnight_grid');run(root/'coarse',workers=workers)
        # An interrupted midpoint generation can already exist on resume. Only
        # the completed coarse cases are inputs to the midpoint decision.
        all_data=collect(root,partial=True)
        data={f'k{k*100:g}_o{o*100:g}':all_data[f'k{k*100:g}_o{o*100:g}'] for k,o in OVERNIGHT_PROTOCOL['budget_grid']}
        grid=midpoint_grid(data)
        decision={'registered_midpoints_triggered':grid,'coarse_completions':{
            n:{s:m['completion'] for s,m in b.items()} for n,b in data.items()}}
        if (root/'stage_decision.json').exists():
            if read(root/'stage_decision.json')!=decision:raise ValueError('Changed midpoint registration evidence')
        else:write_json(root/'stage_decision.json',decision)
        if grid:
            status('running_registered_midpoints',grid=grid)
            if not (root/'midpoint/registration.json').exists():
                base=next(iter(read(root/'coarse/study.json')['candidates'].values()))
                recipes,jobs=budget_jobs(base,grid)
                freeze(root/'coarse/study.json',root/'midpoint',recipes,jobs,overlay=False,registration={
                    'protocol':{'path':str(root/'protocol.json'),'sha256':digest(root/'protocol.json')},
                    'stage_decision':{'path':str(root/'stage_decision.json'),'sha256':digest(root/'stage_decision.json')}})
            run(root/'midpoint',workers=workers)
        selected=select(root)
        if selected['winner']:
            status('full_source_qualification',winner=selected['winner'])
            qualify(root,workers=min(workers,2))
            status('qualified_awaiting_publication_review',winner=selected['winner'])
        else:status('completed_without_risk_feasible_replacement')
    except Exception as e:
        status('failed_evidence_preserved',error=str(e));raise


def midpoint_grid(data):
    grid=[]
    for o in (.5,.625):
        low,high=data[f'k75_o{o*100:g}'],data[f'k87.5_o{o*100:g}']
        if set(low)!=set(BATTERY) or set(high)!=set(BATTERY):raise ValueError('Midpoint decision needs complete batteries')
        if all(m['gate']['pass'] for m in low.values()) and not all(m['gate']['pass'] for m in high.values()):grid.append([.8125,o])
    return grid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run','partition','refine','select','qualify','pipeline','overnight','overnight-pipeline']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path)
    p.add_argument('--group',default='all');p.add_argument('--full',action='store_true')
    p.add_argument('--workers',type=int,choices=[1,2,3],default=1);a=p.parse_args()
    if a.command=='prepare':prepare(a.root.resolve())
    elif a.command=='run':run(a.root.resolve(),a.group,a.full,a.workers)
    elif a.command=='partition':partition(a.root.resolve())
    elif a.command=='refine':refinement(a.root.resolve())
    elif a.command=='select':select(a.root.resolve())
    elif a.command=='pipeline':pipeline(a.root.resolve(),a.workers)
    elif a.command=='overnight':
        if a.reference is None:p.error('overnight requires --reference')
        overnight(a.root.resolve(),a.reference.resolve())
    elif a.command=='overnight-pipeline':overnight_pipeline(a.root.resolve(),a.workers)
    else:qualify(a.root.resolve(),a.workers)


if __name__=='__main__':main()

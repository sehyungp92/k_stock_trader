"""Read-only joint exit qualification; no native driver or economic mutation."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip,json
import numpy as np
from .retention_research_io_v1 import read,write,bound,verify
from .late_artifact_validation_v2 import validate_native
from .population_research_io_v1 import bridge_accounts
from .fixed_allocation_funding_v1 import funding_review
GLOBAL={'baseline':73.1303145472,'combined':47.8121465627}


def paired_blocks(candidate,parent,block,reps=10000,seed=20261011):
    """Shared circular session blocks; interval conditional on reused selection."""
    candidate=np.asarray(candidate,float);parent=np.asarray(parent,float)
    if candidate.shape!=parent.shape or candidate.ndim!=1 or len(candidate)<block:
        raise ValueError('Unpaired or insufficient sessions')
    rng=np.random.default_rng(seed);n=len(candidate)
    starts=rng.integers(0,n,size=(reps,(n+block-1)//block))
    idx=((starts[:,:,None]+np.arange(block))%n).reshape(reps,-1)[:,:n]
    delta=(np.prod(1+candidate[idx],axis=1)-np.prod(1+parent[idx],axis=1))*100
    return {'block_sessions':block,'replicates':reps,'seed':seed,
        'full_period_net_return_delta_pp_95_interval':np.quantile(delta,[.025,.975]).tolist(),
        'positive_delta_share':float(np.mean(delta>0)),
        'observed_delta_pp':float((np.prod(1+candidate)-np.prod(1+parent))*100),
        'interpretation':'Paired conditional sensitivity, not independent validation, posterior probability or multiplicity-adjusted evidence.'}


def account_attribution(folder):
    daily=read(folder/'daily.json');trades=read(folder/'trades.json');equity=read(folder/'equity.json')
    previous=100000000.;returns=[];month=defaultdict(lambda:{'net_change_krw':0.,'sessions':0,'first_equity':None,'last_equity':0})
    for row in daily:
        d=row['day'];v=float(row['equity']);returns.append(v/previous-1)
        m=month[d[:7]]
        if m['first_equity'] is None:m['first_equity']=previous
        m['net_change_krw']+=v-previous;m['sessions']+=1;m['last_equity']=v;previous=v
    for m in month.values():m['return_pct']=100*(m['last_equity']/m['first_equity']-1)
    issuers=defaultdict(lambda:{'net_pnl_krw':0.,'gross_positive_net_trades_krw':0.,'trades':0})
    for t in trades:
        s=issuers[t['symbol']];s['net_pnl_krw']+=t['net_pnl'];s['gross_positive_net_trades_krw']+=max(0,t['net_pnl']);s['trades']+=1
    assert abs(sum(x['net_pnl_krw'] for x in issuers.values())-(previous-100000000))<1e-4
    peak=100000000.;peak_time='initial';worst={'drawdown_pct':0.};trough_index=0
    for i,(ts,v) in enumerate(equity):
        if v>peak:peak=v;peak_time=ts
        dd=100*(1-v/peak)
        if dd>worst['drawdown_pct']:worst={'drawdown_pct':dd,'peak_at':peak_time,'peak_equity':peak,'trough_at':ts,'trough_equity':v};trough_index=i
    worst['recovered_at']=next((ts for ts,v in equity[trough_index+1:] if v>=worst.get('peak_equity',100000000.)),None)
    winners=sorted((t['net_pnl'] for t in trades if t['net_pnl']>0),reverse=True)
    positives=sum(winners)
    losses=sorted((t['net_pnl'] for t in trades if t['net_pnl']<0))
    fills=read(folder/'fills.json')
    fees=sum(f['metadata']['fee'] for f in fills)
    slippage=sum(abs(f['price']-f['metadata']['base_price'])*f['qty'] for f in fills)
    episodes=[];episode=None;running_peak=100000000.;running_peak_at='initial'
    for stamp,value in equity:
        if value>=running_peak:
            if episode is not None:
                episode['recovered_at']=stamp;episodes.append(episode);episode=None
            running_peak=value;running_peak_at=stamp
        else:
            dd=100*(1-value/running_peak)
            if episode is None:
                episode={'peak_at':running_peak_at,'peak_equity':running_peak,'trough_at':stamp,'trough_equity':value,'drawdown_pct':dd,'recovered_at':None}
            elif dd>episode['drawdown_pct']:
                episode.update(trough_at=stamp,trough_equity=value,drawdown_pct=dd)
    if episode is not None:episodes.append(episode)
    return {'daily_dates':[x['day'] for x in daily],'daily_returns':returns,'months':dict(month),'issuers':dict(issuers),
        'largest_winner_krw':max(winners,default=0),'top5_winners_krw':sum(winners[:5]),
        'largest_loss_krw':min(losses,default=0),'worst5_losses_krw':sum(losses[:5]),'mtm_drawdown_episodes':episodes,
        'fees_tax_krw':fees,'modeled_adverse_slippage_krw':slippage,
        'cost_scope':'Fees reconstructed from fills; adverse price slippage is already embedded in trade prices and net equity. Do not subtract it again.',
        'top_issuer_gross_profit_share':max((x['gross_positive_net_trades_krw'] for x in issuers.values()),default=0)/positives if positives else None,
        'worst_mtm_episode':worst,'closed_round_trips':len(trades),'profitable_round_trips':len(winners),'losing_round_trips':len(losses),'profitable_round_trips_per_session':len(winners)/len(daily),'closed_round_trips_per_session':len(trades)/len(daily),'concentration_scope':'Observed account allocation. Removing an issuer ex post is not an executable funded counterfactual.'}


def activity(folder):
    trades=read(folder/'trades.json'); fills=read(folder/'fills.json')
    days=len(read(folder/'daily.json'))
    return {'closed_trades':len(trades),'closed_trades_per_session':len(trades)/days,
        'filled_entry_issuer_days':len({(r['symbol'],r['timestamp'][:10]) for r in fills if r['side']=='BUY'}),
        'filled_entry_orders':len({r['order_id'] for r in fills if r['side']=='BUY'}),
        'profitable_trades':sum(t['net_pnl']>0 for t in trades),
        'win_rate':sum(t['net_pnl']>0 for t in trades)/len(trades) if trades else None,
        'average_positive_net_pnl':sum(max(t['net_pnl'],0) for t in trades)/max(1,sum(t['net_pnl']>0 for t in trades)),
        'average_negative_net_pnl':sum(min(t['net_pnl'],0) for t in trades)/max(1,sum(t['net_pnl']<0 for t in trades))}


def entry_outcomes(folder):
    result=defaultdict(lambda:{'net_pnl':0.,'qty':0})
    for t in read(folder/'trades.json'):
        key=(t['symbol'],t['entry_fill_time'])
        result[key]['net_pnl']+=t['net_pnl'];result[key]['qty']+=t['qty']
    return result


def tail_comparison(candidate,parent):
    c,p=entry_outcomes(candidate),entry_outcomes(parent)
    top=sorted(p,key=lambda k:p[k]['net_pnl'],reverse=True)[:10]
    return {'parent_top10_entries':[{'symbol':k[0],'entry_fill_time':k[1],
        'parent':p[k],'candidate':c.get(k)} for k in top],
        'parent_top10_net_pnl':sum(p[k]['net_pnl'] for k in top),
        'candidate_same_entry_net_pnl':sum(c.get(k,{}).get('net_pnl',0) for k in top),
        'new_entry_keys':len(set(c)-set(p)),'absent_parent_entry_keys':len(set(p)-set(c)),
        'scope':'Entry-key attribution; quantities, partial exits and funding descendants may differ. No isolated causal PnL claim.'}


def support(folder):
    e=read(folder/'mechanism_evidence.json'); fills=read(folder/'fills.json')
    actual={f['metadata']['execution_id']:f for f in fills}
    seen=set()
    for x in e['summary']['executions']:
        if x['execution_id'] in seen:raise ValueError('Duplicate intervention execution')
        seen.add(x['execution_id']);f=actual[x['execution_id']]
        if f['side']!='SELL' or f['order_id']!=x['order_id'] or f['reason']!='opening_thesis_failure' or f['qty']!=x['qty']:
            raise ValueError('Intervention is not an actual matching native exit')
        if x['parent_exit_reason'] is not None:
            raise ValueError('Parent exit priority was violated')
    entries=e['entry_probes']
    classifications=Counter()
    for x in entries:
        if not x.get('native_prefix_matched'):raise ValueError('Unbound alternative prefix')
        if x.get('already_candidate'):classifications['already_candidate']+=1
        elif x.get('native_order_submitted'):classifications['native_submission_in_copy']+=1
        else:
            reasons={d['reason'] for d in x.get('decisions',[])}
            classifications.update('blocked:'+r for r in reasons or {'no_decision'})
    return {'summary':e['summary'],'entry_preview_classification':dict(classifications),
        'observed_entry_previews':entries,
        'low_support':e['summary']['direct_filled_positions']<20,
        'interpretation':'Fewer than20 positions flagged as low support descriptively, not a performance acceptance threshold. Previews submit only to account copies, not funded fills or economic benefit; retries/descendants dependent.'}


def expected_cases(protocol):
    return {(mode,name,period,scenario) for mode,recipes in protocol['stages'].items()
        for name,spec in recipes.items() for period in ('is','oos') for scenario in spec['scenarios']}


def require_design(root):
    root=Path(root);protocol=read(root/'protocol.json');done=read(root/'stage_completion.json')
    rows=read(root/'progress.json');actual=[tuple(r[k] for k in ('mode','name','period','scenario')) for r in rows]
    if not done.get('complete') or done.get('native_cases')!=40 or len(actual)!=40 or set(actual)!=expected_cases(protocol):
        raise ValueError('Missing, duplicate or undeclared qualification case')
    if sum(r['authenticated_reuse'] for r in rows)!=12 or done.get('fresh_native_cases')!=28:
        raise ValueError('Incorrect fresh/reuse scope')
    if sum(r['mode']=='identity' and r['exact_bridge'] for r in rows)!=8:
        raise ValueError('Eight disabled-overlay bridges required')
    if sum(r['mode']=='fixed' and r['name']=='control' and r['exact_bridge'] for r in rows)!=4:
        raise ValueError('Four funded parent bridges required')
    if sum(r['mode']=='compounding' and r['name']=='control' and r['exact_bridge'] for r in rows)!=8:
        raise ValueError('Eight compounding parent bridges required')
    for row in rows:
        if row['metrics']['audit_status']!='passed':raise ValueError('Unaudited native case')
        verify([row[k] for k in ('result','input_profile','funding','evidence')])
        if row['log']:verify([row['log']])
    return rows


def allocation_contract(folder, mode):
    if mode=='compounding':return None
    evidence=read(Path(folder)/'allocation_reference.json')
    if (evidence['version']!='fixed-allocation-reference-v1' or
        evidence['enabled'] is not (mode=='fixed') or evidence['reference']!=100_000_000. or
        evidence['ledger_cash_mutations']!=0 or evidence['external_capital_flows']!=0):
        raise ValueError('Changed funded allocation contract')
    return {**{k:v for k,v in evidence.items() if k!='rows'},'binding':bound(Path(folder)/'allocation_reference.json'),
        'rows_scope':'Allocation references use last mark time; actual funding proof comes from broker orders/fills, not these timestamps.'}


def direct_support(folder):
    folder=Path(folder);result=support(folder);summary=result['summary']
    submitted={}
    with gzip.open(folder/'orders.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            event=json.loads(line);order=event.get('order',{})
            if event['event']=='submit' and order.get('side')=='SELL' and order.get('reason')=='opening_thesis_failure':
                if order['order_id'] in submitted:raise ValueError('Duplicate exit submission')
                submitted[order['order_id']]=order
    positions=set();sessions=set();issuers=set();orders=set()
    for execution in summary['executions']:
        order=submitted.get(execution['order_id'])
        if order is None or order['symbol']!=execution['symbol']:
            raise ValueError('Unsupported exit fill without submitted native order')
        if datetime.fromisoformat(order['metadata']['decision_available_at'])>datetime.fromisoformat(execution['timestamp']):
            raise ValueError('Exit execution before completed decision')
        positions.add((execution['symbol'],execution['entry_time']));sessions.add(execution['timestamp'][:10])
        issuers.add(execution['symbol']);orders.add(execution['order_id'])
    if (len(positions)!=summary['direct_filled_positions'] or len(sessions)!=summary['direct_filled_sessions'] or
        len(orders)!=summary['filled_exit_orders'] or len(submitted)!=summary['submitted_exit_orders']):
        raise ValueError('Dependent exit support count drift')
    result['authenticated_actual_support']={'submitted_orders':len(submitted),'filled_orders':len(orders),
        'filled_positions':len(positions),'filled_sessions':len(sessions),'filled_issuers':len(issuers),
        'filled_months':len({d[:7] for d in sessions}),'partial_executions':len(summary['executions']),
        'scope':'Actual submitted and filled new exits with complete-parent no-exit observer proof. Retries/partials/descendants remain dependent; no isolated saved-loss claim.'}
    return result


def account_evidence(folder,item,scenario,period,*,mode='compounding',mechanism=True):
    folder=Path(folder);metrics=validate_native(folder,item,scenario,save=False)
    attr=account_attribution(folder)
    if len(attr['daily_dates'])!={'is':156,'oos':69}[period]:raise ValueError('Shortened calendar')
    if abs(attr['worst_mtm_episode']['drawdown_pct']-metrics['drawdown_pct'])>1e-8:
        raise ValueError('Timestamped MTM drawdown attribution drift')
    funding=funding_review(folder)
    if (folder/'funding_review.json').exists() and funding!=read(folder/'funding_review.json'):
        raise ValueError('Actual funding evidence changed')
    files=['result.json','execution_audit.json','independent_account_verification.json','daily.json','equity.json',
        'fills.json','trades.json','orders.jsonl.gz','decisions.jsonl.gz']
    if mechanism:files.append('mechanism_evidence.json')
    return {'folder':str(folder),'metrics':metrics,'attribution':attr,'activity':activity(folder),
        'support':direct_support(folder) if mechanism else {'scope':'Unchanged prior125 alternative; no opening-thesis exit observer, no inferred support.'},
        'funding':funding,'allocation':allocation_contract(folder,mode),
        'lineage':{k:item[k] for k in ('source_generation_hash','manifest','plan','policy')},
        'bindings':[bound(folder/f) for f in files]}


def comparison(candidate,parent,period,scenario):
    c,p=candidate,parent;ca,pa=c['attribution'],p['attribution']
    if ca['daily_dates']!=pa['daily_dates']:raise ValueError('Unpaired calendars')
    sacrifice={}
    if period=='is':
        sacrifice={'local':p['metrics']['return_pct']-c['metrics']['return_pct']}
        if scenario in GLOBAL:sacrifice['global']=GLOBAL[scenario]-c['metrics']['return_pct']
    return {'metric_delta':{k:c['metrics'][k]-p['metrics'][k] for k in ('return_pct','drawdown_pct','trades','active_days')},
        'activity_delta':{k:c['activity'][k]-p['activity'][k] for k in ('closed_trades','filled_entry_issuer_days','filled_entry_orders','profitable_trades')},
        'IS_sacrifice_pp':sacrifice,'IS_reporting_bands':{str(b):{k:v<=b for k,v in sacrifice.items()} for b in (2,5,10)},
        'right_tail':tail_comparison(Path(c['folder']),Path(p['folder'])),
        'month_delta':{m:{k:ca['months'][m][k]-pa['months'][m][k] for k in ('net_change_krw','return_pct')} for m in ca['months']},
        'issuer_delta_krw':{s:ca['issuers'].get(s,{}).get('net_pnl_krw',0)-pa['issuers'].get(s,{}).get('net_pnl_krw',0)
            for s in sorted(set(ca['issuers'])|set(pa['issuers']))},
        'fees_delta_krw':ca['fees_tax_krw']-pa['fees_tax_krw'],
        'paired_uncertainty':[paired_blocks(ca['daily_returns'],pa['daily_returns'],b) for b in (5,10)],
        'scope':'Paired funded paths, not isolated causal intervention PnL. April-July reused development; block sensitivity is conditional and not multiplicity adjusted.'}


def difference_in_deltas(fixed,compounding):
    return {k:fixed['metric_delta'][k]-compounding['metric_delta'][k]
        for k in ('return_pct','drawdown_pct','trades','active_days')}


def assess(output):
    output=Path(output);reg=read(output/'protocol.json');verify(reg['programs']+reg['bindings'])
    from .late_signal_reuse_v1 import require_closed
    require_closed(reg['native_supervisor'])
    root=Path(reg['native_root']);rows=require_design(root);protocol=read(root/'protocol.json')
    verify(protocol['programs']+protocol['bindings'])
    if (output/'assessment.json').exists():raise ValueError('Preserve completed assessment')
    accounts={};bridges=[];bindings=[]
    for row in rows:
        mode,name,period,scenario=(row[k] for k in ('mode','name','period','scenario'))
        folder=root/mode/'cases'/name/period/scenario;item=read(folder/'qualification_input_profile.json')
        key='/'.join((mode,name,period,scenario))
        accounts[key]=account_evidence(folder,item,scenario,period,mode=mode)
        bindings+=accounts[key]['bindings']+[bound(folder/'qualification_input_profile.json')]
        reference=None
        if mode=='identity':reference=root/'compounding/cases'/name/period/scenario
        elif mode=='fixed' and name=='control':reference=Path(protocol['fixed_parent_references'][period][scenario])
        elif mode=='compounding' and name=='control':reference=Path(protocol['parent_stress_references'][period][scenario])
        if reference is not None:
            bridge=bridge_accounts(folder,reference);bridges.append({'case':key,'bridge':bridge})
        print(json.dumps({'audited_case':key,'count':len(accounts)}),flush=True)
    if len(bridges)!=20 or not all(b['bridge']['passed'] for b in bridges):raise ValueError('Missing exact native bridge')
    competitor=read(reg['competing_protocol']['path']);verify([reg['competing_protocol']])
    for period,scenarios in protocol['competing_prior125_references'].items():
        for scenario,path in scenarios.items():
            key='/'.join(('competing','prior125',period,scenario))
            item=competitor['profiles']['expanding_removal_prior125'][period]
            accounts[key]=account_evidence(Path(path),item,scenario,period,mechanism=False)
            bindings+=accounts[key]['bindings']
            print(json.dumps({'audited_competitor':key}),flush=True)
    trials={};funded={};backouts={};competitors={}
    for name in ('prior100_failure6','prior100_failure8'):
        for period in ('is','oos'):
            for scenario in ('baseline','participation','cost','combined'):
                key='/'.join((name,period,scenario));candidate=accounts['compounding/'+key]
                parent=accounts['/'.join(('compounding','control',period,scenario))]
                trials[key]=comparison(candidate,parent,period,scenario)
                if name=='prior100_failure8':
                    backouts[key]=comparison(candidate,accounts['/'.join(('compounding','prior100_failure6',period,scenario))],period,scenario)
                    competitors[key]=comparison(candidate,accounts['/'.join(('competing','prior125',period,scenario))],period,scenario)
                    if scenario in ('baseline','combined'):
                        cf=accounts['/'.join(('fixed',name,period,scenario))];pf=accounts['/'.join(('fixed','control',period,scenario))]
                        f=comparison(cf,pf,period,scenario)
                        funded[key]={'fixed_vs_fixed_parent':f,'policy_delta_fixed_minus_compounding':difference_in_deltas(f,trials[key]),
                            'interpretation':'Four funded path comparison; nominal allocation fixed, no cash replenishment. Not isolated compounding PnL.'}
    verify(reg['programs']+reg['bindings']+bindings)
    result={'created_at_utc':datetime.now(timezone.utc).isoformat(),'registered':bound(output/'protocol.json'),
        'native_qualification_cases':40,'fresh_native_cases':28,'authenticated_reuse':12,'additional_competing_prior125_cells':8,
        'all_source_accounts_passed':True,'accounts':accounts,'exact_bridges':bridges,'local_comparisons':trials,
        'hold8_vs_hold6_backout':backouts,'hold8_vs_unchanged_prior125':competitors,'funded_comparisons':funded,
        'retained_screening_assessment':reg['screening_assessment'],'bindings':bindings,'study_complete':False,
        'canonical_replaced':False,'production_enabled':False,'next':'Economic review, truthful existing-engine full reports, and separately registered evidence-supported next entry investigation. No automatic promotion.',
        'limitations':['OOS is reused development, not independent validation.','Direct exit positions, partial fills, retries and account descendants are dependent.','No date/symbol exclusions or retrospective recovered-MFE claim.','DD8/10% and IS2/5/10pp bands are descriptive preferences, not new hard thresholds.']}
    write(output/'assessment.json',result)
    write(output/'completion.json',{'complete':True,'assessment':bound(output/'assessment.json'),'study_complete':False,'source_account_cases':48,'exact_bridges':20})

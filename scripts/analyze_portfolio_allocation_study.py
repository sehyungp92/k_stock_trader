"""Descriptive diagnostics and independent forecast binding for allocation trials."""
import argparse
from collections import Counter
from datetime import datetime, timedelta
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,indent=2,default=str),encoding='utf-8')


def case_paths(study_path,seen=None):
    """Resolve bound parent evidence, preferring this generation on name clashes."""
    study_path=Path(study_path).resolve();seen=set() if seen is None else seen
    if study_path in seen:raise ValueError('Cyclic study ancestry')
    seen.add(study_path);root=study_path.parent;paths={}
    final=read(root/'finalist.json') if (root/'finalist.json').exists() else {}
    prior=final.get('prior_study')
    if prior:
        parent=Path(prior['path'])
        if digest(parent)!=prior['sha256'] or digest(parent.parent/'finalist.json')!=prior['finalist_sha256']:
            raise ValueError('Changed prior study evidence')
        paths.update(case_paths(parent,seen))
    challenge=final.get('reference_challenge')
    if challenge:
        paths.update({p.name:p for p in (Path(challenge['study']).parent/'cases').iterdir()})
    paths.update({p.name:p for p in (root/'cases').iterdir() if (p/'completion.json').exists()})
    return paths


def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:yield json.loads(line)


def summarize(path,initial):
    result=read(path/'result.json');daily=read(path/'daily.json');trades=read(path/'trades.json')
    equity=np.array([initial,*[d['equity'] for d in daily]])
    returns=equity[1:]/equity[:-1]-1
    tail=returns[returns<=np.quantile(returns,.05)]
    sessions={d['day']:i for i,d in enumerate(daily)}
    o=[t for t in trades if t['strategy_id']=='OLR']
    exposure=np.array([(d['equity']-d['cash'])/d['equity'] for d in daily])
    beginning_equity={d['day']:equity[i] for i,d in enumerate(daily)}
    ticket_fractions=np.array([t['qty']*t['entry_price']/beginning_equity[t['entry_fill_time'][:10]] for t in trades])
    sleeves={}
    for sid in ('KALCB','OLR'):
        rows=[t for t in trades if t['strategy_id']==sid]
        if rows:
            sleeves[sid]={'trades':len(rows),'net_pnl_krw':sum(t['net_pnl'] for t in rows),
                'equal_notional_net_bps':float(np.mean([t['net_pnl']/(t['qty']*t['entry_price'])*10000 for t in rows]))}
    result['diagnostics']={'worst_session_pct':float(min(returns)*100),'mean_worst_5pct_sessions_pct':float(np.mean(tail)*100),
        'mean_eod_exposure_pct':float(np.mean(exposure)*100),'max_eod_exposure_pct':float(max(exposure)*100),
        'olr_trades':len(o),'olr_final_exit_later_than_next_session':sum(sessions[t['exit_fill_time'][:10]]-sessions[t['entry_fill_time'][:10]]>1 for t in o),
        'equal_notional_trade_net_bps':float(np.mean([t['net_pnl']/(t['qty']*t['entry_price'])*10000 for t in trades])),
        'entry_notional_to_prior_close_equity_sum':float(sum(ticket_fractions)),
        'mean_entry_turnover_per_session':float(sum(ticket_fractions)/len(daily)),
        'median_entry_ticket_pct_prior_close_equity':float(np.median(ticket_fractions)*100),
        'trades_at_least_1pct_prior_close_equity':int(sum(ticket_fractions>=.01)),
        'trades_below_1pct_prior_close_equity':int(sum(ticket_fractions<.01)),
        'profit_jan_feb_share':sum(daily[i]['equity']-equity[i] for i in range(len(daily)) if daily[i]['day'].startswith(('2026-01','2026-02')))/(equity[-1]-initial)}
    result['diagnostics']['sleeves']=sleeves
    if len(sleeves)==2:
        k=np.array([sum(t['net_pnl'] for t in trades if t['strategy_id']=='KALCB' and t['exit_fill_time'][:10]==d['day']) for d in daily])
        o=equity[1:]-equity[:-1]-k
        worst=int(np.argmin(returns))
        result['diagnostics']['marked_sleeve_attribution']={
            'basis':'KALCB is flat at session close. OLR is the residual marked account change after realized KALCB net PnL.',
            'correlation':float(np.corrcoef(k/equity[:-1],o/equity[:-1])[0,1]),
            'both_negative_sessions':int(sum((k<0)&(o<0))),
            'worst_account_session':daily[worst]['day'],
            'worst_session_kalcb_contribution_pct':float(k[worst]/equity[worst]*100),
            'worst_session_olr_contribution_pct':float(o[worst]/equity[worst]*100),
            'terminal_olr_reconciliation_error_krw':float(sum(o)-sleeves['OLR']['net_pnl_krw'])}
    return result,daily,returns


def validate_forecasts(path,study):
    if not (path/'allocation_events.jsonl.gz').exists():return None
    from backtests.strategies.portfolio_synergy.quality import CausalQualityModel
    frozen=Path(study['code']);name='backtests/strategies/portfolio_synergy/quality.py'
    if digest(frozen/name)!=digest(REPO/name):raise ValueError('Use the frozen quality model for independent binding')
    config=study['candidates'][read(path/'result.json')['recipe']]['allocation']
    model=CausalQualityModel(read(frozen/study['labels'])) if config.get('rank_mode')=='supported_value' else None
    native={};errors=[];reasons=Counter();bases=Counter();approved=set();intents=0;cycles=0
    cross_strategy=0;competitive=0;capacity_coverage=Counter();support=Counter();native_submissions=0
    for row in rows(path/'allocation_events.jsonl.gz'):
        if row['event']=='entry_intent':native[row['intent_id']]=row;intents+=1
        if row['event']=='native_submission':
            native[row['intent_id']]=row;intents+=1;native_submissions+=1
            if row['submitted_order_id']:approved.add(row['intent_id'])
        if row['event']!='allocation_cycle':continue
        cycles+=1;at=datetime.fromisoformat(row['timestamp'])
        cross_strategy+=len({x['strategy_id'] for x in row['intents']})>1
        desired={x['intent_id']:x['desired_qty'] for x in row['intents']}
        competitive+=len(desired)>1 and any(x['approved_qty']<desired[x['intent_id']] for x in row['decisions'])
        for item in row['intents']:
            original=native[item['intent_id']];metadata=original['native_metadata']
            prediction=model.prediction(item['strategy_id'],metadata,at) if model else None
            if prediction and prediction['cohort_support']<20:prediction=None
            declared=row['forecasts'][item['intent_id']]['value']
            support['supported' if prediction else 'no_supported_value']+=1
            for kind,part in row['forecasts'][item['intent_id']]['capacity'].items():
                capacity_coverage[kind+('_forecast' if 'quantity' in part else '_native_fallback')]+=1
            if declared!=prediction:errors.append('prediction_not_bound_to_frozen_model')
            if prediction:
                if (item['robust_net_bps']!=prediction['score_bps'] or item['value_source']!=model.label_hash
                        or datetime.fromisoformat(prediction['latest_label_exit']).date()>=at.date()):
                    errors.append('incorrect_or_noncausal_prediction')
            elif item['robust_net_bps'] is not None:errors.append('unbound_numeric_prediction')
            if datetime.fromisoformat(original['available_at'])>at:errors.append('intent_used_before_observation')
            if datetime.fromisoformat(original['expires_at'])<at:errors.append('expired_intent_competed')
        for decision in row['decisions']:
            reasons[decision['reason']]+=1;bases[decision['rank_basis']]+=1
            if decision['approved_qty']>0:approved.add(decision['intent_id'])
    return {'pass':not errors,'errors':sorted(set(errors)),'native_intents':intents,'allocation_cycles':cycles,
        'approved_intent_count':len(approved),'never_approved_intent_count':len(set(native)-approved),
        'native_kalcb_submission_count':native_submissions,
        'decision_event_reasons':dict(reasons),'rank_basis_events':dict(bases),
        'cross_strategy_cohorts':cross_strategy,'multi_intent_constrained_cohorts':competitive,
        'value_support_events':dict(support),'capacity_forecast_events':dict(capacity_coverage),
        'source':'Frozen label file and quality-model bytes, independently re-evaluated at each decision timestamp'}


def opportunity_retention(path,standalone_paths,reference_kind='native standalone'):
    """Partial hindsight diagnostic; comparison fills are not universal labels."""
    references={}
    for p in standalone_paths:
        for t in read(p/'trades.json'):
            origin=t.get('route_metadata',{}).get('allocation_origin_at')
            at=datetime.fromisoformat(origin) if origin else datetime.fromisoformat(t['entry_decision_time'])+timedelta(minutes=5)
            key=(t['strategy_id'],t['symbol'],at.isoformat())
            if key in references:raise ValueError('Ambiguous standalone opportunity identity')
            references[key]=t
    intents={};accepted=set()
    for row in rows(path/'allocation_events.jsonl.gz'):
        if row['event'] in ('entry_intent','native_submission'):intents[row['intent_id']]=row
        if row['event']=='native_submission' and row['submitted_order_id']:accepted.add(row['intent_id'])
        if row['event']=='allocation_cycle':
            accepted.update(x['intent_id'] for x in row['decisions'] if x['approved_qty']>0)
    output={}
    for sid in ('KALCB','OLR'):
        values={'approved':[],'unapproved':[]};matched=set();native_count=0;unknown=0
        for ident,item in intents.items():
            if item['strategy_id']!=sid:continue
            native_count+=1
            key=(sid,item['symbol'],item['available_at']);reference=references.get(key)
            if reference is None:unknown+=1;continue
            matched.add(key)
            value=reference['net_pnl']/(reference['qty']*reference['entry_price'])*10000
            values['approved' if ident in accepted else 'unapproved'].append(value)
        output[sid]={'native_intents':native_count,'intents_without_reference_label':unknown,
            'reference_trades_without_matching_intent':sum(k[0]==sid and k not in matched for k in references),
            'matched_reference_net_bps':{name:{'count':len(v),'mean':float(np.mean(v)) if v else None,
                'median':float(np.median(v)) if v else None,'positive_count':sum(x>0 for x in v)} for name,v in values.items()}}
    return {'scope':f'Descriptive matched {reference_kind} outcomes only; different sizing and shared execution can change outcomes. No label for never-traded opportunities; no rejected profit is credited to portfolio.',
            'strategies':output}


def paired_bootstrap(a,b,repetitions=5000):
    if len(a)!=len(b):raise ValueError('Common dates required')
    n=len(a);rng=np.random.default_rng(20261008);block=5
    starts=rng.integers(0,n-block+1,size=(repetitions,(n+block-1)//block))
    indexes=(starts[:,:,None]+np.arange(block)).reshape(repetitions,-1)[:,:n]
    delta=(np.prod(1+a[indexes],axis=1)-np.prod(1+b[indexes],axis=1))*100
    return {'observed_return_difference_pct_points':float((np.prod(1+a)-np.prod(1+b))*100),
        'descriptive_95pct_interval_pct_points':[float(x) for x in np.quantile(delta,[.025,.975])],
        'block_sessions':block,'resamples':repetitions,'seed':20261008,
        'scope':'Development-sample dependence sensitivity; does not correct repeated strategy/policy selection or establish future expected returns'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True)
    args=parser.parse_args();study=read(args.study);root=args.study.parent
    results={};daily={};returns={};forecast={}
    finalist=read(root/'finalist.json') if (root/'finalist.json').exists() else None
    paths=case_paths(args.study)
    for path in sorted(paths.values()):
        if not (path/'completion.json').exists():continue
        results[path.name],daily[path.name],returns[path.name]=summarize(path,study['initial_equity'])
        forecast[path.name]=validate_forecasts(path,study)
    report={'scope':'reused_development_only','results':results,'forecast_binding':forecast}
    report['value_model_ablation_retention']={scenario:opportunity_retention(
        paths['value_capacity_75_'+scenario],[paths['deferred_capacity_75_'+scenario]],
        'otherwise identical no-value-model control') for scenario in ('baseline','combined')
        if all(name+'_'+scenario in results for name in ('value_capacity_75','deferred_capacity_75'))}
    if finalist and finalist['winner']:
        name=finalist['winner'];comparisons={};retention={};sized_retention={}
        for scenario in ('baseline','combined'):
            retention[scenario]=opportunity_retention(paths[name+'_'+scenario],
                [paths[sid+'_native_'+scenario] for sid in ('kalcb','olr')])
            sized_retention[scenario]=opportunity_retention(paths[name+'_'+scenario],
                [paths['kalcb_size_200_'+scenario],paths['olr_native_'+scenario]],
                'KALCB2 standalone and native OLR')
        for reference in ['legacy_selected',*finalist['reference_frontier']]:
            for scenario in ('baseline','combined'):
                a=name+'_'+scenario;b=reference+'_'+scenario
                if [x['day'] for x in daily[a]]!=[x['day'] for x in daily[b]]:raise ValueError('Different benchmark dates')
                comparisons[b]=paired_bootstrap(returns[a],returns[b])
        report['paired_development_uncertainty']=comparisons
        report['selected_opportunity_retention']=retention
        report['selected_sized_reference_retention']=sized_retention
    write(root/'allocation_diagnostics.json',report)
    print(json.dumps({'cases':len(results),'forecast_checks_passed':all(x is None or x['pass'] for x in forecast.values()),
        'output':str(root/'allocation_diagnostics.json')}))


if __name__=='__main__':main()

"""Independent equations and economic diagnostics for fixed V38 slope interactions."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
ROOT=BASE/'v38_checkpoint_context_interaction'
TEST=REPO/'tests/backtests/strategies/test_kalcb_checkpoint_interaction_review_v1.py'
SCENARIOS=('baseline','combined')
RECIPES=('peer_context','context_modulated')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def bound(p):
    with Path(p).open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(Path(p).resolve()),'sha256':d}


def verify(items):
    for v in items:
        if bound(v['path'])['sha256']!=v['sha256']:raise ValueError('Changed source '+v['path'])


def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)


def lines(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return list(map(json.loads,f))


def close(a,b):
    if not math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-8):
        raise ValueError(f'Independent numerical mismatch: {a} != {b}')


def signature(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()


def augment(d,columns):
    if 'own_return_x_market_breadth_change' in columns:
        d['own_return_x_market_breadth_change']=d['post_entry_return_atr']*d['peer_market_vwap_breadth_change']
        d['own_return_x_sector_breadth_change']=d['post_entry_return_atr']*d['peer_sector_vwap_breadth_change']
    return d


def prediction(pair,proof,recipe,columns,n_own):
    if set(pair)!=set(SCENARIOS):return {'supported':False,'reason':'missing_scenario_holding'}
    x={}
    for s,r in pair.items():
        if not proof['valid_from']<=r['day']<=proof['valid_through']:raise ValueError('Wrong forecast clock')
        groups=(r['own_features'],r['state_features'],r['context_features'])
        if r['eligibility']!='eligible' or r['held_quantity']<=0 or r['pending_buy_quantity']!=0 or any(g is None for g in groups):
            return {'supported':False,'reason':'no_holding_pending_parent_exit_or_features'}
        features={k:v for g in groups for k,v in g.items()}
        features=augment(features,columns)
        if any(k not in features for k in columns):raise ValueError('Missing feature')
        if any(features[k] is None or not math.isfinite(features[k]) for k in columns):
            return {'supported':False,'reason':'no_holding_pending_parent_exit_or_features'}
        x[s]=[features[k] for k in columns]
    if proof['status']!='supported':return {'supported':False,'reason':'mature_session_support'}
    values={};outside={}
    for s,r in pair.items():
        if not proof['quantity_support'].get(str(float(r['quantity_scale'])),{}).get('supported'):
            return {'supported':False,'reason':'quantity_support'}
        m=proof['models'][s];outside[s]=[columns[i] for i,v in enumerate(x[s]) if v<m['minimum'][i]-1e-12 or v>m['maximum'][i]+1e-12]
        n=n_own if recipe=='peer_context' else len(columns)
        z=[1.]+[(v-c)/sd for v,c,sd in zip(x[s][:n],m['center'][:n],m['scale'][:n])]
        values[s]=sum(a*b for a,b in zip(z,m['coefficients'][recipe]))
    if any(outside.values()):return {'supported':False,'reason':'prior_feature_envelope','outside':outside}
    return {'supported':True,'values':values}


def check_prediction(record,expected):
    if record['supported']!=expected['supported']:raise ValueError('Changed forecast support')
    if not expected['supported']:
        if record['reason']!=expected['reason']:raise ValueError('Wrong abstention reason')
        if expected.get('outside')!=record.get('outside'):raise ValueError('Wrong prior domain')
    else:
        for s in SCENARIOS:close(record['values'][s],expected['values'][s])


def affine_points(history,allowed,scenario):
    points=[];seen=set()
    for r in history:
        if r['day'] not in allowed:continue
        if r['valid_from']>r['day']:raise ValueError('Nonforward calibration')
        if not r['matched_predecision_both'] or not r['risk_pass'] or not r['raw']['supported']:continue
        if not all(r['labels'][s]['valid'] and r['labels'][s]['available_at'] and
                   r['labels'][s]['available_at'][:10]<=max(allowed) for s in SCENARIOS):continue
        key=(r['day'],r['symbol'])
        if key in seen:raise ValueError('Repeated calibration issuer-day')
        seen.add(key);points.append((key,r['raw']['values'][scenario],r['labels'][scenario]['value']))
    return points


def affine_equations(points):
    n=len(points);days=len({x[0][0] for x in points})
    if n<20 or days<8:return {'supported':False,'issuer_days':n,'sessions':days,'center':0.,'offset':0.,'slope':1.}
    mean=sum(x[1] for x in points)/n;variance=sum((x[1]-mean)**2 for x in points)/n
    slope=(sum((x[1]-mean)*(x[2]-mean) for x in points)+100*variance)/(n*variance+100*variance) if variance>1e-12 else 1.
    return {'supported':True,'issuer_days':n,'sessions':days,'center':mean,
            'offset':sum(x[2]-mean for x in points)/(n+100),'slope':max(0.,min(1.,slope))}


def metric(records):
    """Equal issuer-day and equal session influence; values are bps, never money."""
    if not records:return {'rows':0,'issuer_days':0,'sessions':0}
    quantities=Counter((r['day'],r['symbol']) for r in records)
    names=Counter(d for d,_ in quantities)
    weights=[1/quantities[r['day'],r['symbol']]/names[r['day']] for r in records];den=sum(weights)
    return {'rows':len(records),'issuer_days':len(quantities),'sessions':len(names),
        'mse_bps2':sum(w*(r['prediction']-r['target'])**2 for w,r in zip(weights,records))/den,
        'mae_bps':sum(w*abs(r['prediction']-r['target']) for w,r in zip(weights,records))/den,
        'bias_bps':sum(w*(r['prediction']-r['target']) for w,r in zip(weights,records))/den,
        'mean_target_bps':sum(w*r['target'] for w,r in zip(weights,records))/den}


def positive_metrics(rows,kind):
    chosen=[r for r in rows if r[kind]['supported'] and min(r[kind]['values'].values())>0]
    known=[r for r in chosen if r['matched_predecision_both'] and all(v['valid'] for v in r['labels'].values())]
    result={'proposals':len(chosen),'issuer_days':len({(r['day'],r['symbol']) for r in chosen}),
            'sessions':len({r['day'] for r in chosen}),'exact_label_proposals':len(known),'unknown_label_proposals':len(chosen)-len(known),
            'scenarios':{},'rows':[{'day':r['day'],'symbol':r['symbol'],'values':r[kind]['values'],
            'exact_predecision':r['matched_predecision_both'],'labels':r['labels']} for r in chosen]}
    for s in SCENARIOS:
        targets=[r['labels'][s]['value'] for r in known]
        result['scenarios'][s]={'beneficial':sum(v>0 for v in targets),'harmful':sum(v<0 for v in targets),
            'mean_current_holding_advantage_bps':sum(targets)/len(targets) if targets else None,
            'minimum_advantage_bps':min(targets,default=None),'maximum_advantage_bps':max(targets,default=None),
            'mean_episode_advantage_bps':sum(r['labels'][s]['episode_value'] for r in known)/len(known) if known else None}
    return result


def review(output,tests):
    import numpy as np
    import psutil
    from threadpoolctl import threadpool_limits
    launch=read(ROOT/'launch_receipt.json');start=read(ROOT/'supervisor_start.json')
    if launch['command']!=start['command'] or launch['process_created']!=start['process_created']:raise ValueError('Wrong source identity')
    try:
        proc=psutil.Process(launch['pid'])
        if proc.cmdline()==launch['command'] and abs(proc.create_time()-launch['process_created'])<.001:raise ValueError('Source still active')
    except psutil.NoSuchProcess:pass
    done=read(ROOT/'completion.json');p=read(ROOT/'protocol.json');test=read(tests)
    if not done['complete'] or test['exit_code']!=0:raise ValueError('Incomplete source/tests')
    verify(done['programs']+done['bindings']+done['artifacts']+done['origins_before']+done['origins_after'])
    verify(test['programs']+[test['log']])
    bindings=done['programs']+done['bindings']+done['artifacts']+[bound(ROOT/'protocol.json'),bound(ROOT/'completion.json'),bound(ROOT/'post_completion_health_v1.json'),bound(tests)]
    output.mkdir(exist_ok=False);programs=[bound(__file__),bound(TEST)]
    write(output/'protocol.json',{'at':datetime.now(timezone.utc).isoformat(),'programs':programs,'bindings':bindings,
          'scope':'Independent NumPy ridge equations, clocks, weights, original/native forecasts, native feature/state provenance, selected forward affine maps, all prior risk exposures; descriptive economic evidence, no fitting changes/orders.',
          'study_complete':False})
    proc=psutil.Process();write(output/'supervisor_start.json',{'pid':proc.pid,'process_created':proc.create_time(),'command':proc.cmdline()})
    columns=p['own_state_features']+p['incremental_context_features'];n_own=len(p['own_state_features'])
    targets=lines(BASE/'v36_checkpoint_holding_targets/targets.jsonl.gz');groups=defaultdict(dict)
    for r in targets:groups[r['opportunity_id']][r['scenario']]=r
    calendar=read(BASE.parent/'kalcb_aligned_walk_forward_20261010/v3/fit/calendar.json');proofs=read(ROOT/'chronology.json')
    bywindow={r['payoff']['valid_from']:r for r in proofs};models_checked=0
    inherited={r['payoff']['valid_from']:r['payoff'] for r in read(BASE/'v37_checkpoint_holding_models/chronology.json')}
    for wrapped in proofs:
        for scenario in SCENARIOS:
            if wrapped['payoff']['models'][scenario]['coefficients']['peer_context'] != inherited[wrapped['payoff']['valid_from']]['models'][scenario]['coefficients']['peer_context']:
                raise ValueError('Inherited additive coefficient drift')
    risk_raw={}
    for item in done['consumed']:
        verify([item])
        for r in lines(item['path']):risk_raw[r['opportunity_id'],r['scenario']]=r
    def vector(row):
        d={k:v for field in ('own_features','state_features','context_features') for k,v in row[field].items()}
        d=augment(d,columns)
        return [d[k] for k in columns]
    def eligible(row):
        return row['eligibility']=='eligible' and row['held_quantity']>0 and row['pending_buy_quantity']==0 and row['state_features'] is not None
    with threadpool_limits(limits=1):
        for wrapped in proofs:
            proof=wrapped['payoff'];first=proof['valid_from'];allowed=set(calendar['sessions'][:calendar['sessions'].index(first)-1]);cutoff=max(allowed)
            train=[g for _,g in sorted(groups.items()) if all(eligible(g[s]) and g[s]['current_holding_label_valid'] and
                   g[s]['day'] in allowed and g[s]['label_available_at'][:10]<=cutoff for s in SCENARIOS)]
            rows=[g['baseline'] for g in train];names=Counter((r['day'],r['symbol']) for r in rows);days=Counter(d for d,_ in names)
            if proof['mature_sessions']!=len(days) or proof['issuer_days']!=len(names) or proof['dependent_quantities']!=len(rows):raise ValueError('Training support mismatch')
            if signature([r['opportunity_id'] for r in rows])!=proof['training_keys_hash']:raise ValueError('Training set drift')
            w=np.array([1/names[r['day'],r['symbol']]/days[r['day']] for r in rows]);w*=len(names)/w.sum()
            for s in SCENARIOS:
                x=np.array([vector(g[s]) for g in train]);y=np.array([g[s]['current_holding_advantage_bps'] for g in train])
                mean=np.average(x,axis=0,weights=w);sd=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));sd[sd<1e-12]=1.
                source=proof['models'][s]
                for field,values in [('center',mean),('scale',sd),('minimum',x.min(axis=0)),('maximum',x.max(axis=0))]:
                    for a,b in zip(source[field],values):close(a,b)
                for recipe in RECIPES:
                    n=n_own if recipe=='peer_context' else len(columns)
                    z=np.c_[np.ones(len(x)),((x-mean)/sd)[:,:n]]
                    beta=np.linalg.solve(z.T@(w[:,None]*z)+np.diag([0.]+[120.]*n),z.T@(w*y))
                    for a,b in zip(source['coefficients'][recipe],beta):close(a,b)
                    models_checked+=1
            outcomes={}
            for g in groups.values():
                if g['baseline']['day'] not in allowed or not all(eligible(g[s]) for s in SCENARIOS):continue
                rr=[risk_raw[g[s]['opportunity_id'],s]['liquidate'] for s in SCENARIOS]
                if not all(v['checkpoint']['action_submitted'] and v['risk_available_at'] and v['risk_available_at'][:10]<=cutoff for v in rr):continue
                key=(g['baseline']['day'],g['baseline']['symbol'])
                unsafe=any(not v['independent_execution_pass'] or not v['independent_account_pass'] or v['deadline_violation'] or v['unresolved'] or
                           v['actual_changed_exit_quantity']!=g[s]['held_quantity'] for s,v in zip(SCENARIOS,rr))
                outcomes[key]=outcomes.get(key,False) or unsafe
            risk=wrapped['risk']
            if risk['issuer_days']!=len(outcomes) or risk['unsafe_issuer_days']!=sum(outcomes.values()) or risk['mature_sessions']!=len({k[0] for k in outcomes}):raise ValueError('Risk count mismatch')
            if not any(outcomes.values()):close(risk['probability'],1/(len(outcomes)+100))
    original=lines(ROOT/'original_population_forecasts.jsonl.gz');native=lines(ROOT/'native_forecasts.jsonl.gz');original_metrics=defaultdict(list);native_metrics=defaultdict(list)
    for row in original:
        proof=bywindow[row['valid_from']]['payoff'];pair=groups[row['opportunity_id']]
        for recipe in RECIPES:
            expected=prediction(pair,proof,recipe,columns,n_own);check_prediction(row['forecasts'][recipe],expected)
            if expected['supported'] and all(row['labels'][s]['valid'] for s in SCENARIOS):
                for s in SCENARIOS:
                    close(row['labels'][s]['value'],pair[s]['current_holding_advantage_bps'])
                    key=('is' if row['day']<'2026-04-01' else 'oos',recipe,s)
                    original_metrics[key].append({'day':row['day'],'symbol':row['symbol'],'prediction':expected['values'][s],'target':row['labels'][s]['value']})
    contexts={(r['day'],r['symbol']):r for r in lines(BASE/'v31_checkpoint_context_support/contexts.jsonl.gz')}
    comparisons={}
    for r in read(BASE/'v34_checkpoint_contrast_support/selected_parent_comparability.json')['rows']:
        prior,period,s=r['case'].split('/');comparisons[prior,r['position_id'],s]=r
    history=defaultdict(list);calibrations_checked=0
    for wrapped in proofs:
        first=wrapped['payoff']['valid_from'];allowed=set(calendar['sessions'][:calendar['sessions'].index(first)-1])
        for prior in ('control','control_prior125'):
            for recipe in RECIPES:
                for s in SCENARIOS:
                    points=affine_points(history[prior,recipe],allowed,s);expected=affine_equations(points);given=wrapped['calibrations'][prior][recipe][s]
                    for k,v in expected.items():
                        if isinstance(v,bool):
                            if given[k]!=v:raise ValueError('Wrong calibration support')
                        else:close(given[k],v)
                    if signature(points)!=given['points_hash']:raise ValueError('Calibration point drift')
                    calibrations_checked+=1
        for row in (r for r in native if r['valid_from']==first):
            ctx=contexts[row['day'],row['symbol']];matches=[]
            for s,r in row['causal_states'].items():
                c=comparisons[row['prior'],row['position_ids'][s],s];state=c['native'];held=state['held_quantity'];planned=state['planned_quantity']
                if r['opportunity_id']!=c['action_id'] or r['quantity_scale']!=c['quantity_scale'] or r['checkpoint_at']!=ctx['available_at']:
                    raise ValueError('Wrong native action/clock')
                if r['eligibility']!=('parent_exit' if c['parent_exit_already_decided'] else 'eligible'):
                    raise ValueError('Parent complete exit blocker changed')
                for k in ('held_quantity','planned_quantity','pending_buy_quantity'):
                    if r[k]!=state[k]:raise ValueError('Substituted native quantity')
                expected_state=[math.log(held*ctx['completed_close']),math.log1p(held/ctx['consumed_bars'][-1]['volume']),held/planned,
                      state['pending_buy_quantity']/held,(ctx['completed_close']/state['mean_entry_fill_price']-1)/ctx['daily_atr_pct']]
                for k,v in zip(p['own_state_features'][6:],expected_state):close(r['state_features'][k],v)
                if r['own_features']!=ctx['own_features'] or r['context_features']!=ctx['context_features']:raise ValueError('Wrong native market features')
                reference=groups[c['action_id']][s]
                exact=all(r[k]==reference[k] for k in ('held_quantity','planned_quantity','pending_buy_quantity','checkpoint_at')) and all(
                    math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10) for a,b in zip(vector(r),vector(reference)))
                matches.append(exact)
                if row['labels'][s]['valid']:
                    if not exact or not reference['current_holding_label_valid']:raise ValueError('Incomparable native payoff label')
                    close(row['labels'][s]['value'],reference['current_holding_advantage_bps'])
            expected_match=len(matches)==2 and all(matches)
            if expected_match!=row['matched_predecision_both']:raise ValueError('Wrong native transfer label')
            expected=prediction(row['causal_states'],wrapped['payoff'],row['recipe'],columns,n_own);check_prediction(row['raw'],expected)
            risk=wrapped['risk'];rp=risk['supported'] and risk['probability']<.001
            if row['risk_pass']!=rp:raise ValueError('Changed physical gate')
            maps=wrapped['calibrations'][row['prior']][row['recipe']];cal=row['calibrated']
            supported=expected['supported'] and rp and all(maps[s]['supported'] for s in SCENARIOS)
            if cal['supported']!=supported:raise ValueError('Changed calibrated gate')
            if supported:
                for s in SCENARIOS:
                    a=maps[s];close(cal['values'][s],a['center']+a['offset']+a['slope']*(expected['values'][s]-a['center']))
                if cal['positive_both']!=(min(cal['values'].values())>0):raise ValueError('Changed economic comparison')
            if expected_match and all(v['valid'] for v in row['labels'].values()):
                for kind in ('raw','calibrated'):
                    if row[kind]['supported']:
                        for s in SCENARIOS:
                            key=(row['prior'],'is' if row['day']<'2026-04-01' else 'oos',row['recipe'],kind,s)
                            native_metrics[key].append({'day':row['day'],'symbol':row['symbol'],'prediction':row[kind]['values'][s],'target':row['labels'][s]['value']})
            history[row['prior'],row['recipe']].append(row)
    diagnostics={}
    for prior in ('control','control_prior125'):
        for period in ('is','oos'):
            for recipe in RECIPES:
                rows=[r for r in native if r['prior']==prior and r['recipe']==recipe and ('is' if r['day']<'2026-04-01' else 'oos')==period]
                diagnostics['/'.join((prior,period,recipe))]={'rows':len(rows),'raw_supported':sum(r['raw']['supported'] for r in rows),
                    'calibrated_supported':sum(r['calibrated']['supported'] for r in rows),'exact_both':sum(r['matched_predecision_both'] for r in rows),
                    'abstentions':dict(Counter(r['calibrated']['reason'] for r in rows if not r['calibrated']['supported'])),
                    'raw_positive':positive_metrics(rows,'raw'),'calibrated_positive':positive_metrics(rows,'calibrated')}
    result={'passed':True,'checked_coefficients_models':models_checked,'checked_affine_maps':calibrations_checked,
        'checked_original_forecast_pairs':len(original),'checked_native_forecasts':len(native),
        'original_population_metrics':{'/'.join(k):metric(v) for k,v in original_metrics.items()},
        'native_exact_metrics':{'/'.join(k):metric(v) for k,v in native_metrics.items()},'native_diagnostics':diagnostics,
        'limits':'Isolated current-holding action values are not funded returns/DD. Scenarios/priors/quantities are dependent. Unmatched actual quantities have no valid economic evaluation label. Reused OOS is development, not independent validation.',
        'study_complete':False,'native_cases':0}
    write(output/'review.json',result)
    verify(programs+bindings+done['consumed'])
    unexpected=[]
    for name,module in tuple(sys.modules.items()):
        f=getattr(module,'__file__',None)
        if f and Path(f).resolve().is_relative_to(REPO) and Path(f).resolve()!=Path(__file__).resolve():unexpected.append((name,f))
    if unexpected:raise ValueError('Unexpected application imports '+str(unexpected))
    write(output/'completion.json',{'complete':True,'passed':True,'at':datetime.now(timezone.utc).isoformat(),
        'programs':programs,'bindings':bindings,'review':bound(output/'review.json'),'independent_application_origins':[bound(__file__)],
        'native_cases':0,'study_complete':False})
    print(json.dumps({'passed':True,'models':models_checked,'maps':calibrations_checked,'native_records':len(native)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--tests',type=Path,required=True);a=p.parse_args()
    try:review(a.output,a.tests)
    except Exception as error:
        if a.output.exists():write(a.output/'failure.json',{'at':datetime.now(timezone.utc).isoformat(),'error':repr(error),'study_complete':False})
        raise

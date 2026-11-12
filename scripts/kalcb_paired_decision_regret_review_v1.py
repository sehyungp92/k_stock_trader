"""Independent verification of V45 weighted-decision-loss models and reused controls."""
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import argparse, gc, gzip, hashlib, importlib.abc, importlib.machinery, json, math, sys

ROOT = Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
PHYSICAL = ('action_notional_equity', 'action_quantity_last_volume',
            'action_quantity_opening_volume', 'action_low_volume_exit_slices')
PAIRS = ((.25, .5), (.5, 1.))

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def rows(p):
    op = gzip.open if str(p).endswith('.gz') else open
    with op(p, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def save(p, value):
    with p.open('x', encoding='utf-8') as f: json.dump(value, f, indent=2, allow_nan=False)
def bind(p): return {'path': str(p), 'sha256': sha(p)}
def verify(bs):
    for b in bs:
        if sha(b['path']) != b['sha256']: raise ValueError('Changed binding: ' + b['path'])
def close(a, b, abs_tol=1e-7):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=abs_tol): raise ValueError(f'{a} != {b}')

def main(source, out):
    import psutil
    receipt = read(source/'supervisor_start.json')
    try:
        live = psutil.Process(receipt['pid'])
        if abs(live.create_time()-receipt['create_time']) < .001:
            if live.cmdline() == receipt['cmdline']: raise ValueError('Producer still active')
            raise ValueError('Producer identity inconsistent')
    except psutil.NoSuchProcess: pass
    done = read(source/'completion.json'); assert done['complete']
    own = read(source/'protocol.json')
    protocol = read(ROOT/'v44_paired_quantity_value/protocol.json')
    bindings = own['bindings']+own['programs']+done['outputs']
    bindings += [bind(ROOT/'v43_funded_decision_calibration/targets.jsonl.gz'),
                 bind(ROOT/'v43_funded_decision_calibration/review_v1/completion.json')]
    verify(bindings)
    out.mkdir(exist_ok=False)
    save(out/'protocol.json', {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Independent exact quantity/paired-target/chronology/domain/normal-equation/forecast and selected-parent review. No fits, policies or native reruns.',
        'producer_exited': receipt, 'bindings': bindings, 'program': bind(Path(__file__).resolve()),
        'uncertainty': 'Circular5/10-session10000repseed20261011 on isolated contrasts only, with zero unchanged days. Conditional reused-development sensitivity, not independent/multiplicity adjusted or portfolio returns.'})
    proc = psutil.Process()
    save(out/'supervisor_start.json', {'pid':proc.pid, 'create_time':proc.create_time(), 'cmdline':proc.cmdline()})
    profile = read(ROOT/'v25_pending_funding_screen/cases/control/oos/input_profile.json')
    manifest = read(profile['manifest']['path']); frozen = Path(profile['manifest']['path']).parent/'files'
    roots = {Path(k).parts[0].removesuffix('.py') for k in manifest['files'] if k.endswith('.py')}; origins = {}
    class Gate(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split('.')[0] not in roots: return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None: raise ImportError(name)
            if spec.origin and spec.origin not in ('built-in','frozen'):
                file = Path(spec.origin).resolve(); rel = file.relative_to(frozen.resolve()).as_posix(); h = manifest['files'].get(rel)
                if isinstance(h,dict): h=h['sha256']
                if not h or sha(file)!=h: raise ImportError('Unregistered origin '+name)
                origins[name]={'path':str(file),'sha256':h}
            return spec
    sys.path.insert(0,str(frozen)); sys.meta_path.insert(0,Gate())
    import numpy as np, joblib
    from scipy.special import expit
    from threadpoolctl import threadpool_limits
    from backtests.strategies.kalcb import continuation_value as cv
    models = read(source/'models.json'); forecasts = list(rows(source/'forecasts.jsonl.gz'))
    output = {(r['first'],r['scenario'],r['id']):r for r in forecasts}; assert len(output)==len(forecasts)
    controls={(r['first'],r['scenario'],r['id']):r for r in rows(ROOT/'v44_paired_quantity_value/forecasts.jsonl.gz')}
    assert set(output)==set(controls)
    for key,r in output.items():assert {k:v for k,v in r.items() if k!='regret_prediction'}==controls[key]
    cal = read(ROOT.parent/'kalcb_aligned_walk_forward_20261010/v3/fit/calendar.json'); sessions=cal['sessions']
    labels={s:{} for s in ('baseline','combined')}; actions=[]; source_invalid=Counter()
    align=ROOT.parent/'kalcb_selected_target_alignment_20261010/v2'
    for cell in protocol['cells']:
        for scenario in labels:
            folder=align/'labels'/f"{cell['period']}_opening_{scenario}"; g=read(folder/'grid_manifest.json')
            for name in sorted(g['files']):
                if not name.endswith('.jsonl'): continue
                for row in rows(folder/name):
                    key=row['opportunity_id']; assert key not in labels[scenario]
                    labels[scenario][key]={k:row[k] for k in ('label_valid','label_available_at','net_value_equity_bps')}
                    if not row['label_valid']: source_invalid[scenario]+=1
        wanted=set(cell['sessions']); prepared=Path(cell['prepared'])
        for r in rows(prepared/'opportunities.jsonl'):
            if r['trade_date'] not in wanted or not r['action']['ready'] or not r['guard_at_rank1']: continue
            full=int(r['action']['quantity']); price=r['opening_bars'][-1]['close']
            for scale in (.25,.5,1.):
                qty=math.floor(full*scale); f=dict(r['features'])
                for k in PHYSICAL: f[k]=r['features'][k]*qty/full
                f['action_quantity_scale']=scale
                close(f['action_notional_equity'],qty*price/1e8)
                actions.append({'id':r['opportunity_id']+f'/q{scale:g}', 'base_id':r['opportunity_id'],
                    'day':r['trade_date'],'q':scale,'qty':qty,'notional':qty*price/1e8,
                    'available_at':r['feature_available_at'],'features':f})
    actions.sort(key=lambda r:(r['day'],r['base_id'],r['q'])); ids={r['id']:i for i,r in enumerate(actions)}
    assert len(ids)==len(actions) and all(set(x)==set(ids) for x in labels.values())
    bases=sorted({r['base_id'] for r in actions}); pairs=[]; aliases=0
    for base in bases:
        for lo,hi in PAIRS:
            i=ids[base+f'/q{lo:g}']; j=ids[base+f'/q{hi:g}']; low=actions[i]; high=actions[j]
            if low['qty']==high['qty']: aliases+=1; continue
            assert low['qty']<high['qty'] and low['available_at']==high['available_at']
            ls=[labels[s][a['id']] for s in labels for a in (low,high)]
            valid=all(x['label_valid'] for x in ls)
            maturity=max((x['label_available_at'] for x in ls if x['label_available_at']),default=None)
            pairs.append({'id':base+f'/q{lo:g}_q{hi:g}','base_id':base,'day':low['day'], 'i':i,'j':j,
                'low':lo,'high':hi,'valid':valid,'mature':maturity,
                'physical':[high['features'][k]-low['features'][k] for k in PHYSICAL],
                'values':{s:labels[s][high['id']]['net_value_equity_bps']-labels[s][low['id']]['net_value_equity_bps'] if valid else None for s in labels}})
    target_lookup={r['id']:r for r in pairs}; normal_checks=0; forecast_checks=0; max_normal_residual=0.
    for model in models:
        first=model['first']; scenario=model['scenario']; allowed=set(sessions[:sessions.index(first)-1]); cutoff=datetime.fromisoformat(first+'T00:00:00+09:00')
        training=[r for r in pairs if r['day'] in allowed and r['valid'] and r['mature'] and datetime.fromisoformat(r['mature'])<cutoff]
        assert [r['id'] for r in training]==model['training_ids']
        days=len({r['day'] for r in training}); nbase=len({r['base_id'] for r in training})
        assert (days,nbase)==(model['mature_sessions'],model['issuer_days'])
        assert model['supported']==(days>=60 and nbase>=100)
        native_map=joblib.load(model['frozen_map_model_path'])
        fold=next(x for x in protocol['folds'] if x['valid_from']==first and x['scenario']==scenario)
        matrix=cv.matrix(actions,fold['feature_columns']); transformed=native_map.features.transform(matrix)
        phi=np.column_stack([np.ones(len(actions)),transformed])*np.array([r['q'] for r in actions])[:,None]
        xx=np.array([phi[r['j']]-phi[r['i']] for r in training]); yy=np.array([r['values'][scenario] for r in training])
        if model['supported']:
            cnt=Counter(r['base_id'] for r in training); daybases=defaultdict(set)
            for r in training: daybases[r['day']].add(r['base_id'])
            ww=np.array([nbase/days/len(daybases[r['day']])/cnt[r['base_id']] for r in training])
            np.testing.assert_allclose(ww,model['base_weights'],atol=1e-12,rtol=1e-12)
            scale=float(np.dot(ww,abs(yy))/sum(ww));close(scale,model['training_absolute_value_scale'])
            ew=ww*abs(yy)/scale;np.testing.assert_allclose(ew,model['weights'],atol=1e-10,rtol=1e-10)
            b=np.array(model['coefficients'])
            with threadpool_limits(limits=1): residual=xx.T@(ew*(expit(xx@b)-(yy>0)))+120*b
            max_normal_residual=max(max_normal_residual,float(abs(residual).max()))
            assert np.max(abs(residual))<1e-4; normal_checks+=1
            for lo,hi in PAIRS:
                ix=[i for i,r in enumerate(training) if r['low']==lo and r['high']==hi]; subset=[training[i] for i in ix]
                if len(ix)<20 or len({r['day'] for r in subset})<8: assert f'{lo:g}_{hi:g}' not in model['domains']; continue
                domain=model['domains'][f'{lo:g}_{hi:g}']; phys=np.array([r['physical'] for r in subset])
                np.testing.assert_allclose(domain['min'],phys.min(0),atol=1e-12); np.testing.assert_allclose(domain['max'],phys.max(0),atol=1e-12)
                assert domain['issuer_days']==len(ix) and domain['sessions']==len({r['day'] for r in subset})
        for pair in pairs:
            if not first<=pair['day']<=model['through']:continue
            row=output[first,scenario,pair['id']]
            assert pair['valid']==row['target_valid'] and pair['mature']==row['mature_at']
            if pair['valid']:close(pair['values'][scenario],row['value'])
            domain=model['domains'].get(f"{pair['low']:g}_{pair['high']:g}")
            supported=bool(model['supported'] and domain and all(a<=v<=b for a,v,b in zip(domain['min'],pair['physical'],domain['max'])))
            pred=row['regret_prediction'];assert pred['supported']==supported
            if supported:
                expected=float((phi[pair['j']]-phi[pair['i']])@np.array(model['coefficients']))
                close(expected,pred['score'])
            else:assert pred['score'] is None
            forecast_checks+=1
        print(json.dumps({'reviewed':first,'scenario':scenario,'normal_equations':normal_checks}),flush=True)
        del matrix,transformed,phi,xx,yy,native_map;gc.collect()
    audit={r['action_id']:r for r in rows(ROOT/'v26_opening_decision_audit_v2/decisions.jsonl.gz')}
    funded={(r['parent'],r['scenario'],r['id']):r for r in rows(ROOT/'v43_funded_decision_calibration/targets.jsonl.gz')}
    assessment=read(source/'assessment.json'); joined={(r['id'],r['scenario']):r for r in forecasts}; changes=[]
    for r in forecasts:
        if r['scenario']!='baseline':continue
        c=joined[r['id'],'combined'];ps=[x['regret_prediction'] for x in (r,c)]
        if not all(x['supported'] and x['score']<0 for x in ps) or not all(x['lower_physical_capacity_pass'] for x in (r,c)):continue
        for parent in r['high_parent_selections']:
            changes.append({'parent':parent,'period':'is' if r['day']<='2026-03-31' else 'oos','id':r['id'],'base_id':r['base_id'],
                'day':r['day'],'low':r['low'],'high':r['high'],
                'scores':{s:joined[r['id'],s]['regret_prediction']['score'] for s in labels},
                'delta_isolated_equity_bps':{s:-joined[r['id'],s]['value'] if joined[r['id'],s]['target_valid'] else None for s in labels}})
    assert sorted(changes,key=lambda r:(r['parent'],r['id']))==sorted(assessment['changes'],key=lambda r:(r['parent'],r['id']))
    economics={};rng=np.random.default_rng(20261011)
    for key,summary in assessment['summary'].items():
        parent,period=key.split('/');cs=[r for r in changes if (r['parent'],r['period'])==(parent,period)]
        assert summary['changed_issuer_days']==len({r['base_id'] for r in cs}) and summary['changed_sessions']==len({r['day'] for r in cs})
        dates=[d for d in sessions if ('2025-08-07'<=d<='2026-03-31' if period=='is' else '2026-04-01'<=d<='2026-07-10')]
        result={'changed_issuer_days':len(cs),'changed_sessions':len({r['day'] for r in cs}),'scenarios':{}}
        for scenario in labels:
            ds=dict.fromkeys(dates,0.);beneficial=0;winners=0;lost=0.;matched=0;month=defaultdict(float);issuer=defaultdict(float)
            for r in cs:
                pair=target_lookup[r['id']];high=actions[pair['j']];delta=r['delta_isolated_equity_bps'][scenario]
                assert delta is not None;ds[r['day']]+=delta;month[r['day'][:7]]+=delta;issuer[r['base_id'].split('/')[1]]+=delta
                if delta>0:beneficial+=1
                if delta<0 and labels[scenario][high['id']]['net_value_equity_bps']>0:winners+=1;lost-=delta
                native=funded.get((parent,scenario,high['id']))
                if native and native['native']['submitted_quantity']==high['qty']:matched+=1
            close(sum(ds.values()),summary['isolated_contrast_sum_not_portfolio_pnl'][scenario]);ci={};v=np.array(list(ds.values()));n=len(v)
            for block in (5,10):
                starts=rng.integers(0,n,size=(10000,math.ceil(n/block)));ix=(starts[:,:,None]+np.arange(block))%n
                boot=v[ix.reshape(10000,-1)[:,:n]].sum(1)
                ci[str(block)]={'isolated_sum_95pct':np.quantile(boot,[.025,.975]).tolist(),'probability_positive':float(np.mean(boot>0))}
            result['scenarios'][scenario]={'isolated_sum_equity_bps_not_portfolio':sum(ds.values()),'beneficial_changes':beneficial,
                'profitable_high_actions_harmed':winners,'positive_tail_value_sacrificed_equity_bps':lost,
                'exact_parent_submitted_quantity_matches':matched,'unmatched_or_not_submitted':len(cs)-matched,
                'quantity_match_is_not_full_state_comparability':True,'months':dict(month),'issuers':dict(issuer),'conditional_intervals':ci}
        economics[key]=result
    errors={};zero_scores=0
    for period in ('is','oos'):
      for scenario in labels:
       for cohort in ('all','control','control_prior125'):
        rs=[r for r in forecasts if r['scenario']==scenario and ('is' if r['day']<='2026-03-31' else 'oos')==period and r['target_valid'] and r['regret_prediction']['supported'] and (cohort=='all' or cohort in r['high_parent_selections'])]
        day=defaultdict(list)
        for r in rs:day[r['day']].append(r)
        losses={}
        for recipe in ('pair_mean','paired_state','decision_regret'):
            ds=[]
            for group in day.values():
                costs=[]
                for r in group:
                    score=r['regret_prediction']['score'] if recipe=='decision_regret' else r['predictions'][recipe]['value']
                    if score==0:zero_scores+=1
                    costs.append(abs(r['value']) if (score>0)!=(r['value']>0) else 0.)
                ds.append(sum(costs)/len(costs))
            losses[recipe]=sum(ds)/len(ds)
        k=period+'/'+scenario+'/'+cohort;errors[k]={'observations':len(rs),'sessions':len(day),'session_balanced_economic_mistake_equity_bps':losses}
        for recipe,value in losses.items():close(value,assessment['economic_decision_errors'][k]['session_balanced_economic_mistake_equity_bps'][recipe])
    assert zero_scores==0,'Tie policy requires explicit inspection'
    readiness={}
    for parent in ('control','control_prior125'):
        is_summary=assessment['summary'][parent+'/is'];oos_summary=assessment['summary'][parent+'/oos']
        frozen={'is_selected_economic_error_improves_both':all(errors['is/'+s+'/'+parent]['session_balanced_economic_mistake_equity_bps']['decision_regret']<errors['is/'+s+'/'+parent]['session_balanced_economic_mistake_equity_bps']['paired_state'] for s in labels),
            'is_intervention_support':is_summary['changed_issuer_days']>=20 and is_summary['changed_sessions']>=8,
            'is_isolated_direction_nonnegative_both':all(v>=0 for v in is_summary['isolated_contrast_sum_not_portfolio_pnl'].values())}
        readiness[parent]={'preregistered_initial_screen':frozen,'initial_screen_passed':all(frozen.values()),
            'oos_isolated_direction_nonnegative_both':all(v>=0 for v in oos_summary['isolated_contrast_sum_not_portfolio_pnl'].values()),
            'selected_forward_calibration_and_native_state_qualification_complete':False,'native_dispatch_authorized_by_producer':False}
    verify(bindings);verify(list(origins.values()))
    result={'passed':True,'normal_equations':normal_checks,'maximum_gradient_residual':max_normal_residual,'forecast_values':forecast_checks,
        'paired_targets':len(pairs),'source_actions':len(actions),'origins':origins,'controls_reused_exact':len(controls),
        'economics':economics,'errors':errors,'readiness':readiness,
        'scope':'Selected-management cash/fill/MTM source qualification and V44 controls authenticated/reused. No original labels or native portfolio rerun. Isolated values are not portfolio returns/DD/activity gains.'}
    save(out/'review.json',result);save(out/'completion.json',{'passed':True,'study_complete':False,'review':bind(out/'review.json'),'completed_at_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'passed':True,'normal_equations':normal_checks,'forecast_values':forecast_checks,'readiness':readiness}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.source,a.output)

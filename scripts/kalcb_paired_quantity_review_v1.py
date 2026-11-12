"""Independent reconstruction of the fixed V44 paired-value experiment."""
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
    protocol = read(source/'protocol.json')
    bindings = protocol['bindings']+protocol['programs']+done['outputs']
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
    from threadpoolctl import threadpool_limits
    from backtests.strategies.kalcb import continuation_value as cv
    models = read(source/'models.json'); forecasts = list(rows(source/'forecasts.jsonl.gz'))
    output = {(r['first'],r['scenario'],r['id']):r for r in forecasts}; assert len(output)==len(forecasts)
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
            np.testing.assert_allclose(ww,model['weights'],atol=1e-12,rtol=1e-12)
            b=np.array(model['coefficients'])
            with threadpool_limits(limits=1): residual=xx.T@(ww*(xx@b-yy))+120*b
            max_normal_residual=max(max_normal_residual,float(abs(residual).max()))
            assert np.max(abs(residual))<1e-5; normal_checks+=1
            for lo,hi in PAIRS:
                ix=[i for i,r in enumerate(training) if r['low']==lo and r['high']==hi]; subset=[training[i] for i in ix]
                if len(ix)<20 or len({r['day'] for r in subset})<8: assert f'{lo:g}_{hi:g}' not in model['domains']; continue
                domain=model['domains'][f'{lo:g}_{hi:g}']; phys=np.array([r['physical'] for r in subset])
                np.testing.assert_allclose(domain['min'],phys.min(0),atol=1e-12); np.testing.assert_allclose(domain['max'],phys.max(0),atol=1e-12)
                assert domain['issuer_days']==len(ix) and domain['sessions']==len({r['day'] for r in subset})
                close(domain['mean_control'],float(ww[ix]@yy[ix]/(sum(ww[ix])+100)))
        for pair in pairs:
            if not first<=pair['day']<=model['through']:continue
            row=output[first,scenario,pair['id']]
            assert pair['valid']==row['target_valid'] and pair['mature']==row['mature_at']
            if pair['valid']:close(pair['values'][scenario],row['value'])
            domain=model['domains'].get(f"{pair['low']:g}_{pair['high']:g}")
            supported=bool(model['supported'] and domain and all(a<=v<=b for a,v,b in zip(domain['min'],pair['physical'],domain['max'])))
            for recipe,pred in row['predictions'].items():
                assert pred['supported']==supported
                if supported:
                    expected=domain['mean_control'] if recipe=='pair_mean' else float((phi[pair['j']]-phi[pair['i']])@np.array(model['coefficients']))
                    close(expected,pred['value'])
                else:assert pred['value'] is None
                forecast_checks+=1
        print(json.dumps({'reviewed':first,'scenario':scenario,'normal_equations':normal_checks}),flush=True)
        del matrix,transformed,phi,xx,yy,native_map;gc.collect()
    audit={r['action_id']:r for r in rows(ROOT/'v26_opening_decision_audit_v2/decisions.jsonl.gz')}
    funded={(r['parent'],r['scenario'],r['id']):r for r in rows(ROOT/'v43_funded_decision_calibration/targets.jsonl.gz')}
    assessment=read(source/'assessment.json'); changes=[]
    joined={(r['id'],r['scenario']):r for r in forecasts}
    for row in forecasts:
        pair=target_lookup[row['id']];low=actions[pair['i']];high=actions[pair['j']]
        assert row['high_parent_selections']==list(audit[high['id']]['selected'])
        assert row['lower_physical_capacity_pass']==bool(audit[low['id']]['physical_risk_pass'] and audit[low['id']]['capacity_pass'])
        close(row['original_raw_difference'],audit[high['id']]['raw'][row['scenario']]-audit[low['id']]['raw'][row['scenario']])
        if row['scenario']!='baseline':continue
        other=joined[row['id'],'combined']
        for parent in row['high_parent_selections']:
            for recipe in ('pair_mean','paired_state'):
                preds=[x['predictions'][recipe] for x in (row,other)]
                if not all(x['lower_physical_capacity_pass'] for x in (row,other)) or not all(x['supported'] and x['value']<0 for x in preds):continue
                changes.append({'parent':parent,'recipe':recipe,'period':'is' if row['day']<='2026-03-31' else 'oos',
                    'id':row['id'],'base_id':row['base_id'],'day':row['day'],'low':row['low'],'high':row['high'],
                    'forecasts':{s:joined[row['id'],s]['predictions'][recipe]['value'] for s in labels},
                    'delta_isolated_equity_bps':{s:-pair['values'][s] if pair['valid'] else None for s in labels}})
    assert sorted(changes,key=lambda r:(r['parent'],r['recipe'],r['id']))==sorted(assessment['changes'],key=lambda r:(r['parent'],r['recipe'],r['id']))
    economics={}; rng=np.random.default_rng(20261011)
    for key, summary in assessment['summary'].items():
        parent,recipe,period=key.split('/'); cs=[r for r in changes if (r['parent'],r['recipe'],r['period'])==(parent,recipe,period)]
        assert summary['changed_issuer_days']==len({r['base_id'] for r in cs}) and summary['changed_sessions']==len({r['day'] for r in cs})
        dates=[d for d in sessions if ('2025-08-07'<=d<='2026-03-31' if period=='is' else '2026-04-01'<=d<='2026-07-10')]
        item={'changed_issuer_days':len(cs),'changed_sessions':len({r['day'] for r in cs}),'scenarios':{}}
        for scenario in labels:
            ds={d:0. for d in dates}; wins_lost=0; tails_lost=0.; beneficial=0; native_matches=0; unmatched=0; month=defaultdict(float); issuer=defaultdict(float)
            for r in cs:
                pair=target_lookup[r['id']];high=actions[pair['j']];lo=actions[pair['i']];delta=r['delta_isolated_equity_bps'][scenario]
                assert delta is not None; ds[r['day']]+=delta;month[r['day'][:7]]+=delta;issuer[r['base_id'].split('/')[1]]+=delta
                if delta>0:beneficial+=1
                hval=labels[scenario][high['id']]['net_value_equity_bps']
                if delta<0 and hval>0:wins_lost+=1;tails_lost-=delta
                native=funded.get((parent,scenario,high['id']))
                if native and native['native']['submitted_quantity']==high['qty']:native_matches+=1
                else:unmatched+=1
            close(sum(ds.values()),summary['isolated_contrast_sum_not_portfolio_pnl'][scenario])
            ci={};v=np.array(list(ds.values()));n=len(v)
            for block in (5,10):
                starts=rng.integers(0,n,size=(10000,math.ceil(n/block)))
                indices=(starts[:,:,None]+np.arange(block))%n;boot=v[indices.reshape(10000,-1)[:,:n]].sum(axis=1)
                ci[str(block)]={'isolated_sum_95pct':np.quantile(boot,[.025,.975]).tolist(),'probability_positive':float(np.mean(boot>0))}
            item['scenarios'][scenario]={'isolated_sum_equity_bps_not_portfolio':sum(ds.values()),'beneficial_changes':beneficial,
                'profitable_high_actions_harmed':wins_lost,'positive_tail_value_sacrificed_equity_bps':tails_lost,
                'exact_parent_submitted_quantity_matches':native_matches,'unmatched_or_not_submitted':unmatched,
                'quantity_match_is_not_full_state_comparability':True,'months':dict(month),'issuers':dict(issuer),'conditional_intervals':ci}
        economics[key]=item
    errors={}
    for period in ('is','oos'):
        for scenario in labels:
            for cohort in ('all','control','control_prior125'):
                selected=[r for r in forecasts if r['scenario']==scenario and ('is' if r['day']<='2026-03-31' else 'oos')==period and r['target_valid'] and all(p['supported'] for p in r['predictions'].values()) and (cohort=='all' or cohort in r['high_parent_selections'])]
                grouped=defaultdict(list)
                for r in selected:grouped[r['day']].append(r)
                values={recipe:float(np.mean([np.mean([(r['predictions'][recipe]['value']-r['value'])**2 for r in day]) for day in grouped.values()])) for recipe in ('pair_mean','paired_state')}
                if cohort=='all':
                    for k,v in values.items():close(v,assessment['session_balanced_pair_errors'][period+'/'+scenario][k])
                errors[period+'/'+scenario+'/'+cohort]={'observations':len(selected),'sessions':len(grouped),'mse':values}
    readiness={}
    for parent in ('control','control_prior125'):
        s=assessment['summary'][parent+'/paired_state/is']
        readiness[parent]={'is_error_improves_both':all(errors['is/'+x+'/all']['mse']['paired_state']<errors['is/'+x+'/all']['mse']['pair_mean'] for x in labels),
            'is_intervention_support':s['changed_issuer_days']>=20 and s['changed_sessions']>=8,
            'is_isolated_direction_nonnegative_both':all(v>=0 for v in s['isolated_contrast_sum_not_portfolio_pnl'].values()),
            'oos_isolated_direction_nonnegative_both':all(v>=0 for v in assessment['summary'][parent+'/paired_state/oos']['isolated_contrast_sum_not_portfolio_pnl'].values())}
        readiness[parent]['native_candidate_supported']=all(readiness[parent].values())
    verify(bindings);verify(list(origins.values()));assert source_invalid==Counter(assessment['invalid_source_actions_retained'])
    result={'passed':True,'normal_equations':normal_checks,'max_normal_residual':max_normal_residual,'forecast_values':forecast_checks,
        'paired_targets':len(pairs),'source_actions':len(actions),'integer_aliases':aliases,'invalid_source_actions':dict(source_invalid),
        'economics':economics,'errors':errors,'readiness':readiness,'origins':origins,
        'scope':'Original source label cash/fill/MTM qualification reused, not rerun. No portfolio or native return/DD improvement measured. Do not dispatch a rejected candidate.'}
    save(out/'review.json',result)
    save(out/'completion.json',{'passed':True,'study_complete':False,'review':bind(out/'review.json'),'completed_at_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'passed':True,'readiness':readiness,'normal_equations':normal_checks,'forecast_values':forecast_checks}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.source,a.output)

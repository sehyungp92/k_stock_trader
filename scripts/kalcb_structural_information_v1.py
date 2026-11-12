"""Fixed C9-B diagnostics: relative entry correction and belief-conditioned renewal."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.kalcb_entry_lifecycle_panel_v2 import read,write,bound,verify,lines,stamp
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_structural_preparation_20261016')
PANEL=ROOT/'entry_lifecycle_v2'
AUDIT=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011/v26_opening_decision_audit_v2')
DESIGN=REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0033-information-design.md'
SCENARIOS=('baseline','combined')
RENEWALS=('ridge_B','ridge_N','tree_B','tree_N')


def balance(rows):
    names=defaultdict(set);counts=Counter((r['day'],r['base_id']) for r in rows)
    for r in rows:names[r['day']].add(r['base_id'])
    w=np.array([1/len(names[r['day']])/counts[(r['day'],r['base_id'])] for r in rows])
    return w/w.mean()


def demean(X,rows):
    X=np.asarray(X,dtype=float);result=X.copy();groups=defaultdict(list)
    for i,r in enumerate(rows):groups[(r['day'],r['quantity_scale'])].append(i)
    for ids in groups.values():result[ids]-=X[ids].mean(axis=0)
    return result


def ridge_fit(X,y,w,centered=False):
    mean=np.average(X,axis=0,weights=w);scale=np.sqrt(np.average((X-mean)**2,axis=0,weights=w));scale=np.maximum(scale,1e-9)
    z=np.clip((X-mean)/scale,-6,6);zc=np.average(z,axis=0,weights=w);z-=zc
    ym=0. if centered else float(np.average(y,weights=w))
    beta=np.linalg.solve(z.T@(w[:,None]*z)+120*np.eye(X.shape[1]),z.T@(w*(y-ym)))
    return {'mean':mean,'scale':scale,'zc':zc,'ym':ym,'beta':beta,'minimum':X.min(axis=0),'maximum':X.max(axis=0)}


def ridge_predict(model,X):
    return model['ym']+(np.clip((X-model['mean'])/model['scale'],-6,6)-model['zc'])@model['beta']


def entry_features(r,s):
    f=r['features'];atr=f['daily_atr_pct']
    keys=('first30_gap','first30_vwap_ret','first30_range_atr','first30_rel_volume_log','path_efficiency',
          'path_above_vwap_share','sector_intraday_breadth','market_adjusted_first30_ret','daily_return_20d',
          'path_last15_return','path_return_acceleration')
    if atr is None or atr<=0 or any(f.get(k) is None for k in keys):return None
    values=[math.log(atr),f['first30_gap']/atr,f['first30_vwap_ret']/atr,f['first30_range_atr'],f['first30_rel_volume_log'],
      f['path_efficiency'],f['path_above_vwap_share'],f['sector_intraday_breadth'],f['market_adjusted_first30_ret']/atr,
      f['daily_return_20d']/atr,f['path_last15_return']/atr,f['path_return_acceleration']/atr,
      r['quantity_scale'],r['calibrated_core'][s]/r['notional_equity']]
    return values if all(math.isfinite(x) for x in values) else None


def renewal_features(r,a,basis):
    x=r['features35'][:6]+[r['quantity_scale']]+[a['calibrated_core'][s]/a['notional_equity'] for s in SCENARIOS]
    if basis=='N':x+=r['features35'][6:]+[r['pending_qty_at35']/r['planned_qty'],r['quantity_at35']/r['planned_qty']]
    return x


def mature(r,days,origin,scenario=None):
    # Caller passes the same source calendar; current/embargo labels can never enter.
    if r['day'] not in set(days[:max(0,origin-1)]):return False
    lab=r['labels'][scenario] if scenario else r
    return bool(lab['label_valid'] and lab.get('label_available_at') and stamp(lab['label_available_at'])<stamp(days[origin]+'T00:00:00+09:00'))


def metrics(rows,key):
    days=defaultdict(list)
    for r in rows:
        if r.get(key) is not None:days[r['day']].append(r[key])
    return {'rows':len(rows),'issuer_days':len({r['base_id'] for r in rows}),'sessions':len({r['day'] for r in rows}),
      'session_mean_reference_equity_bps':float(np.mean([np.mean(v) for v in days.values()])) if days else None,
      'sum_isolated_reference_equity_bps':sum(v for values in days.values() for v in values),
      'positive_mass_reference_equity_bps':sum(max(0,v) for values in days.values() for v in values),
      'negative_mass_reference_equity_bps':sum(min(0,v) for values in days.values() for v in values)}


def shadow_select(rows,predictions,recipe):
    """Preserve day/q count maxima and original total planned-notional budget."""
    selected={r['action_id'] for r in rows if 'control' in r['selected']}
    budget=sum(r['notional_equity'] for r in rows if r['action_id'] in selected)
    quotas=Counter(r['quantity_scale'] for r in rows if r['action_id'] in selected)
    proposals=[]
    for r in rows:
        p=predictions.get(r['action_id'])
        if p and r['physical_risk_pass'] and r['capacity_pass'] and min(p[recipe].values())>0:
            proposals.append((min(p[recipe].values()),r))
    out=set();names=set();used=0.;counts=Counter()
    for _,r in sorted(proposals,key=lambda v:(-v[0],v[1]['action_id'])):
        if r['base_id'] in names or counts[r['quantity_scale']]>=quotas[r['quantity_scale']]:continue
        if used+r['notional_equity']>budget+1e-12:continue
        out.add(r['action_id']);names.add(r['base_id']);counts[r['quantity_scale']]+=1;used+=r['notional_equity']
    return out


def run(out):
    out.mkdir(exist_ok=False);t0=time.monotonic()
    import psutil
    me=psutil.Process();write(out/'process.json',{'pid':me.pid,'created_epoch':me.create_time(),'argv':me.cmdline()})
    old=read(PANEL/'completion.json')
    for b in (old['registration'],old['panel'],old['assessment']):verify(b)
    audit_receipt=read(AUDIT/'completion.json');ab=next(b for b in audit_receipt['artifacts'] if Path(b['path']).name=='decisions.jsonl.gz');verify(ab)
    days=[r['day'] for r in read(Path('E:/CodexResearch/k_stock_trader/holding_account_20261013/study_v2/cases/control/is/baseline/daily.json'))]+[r['day'] for r in read(Path('E:/CodexResearch/k_stock_trader/holding_account_20261013/study_v2/cases/control/oos/baseline/daily.json'))]
    if len(days)!=225 or days!=sorted(set(days)):raise ValueError('Calendar')
    reg={'at_utc':datetime.now(timezone.utc).isoformat(),'program':bound(__file__),'design':bound(DESIGN),
         'tests':bound(REPO/'tests/test_kalcb_structural_information_v1.py'),
         'inputs':[old['registration'],old['panel'],old['assessment'],ab,bound(ROOT/'d3_support_v1/assessment.json'),
                   bound(REPO/'scripts/kalcb_entry_lifecycle_panel_v2.py')]+[
                   bound(Path('E:/CodexResearch/k_stock_trader/holding_account_20261013/study_v2/cases/control')/period/'baseline/daily.json') for period in ('is','oos')],
         'origins':[days[i] for i in range(61,len(days),20)],'maximum_fits':108,'calendar':days,'native_cases':0}
    write(out/'registration.json',reg)
    all_a=list(lines(AUDIT/'decisions.jsonl.gz'));lookup={r['action_id']:r for r in all_a}
    eligible=[r for r in all_a if r['physical_risk_pass'] and r['capacity_pass'] and all(entry_features(r,s) is not None for s in SCENARIOS)]
    data={s:[] for s in SCENARIOS};exclusions=Counter()
    for r in lines(PANEL/'panel.jsonl.gz'):
        a=lookup.get(r['action_id'])
        if not a:continue
        if not a['physical_risk_pass'] or not a['capacity_pass']:continue
        if not r['prefix_supported']:exclusions['missing_prefix']+=1;continue
        if r['original_exit_intent_by35']:exclusions['original_exit_intent']+=1;continue
        if r['quantity_at35']==0 and r['pending_qty_at35']==0:exclusions['flat_no_pending']+=1;continue
        r['x_B']=renewal_features(r,a,'B');r['x_N']=renewal_features(r,a,'N')
        data[r['scenario']].append(r)
    ep={};rp=[];model_proofs=[];fit_count=0
    starts=list(range(61,len(days),20))
    for n,j in enumerate(starts):
        last=starts[n+1] if n+1<len(starts) else len(days);evaldays=set(days[j:last])
        eval_e=[r for r in eligible if r['day'] in evaldays]
        models={};proof={'origin':days[j],'observed_through':days[j-2],'training':{},'renewal':{}}
        for s in SCENARIOS:
            tr=[r for r in eligible if mature(r,days,j,s)]
            X=np.array([entry_features(r,s) for r in tr]);y=np.array([(r['labels'][s]['net_value_equity_bps']-r['calibrated_core'][s])/r['notional_equity'] for r in tr]);w=balance(tr)
            A=ridge_fit(X,y,w);R=ridge_fit(demean(X,tr),demean(y,tr),w,True);fit_count+=2
            xe=np.array([entry_features(r,s) for r in eval_e]);pa=ridge_predict(A,xe);pr=ridge_predict(R,demean(xe,eval_e));pr=demean(pr,eval_e)
            corrected=np.array([r['calibrated_core'][s]/r['notional_equity'] for r in eval_e])+pa
            # Preserve A's full current-cohort mean; only relative order changes.
            pa_mean=corrected-demean(corrected,eval_e)
            core=np.array([r['calibrated_core'][s]/r['notional_equity'] for r in eval_e]);relative=demean(core,eval_e)+pr+pa_mean
            for r,a,v in zip(eval_e,corrected,relative):
                p=ep.setdefault(r['action_id'],{'day':r['day'],'base_id':r['base_id'],'A':{},'R':{},'origin':days[j]})
                p['A'][s]=float(a);p['R'][s]=float(v)
            proof['training'][s]={'rows':len(tr),'issuer_days':len({r['base_id'] for r in tr}),'sessions':len({r['day'] for r in tr})}
            models[s]={}
            trr=[r for r in data[s] if mature(r,days,j)]
            yr=np.array([r['remaining_value_planned_bps'] for r in trr]);wr=balance(trr)
            for basis in ('B','N'):
                xx=np.array([r['x_'+basis] for r in trr]);rr=ridge_fit(xx,yr,wr)
                tree=HistGradientBoostingRegressor(loss='squared_error',max_iter=50,learning_rate=.05,max_leaf_nodes=4,max_depth=2,
                     min_samples_leaf=60,l2_regularization=10,early_stopping=False,random_state=3301).fit(xx,yr,sample_weight=wr)
                models[s]['ridge_'+basis]=rr;models[s]['tree_'+basis]={'tree':tree,'minimum':xx.min(axis=0),'maximum':xx.max(axis=0)};fit_count+=2
            proof['renewal'][s]={'rows':len(trr),'issuer_days':len({r['base_id'] for r in trr}),'sessions':len({r['day'] for r in trr})}
        joblib.dump(models,out/(f'models_{n:02}.joblib'))
        for actual_s in SCENARIOS:
            evaluation=[r for r in data[actual_s] if r['day'] in evaldays]
            arrays={b:np.array([r['x_'+b] for r in evaluation]) for b in ('B','N')};preds={};domains={}
            for recipe in RENEWALS:
                x=arrays[recipe[-1]]
                for fit_s in SCENARIOS:
                    m=models[fit_s][recipe]
                    preds[(recipe,fit_s)]=ridge_predict(m,x) if recipe.startswith('ridge') else m['tree'].predict(x)
                    domains[(recipe,fit_s)]=np.any((x<m['minimum'])|(x>m['maximum']),axis=1)
            for i,r in enumerate(evaluation):
                held_fraction=r['quantity_at35']*r['close35']/r['planned_notional'];a=lookup[r['action_id']]
                pp={};choices={};outside={}
                for recipe in RENEWALS:
                    pp[recipe]={s:float(preds[(recipe,s)][i]) for s in SCENARIOS}
                    choices[recipe]=all(pp[recipe][s]<-held_fraction*(23 if s=='baseline' else 33) for s in SCENARIOS)
                    outside[recipe]={s:bool(domains[(recipe,s)][i]) for s in SCENARIOS}
                value=None
                if r['label_valid']:
                    value=(-r['remaining_net_after35_krw']-r['quantity_at35']*r['close35']*(23 if actual_s=='baseline' else 33)/1e4)/1e4
                rp.append({'day':r['day'],'base_id':r['base_id'],'action_id':r['action_id'],'period':r['period'],'scenario':actual_s,
                   'selected':r['selected_prior100'],'actually_traded':r['actual_prior100'],'quantity_scale':r['quantity_scale'],
                   'pending':r['pending_qty_at35']>0,'planned_notional':r['planned_notional'],'label_valid':r['label_valid'],
                   'original_net_krw':r['net_reconstructed_krw'],'remaining_net_krw':r['remaining_net_after35_krw'],
                   'optimistic_delta_reference_bps':value,'predictions':pp,'choices':choices,'outside':outside,'origin':days[j]})
        model_proofs.append(proof)
        write(out/'status.json',{'phase':'fitting','origins_complete':n+1,'origins_total':len(starts),'fits':fit_count,'elapsed_s':time.monotonic()-t0},True)
        print(json.dumps({'origin':days[j],'fits':fit_count,'elapsed_s':time.monotonic()-t0}),flush=True)
    er=[];byday=defaultdict(list)
    for r in all_a:
        if r['action_id'] in ep:byday[r['day']].append(r)
    for day,rows in byday.items():
        before={r['action_id'] for r in rows if 'control' in r['selected']}
        # Unsupported feature/physical actions retain parent status in both arms.
        for recipe in ('A','R'):
            after=shadow_select(rows,ep,recipe)
            for r in rows:
                sign=int(r['action_id'] in after)-int(r['action_id'] in before)
                if not sign:continue
                for s in SCENARIOS:
                    label=r['labels'][s]
                    er.append({'day':day,'period':r['period'],'base_id':r['base_id'],'action_id':r['action_id'],'recipe':recipe,'scenario':s,
                       'direction':'added' if sign>0 else 'removed','label_valid':label['label_valid'],
                       'delta_reference_bps':sign*label['net_value_equity_bps'] if label['label_valid'] else None,
                       'planned_notional':r['notional_equity']*1e8,'original_net_krw':label['net_value_equity_bps']*1e4 if label['label_valid'] else None})
    result={'entry':{},'renewal':{},'exclusions':dict(exclusions),'fits':fit_count,'economic_objective_achieved':False}
    for period in ('is','oos'):
        for s in SCENARIOS:
            for recipe in ('A','R'):
                rr=[r for r in er if r['period']==period and r['scenario']==s and r['recipe']==recipe]
                result['entry'][f'{period}/{s}/{recipe}']={'all':metrics(rr,'delta_reference_bps'),'directions':{d:metrics([r for r in rr if r['direction']==d],'delta_reference_bps') for d in ('added','removed')},
                    'months':{m:metrics([r for r in rr if r['day'].startswith(m)],'delta_reference_bps') for m in sorted({r['day'][:7] for r in rr})}}
            for recipe in RENEWALS:
                rr=[dict(r,delta_reference_bps=r['optimistic_delta_reference_bps']) for r in rp if r['period']==period and r['scenario']==s and r['selected'] and r['choices'][recipe]]
                result['renewal'][f'{period}/{s}/{recipe}']={'selected':metrics(rr,'delta_reference_bps'),
                    'pending':sum(r['pending'] for r in rr),'out_of_envelope':sum(any(r['outside'][recipe].values()) for r in rr),
                    'sacrificed_positive_continuation':sum(r['remaining_net_krw']>0 for r in rr),
                    'months':{m:metrics([r for r in rr if r['day'].startswith(m)],'delta_reference_bps') for m in sorted({r['day'][:7] for r in rr})}}
    write(out/'entry_predictions.json',list(ep.values()));write(out/'entry_changes.json',er);write(out/'renewal_predictions.json',rp)
    write(out/'model_support.json',model_proofs);write(out/'assessment.json',result)
    for b in [reg['program'],reg['design'],reg['tests']]+reg['inputs']:verify(b)
    write(out/'completion.json',{'complete':True,'at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-t0,
        'registration':bound(out/'registration.json'),'outputs':[bound(p) for p in sorted(out.iterdir()) if p.name not in ('process.json','status.json','registration.json')],
        'native_cases':0,'fits':fit_count,'qualification':False})
    write(out/'status.json',{'phase':'complete','fits':fit_count,'elapsed_s':time.monotonic()-t0},True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a.output)
    except Exception as exc:
        if a.output.exists() and not (a.output/'failure.json').exists():write(a.output/'failure.json',{'error':repr(exc),'at_utc':datetime.now(timezone.utc).isoformat()})
        raise

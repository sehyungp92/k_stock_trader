"""Continue the IS fit clock across OOS, with inherited forecasts and history."""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime,timezone
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_full_oos_recovery import ROOT,FULL,START,END,read,write,bound,digest,rows,jsonl
from scripts.kalcb_recovery_refit import HISTORY,datasets,assemble,save_policy
from scripts.portfolio_frozen_inference import predictions
from scripts.kalcb_action_quantity_ranker import capacity_ok
from scripts.kalcb_quantity_calibration import corrections
from backtests.strategies.kalcb.value_calibration import fit_support_offsets
from backtests.strategies.kalcb import executed_value as ev,continuation_value as cv,paired_value as pv,action_value as av

def risk_labels(root,kind,wanted):
    """Read physical outcomes from the same authenticated quantity generations."""
    request=read(HISTORY/'kalcb_continuation_20261006/policies_v2/request.json')['arguments']
    sources=[]
    if kind=='opening':sources=[(Path(p),False) for p in request['labels']]+[(Path(p),True) for p in request['grid']]
    else:sources=[(HISTORY/'kalcb_intraday_advantage_20261007/checkpoint'/('labels_'+s),True) for s in ('baseline','combined')]
    sources += [(p,True) for p in (root/'labels').iterdir() if p.is_dir() and p.name.startswith(kind+'_')]
    grid=defaultdict(dict);proofs={}
    fields=('status','risk_available_at','physical_fill_failures','deadline_violation','unresolved')
    for folder,is_grid in sources:
        manifest=folder/('grid_manifest.json' if is_grid else 'labels_manifest.json');m=read(manifest);proofs[str(manifest)]=digest(manifest)
        for name,h in m['files' if is_grid else 'label_files'].items():
            if not name.endswith('.jsonl'):continue
            p=folder/name
            if digest(p)!=h:raise ValueError('Changed physical target evidence')
            with p.open(encoding='utf-8') as f:
                for line in f:
                    r=json.loads(line);key=r['opportunity_id']+('' if is_grid else '/q1')
                    if key not in wanted:continue
                    if key in grid[m['scenario']]:raise ValueError('Duplicate physical action target')
                    grid[m['scenario']][key]={k:r[k] for k in fields}
    if any(set(grid[s])!=wanted for s in ('baseline','combined')):raise ValueError('Incomplete paired physical targets')
    outcome={}
    for key in wanted:
        rs=[grid[s][key] for s in ('baseline','combined')]
        if any(r['status']=='source_incomplete' or not r['risk_available_at'] for r in rs):continue
        outcome[key]={'risk_available_at':max(r['risk_available_at'] for r in rs),
            'unsafe':any(r['physical_fill_failures'] or r['deadline_violation'] or r['unresolved'] for r in rs)}
    return outcome,proofs

def run(root):
    import joblib
    from threadpoolctl import threadpool_limits
    out=root/'continuous_refits';out.mkdir(exist_ok=False)
    write(out/'protocol.json',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),'program':bound(__file__),
        'recipes':['continuous_value_calibration','continuous_all_models'],'refit_sessions':20,'embargo_sessions':1,
        'clock':'Original outer_windows from May 12 history, first fit August 7. Continue the March 10 fold across April 1 until its next scheduled refit.',
        'history':'Original IS rows/labels and historical OOF calibration observations precede OOS; never restart warm-up.',
        'independence':'April-July is reused development history.'})
    data,forward,labels,sessions,old_observations,evidence=datasets(root,True)
    physical={}
    for kind in ('opening','timed'):
        physical[kind],proof=risk_labels(root,kind,{r['opportunity_id'] for r in data[kind]});evidence.update(proof)
    fit_directory=Path(read(FULL/'frozen_inference_bridge.json')['models'][0]['source']['path']).parent
    frozen={};old_proofs=[]
    for kind in ('opening','timed'):
        frozen[kind],p=predictions(fit_directory,forward[kind],kind);old_proofs+=p
    cals0=read(root/'policies/frozen/forward_lineage.json')['calibrations'];op=read(FULL/'forward_features/plan.json');tp=read(FULL/'forward_features/checkpoint_plan.json')
    source=digest(FULL/'forward_features/continuation/opportunities.jsonl');tsource=digest(FULL/'forward_features/checkpoint/opportunities.jsonl')
    windows=[(first,last) for first,last in ev.outer_windows(sessions,min_train=60,block=20) if last>=START]
    write(out/'calendar.json',{'history_start':sessions[0],'first_OOS_index':sessions.index(START),'windows':windows,
        'first_OOS_inherited_fit':windows[0][0],'next_scheduled_refit':windows[1][0]})
    cache={}
    with threadpool_limits(limits=1):
        for name,update_risk in [('continuous_value_calibration',False),('continuous_all_models',True)]:
            result={'snapshots':{},'updates':{},'admissions':[],'timed_admissions':[]};history=list(old_observations)
            folds=[p['training_proof'] for p in old_proofs];calproof=[];forecasts_saved={'opening':[],'timed':[]}
            for first,last in windows:
                dates=[d for d in sessions if max(first,START)<=d<=min(last,END)]
                forecasts={k:{r['opportunity_id']:deepcopy(frozen[k][r['opportunity_id']]) for r in forward[k] if r['trade_date'] in dates} for k in forward}
                cals=deepcopy(cals0)
                if first>=START:
                    for kind in ('opening','timed'):
                        for scenario in ('baseline','combined','failure'):
                            if scenario=='failure' and not update_risk:continue
                            key=(kind,scenario,first,last)
                            if key not in cache:
                                if scenario=='failure':model,preds,proof=av.fit_feasibility(data[kind],physical[kind],sessions,first,last)
                                else:model,preds,proof=cv.fit_forecast(data[kind],labels[kind][scenario],sessions,first,last,columns=cv.COLUMNS+(pv.CHECKPOINT_COLUMNS if kind=='timed' else ()))
                                p=out/'models'/('_'.join(key)+'.joblib');p.parent.mkdir(exist_ok=True);joblib.dump(model,p)
                                proof.update(kind=kind,scenario=scenario,model_path=str(p),model_sha256=digest(p));cache[key]=({r['opportunity_id']:v for r,v in preds},proof)
                            values,proof=cache[key];folds.append(proof)
                            for action,r in forecasts[kind].items():r['failure_probability' if scenario=='failure' else scenario+'_value_bps']=values[action];r['valid_from']=first
                    for origin,mode in [('frozen_core','offset_all'),('new_forecast','support_offset')]:
                        _,p=(corrections(history,labels['opening'],sessions,first) if mode=='offset_all' else fit_support_offsets(history,labels['opening'],sessions,first))
                        p.update(mode=mode,valid_through=last);cals[origin]=p;calproof.append(p)
                else:calproof.extend(cals.values())
                opening=[r for r in forward['opening'] if r['trade_date'] in dates];timed=[r for r in forward['timed'] if r['trade_date'] in dates]
                selected=assemble(opening,timed,forecasts['opening'],forecasts['timed'],cals,op,tp,dates,source,tsource)
                for k in ('snapshots','updates'):result[k].update(selected[k])
                for k in ('admissions','timed_admissions'):result[k].extend(selected[k])
                for r in opening:history.append({**forecasts['opening'][r['opportunity_id']],'trade_date':r['trade_date'],
                    'quantity_scale':r['features']['action_quantity_scale'],'capacity_ok':capacity_ok(r)})
                for k in forecasts_saved:forecasts_saved[k].extend(forecasts[k].values())
                print(json.dumps({'recipe':name,'fit_clock':first,'through':last,'opening':len(result['admissions']),'checkpoint':len(result['timed_admissions'])}),flush=True)
            # This helper's risk description is explicitly replaced below by
            # an additional bound continuity receipt in the policy generation.
            dest=out/'policies'/name
            save_policy(dest,root/'policies/frozen',result,folds,calproof,evidence,{'name':name,'update_physical_risk':update_risk,'continuous_fit_clock':True})
            lineage=read(dest/'lineage.json');lineage['risk_models']='Past-only scheduled expanding physical-feasibility refits' if update_risk else 'Inherited March10 feasibility models'
            write(dest/'lineage.json',lineage);m=read(dest/'policies_manifest.json');m['files']['lineage.json']=digest(dest/'lineage.json');write(dest/'policies_manifest.json',m)
            folder=out/'forecasts'/name;folder.mkdir(parents=True)
            for k,v in forecasts_saved.items():jsonl(folder/(k+'.jsonl'),v)
    write(out/'completion.json',{'complete':True,'protocol':bound(out/'protocol.json'),'calendar':bound(out/'calendar.json')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);run(p.parse_args().root)

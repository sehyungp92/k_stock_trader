"""Causal, separately controlled value-refit and calibration recovery experiments.

Historical OOF forecasts feed calibration. A one-session embargo and label
maturity apply independently to every fit. Physical-feasibility models remain
frozen to isolate economic value/calibration updates from eligibility changes.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import date,datetime,timezone
import json
from pathlib import Path
import shutil
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.kalcb_full_oos_recovery import ROOT,FULL,START,END,POLICY,read,write,rows,jsonl,bound,digest
from backtests.strategies.kalcb import executed_value as ev,continuation_value as cv,paired_value as pv
from scripts.kalcb_action_quantity_labels import scaled_row
from scripts.kalcb_action_quantity_ranker import choose_quantities,capacity_ok
from scripts.kalcb_action_value_study import admit_actions
from scripts.kalcb_checkpoint_fit import native_admit,update_snapshot
from scripts.kalcb_checkpoint_variant import meets_value_margin
from scripts.kalcb_executed_value_study import config_for
from scripts.kalcb_executed_value_replay import verify_policy,timed_policy
from scripts.kalcb_continuation_study import compact_labels
from scripts.kalcb_quantity_calibration import corrections
from backtests.strategies.kalcb.value_calibration import fit_support_offsets
from scripts.portfolio_frozen_inference import predictions,calibrated

HISTORY=Path('E:/CodexResearch/k_stock_trader')
RECIPES={
 'terminal_expanding_frozen_cal':{'values':'terminal','window':'expanding','calibration':'frozen'},
 'scheduled_expanding_frozen_cal':{'values':'scheduled','window':'expanding','calibration':'frozen'},
 'frozen_values_adaptive_cal':{'values':'frozen','window':'expanding','calibration':'adaptive'},
 'scheduled_expanding_adaptive_cal':{'values':'scheduled','window':'expanding','calibration':'adaptive'},
 'scheduled_rolling120_adaptive_cal':{'values':'scheduled','window':'rolling120','calibration':'adaptive'},
 'scheduled_decay60_adaptive_cal':{'values':'scheduled','window':'decay60','calibration':'adaptive'},
}

def ready(prepared):
    meta=read(prepared/'prepared.json')
    if digest(prepared/'opportunities.jsonl')!=meta['opportunities_sha256']:raise ValueError('Changed causal rows')
    return [scaled_row(r,q) for r in rows(prepared/'opportunities.jsonl') if r['action']['ready'] and r['guard_at_rank1'] for q in (.25,.5,1.)]

def datasets(root,need_forward):
    op=HISTORY/'kalcb_continuation_20261006/prepared_v1'
    tp=HISTORY/'kalcb_intraday_advantage_20261007/checkpoint/prepared'
    opening=ready(op);timed=ready(tp);om=read(op/'prepared.json');tm=read(tp/'prepared.json')
    if om['sessions']!=tm['sessions']:raise ValueError('Historical clocks differ')
    request=read(HISTORY/'kalcb_continuation_20261006/policies_v2/request.json')['arguments']
    base,bp=compact_labels([Path(p) for p in request['labels']],om['parent_prepared_sha256'],wanted={r['base_opportunity_id'] for r in opening})
    grid,gp=compact_labels([Path(p) for p in request['grid']],om['original_prepared_sha256'],grid=True)
    for r in opening:
        if r['features']['action_quantity_scale']==1:grid['baseline'][r['opportunity_id']]=base['baseline'][r['base_opportunity_id']]
    tg,tgp=compact_labels([HISTORY/'kalcb_intraday_advantage_20261007/checkpoint'/('labels_'+s) for s in ('baseline','combined')],digest(tp/'prepared.json'),grid=True)
    for rs,labels in ((opening,grid),(timed,tg)):
        expected={r['opportunity_id'] for r in rs}
        if any(set(labels[s])!=expected for s in ('baseline','combined')):raise ValueError('Incomplete historical quantity targets')
    evidence={**bp,**gp,**tgp}
    forward={kind:[r for r in ready(FULL/'forward_features'/stage) if START<=r['trade_date']<=END]
        for kind,stage in [('opening','continuation'),('timed','checkpoint')]}
    if need_forward:
        if not read(root/'labels/completion.json')['complete']:raise ValueError('Forward label generation incomplete')
        for kind,stage,labels in [('opening','continuation',grid),('timed','checkpoint',tg)]:
            paths=[p for p in (root/'labels').iterdir() if p.is_dir() and p.name.startswith(kind+'_')]
            extension,ep=compact_labels(paths,digest(FULL/'forward_features'/stage/'prepared.json'),grid=True)
            expected={r['opportunity_id'] for r in forward[kind]}
            if any(set(extension[s])!=expected for s in ('baseline','combined')):raise ValueError('Incomplete forward quantity targets')
            for s in ('baseline','combined'):
                if set(extension[s])&set(labels[s]):raise ValueError('Overlapping historical and forward targets')
                labels[s].update(extension[s])
            evidence.update(ep)
    sessions=om['sessions']+[d for d in read(root/'preparation.json')['sessions'] if d not in om['sessions']]
    historical_values={r['opportunity_id']:r for r in rows(HISTORY/'kalcb_continuation_20261006/policies_v2/forecasts.jsonl') if r['model']=='cv_interaction'}
    observations=[{**historical_values[r['opportunity_id']],'trade_date':r['trade_date'],
        'quantity_scale':r['features']['action_quantity_scale'],'capacity_ok':capacity_ok(r)}
        for r in opening if r['opportunity_id'] in historical_values]
    return {'opening':opening+forward['opening'],'timed':timed+forward['timed']},forward,{'opening':grid,'timed':tg},sessions,observations,evidence

def assemble(opening,timed,raw,forecast_timed,cals,opening_plan,timed_plan,sessions,source,tsource):
    """Same core-preserving admission rules as the qualified frozen policy."""
    core=calibrated(raw,cals['frozen_core']);new=calibrated(raw,cals['new_forecast'])
    cfg=config_for(opening_plan);tcfg=config_for(timed_plan)
    seed=read(opening_plan['seed'])['mutations'];q3=read(opening_plan['q3'])['mutations']
    predicate=lambda row,rank:ev.eligible(row,seed,q3,rank=rank)
    by_day=defaultdict(list);timed_days=defaultdict(list)
    for r in opening:by_day[r['trade_date']].append(r)
    for r in timed:timed_days[r['trade_date']].append(r)
    base={};updates={};admissions=[];tadmissions=[]
    scores=lambda values:{s:{k:r[s+'_value_bps'] for k,r in values.items()} for s in ('baseline','combined')}
    cs,ns,ts=scores(core),scores(new),scores(forecast_timed)
    for day in sessions:
        items=by_day[day];feasible=[r for r in items if raw[r['opportunity_id']]['failure_probability']<.001]
        original=[r for r in feasible if min(raw[r['opportunity_id']][s+'_value_bps'] for s in ('baseline','combined'))>0]
        def choose(items,values):return choose_quantities(items,values['baseline'],values['combined'],robust=True)[0]
        def admit(choices):return admit_actions(list(choices.values()),predicate,cfg,source,budget=8)[0]
        original_selected=admit(choose(original,cs));original_locked={r['base_opportunity_id']:(r,v) for r,v in original_selected}
        expanded=choose(feasible,cs);expanded.update(original_locked)
        selected_core=admit(expanded);locked={r['base_opportunity_id']:(r,v) for r,v in selected_core}
        eligible=[r for r in feasible if r['base_opportunity_id'] in locked or r['features']['action_quantity_scale']<=.5]
        supported=choose(eligible,ns);supported.update(locked);selected=admit(supported)
        fit=max((raw[r['opportunity_id']]['valid_from'] for r in items),default=START)
        snap=ev.snapshot_for(day,selected,source=source,forecast={'policy':'causal_recovery','valid_from':fit})
        base[date.fromisoformat(day)]=snap
        for row,value in selected:
            admissions.append({'opportunity_id':row['base_opportunity_id'],'action_id':row['opportunity_id'],
                'trade_date':day,'reason':'admitted','score':value,'quantity_scale':row['features']['action_quantity_scale'],
                'valid_from':fit,'forecast_origin':'frozen_core' if row['base_opportunity_id'] in locked else 'new_forecast'})
        candidates=[r for r in timed_days[day] if r['features']['action_quantity_scale']<=.5
            and forecast_timed[r['opportunity_id']]['failure_probability']<.001
            and meets_value_margin(r,forecast_timed[r['opportunity_id']],48)]
        extra=native_admit(snap,choose(candidates,ts),tcfg,tsource)
        if extra:
            updates[date.fromisoformat(day)]=[update_snapshot(snap,extra,tsource)]
            for row,value in extra:tadmissions.append({'trade_date':day,'symbol':row['symbol'],
                'available_at':row['feature_available_at'],'score':value,'action_id':row['opportunity_id'],
                'quantity_scale':row['features']['action_quantity_scale'],'valid_from':forecast_timed[row['opportunity_id']]['valid_from']})
    return {'snapshots':base,'updates':updates,'admissions':admissions,'timed_admissions':tadmissions}

def save_policy(out,parent,result,folds,calibrations,evidence,recipe):
    out.mkdir(parents=True,exist_ok=False)
    write(out/(POLICY+'.snapshots.json'),[s.to_json_dict() for s in result['snapshots'].values()])
    jsonl(out/(POLICY+'.admission.jsonl'),result['admissions'])
    jsonl(out/'timed.snapshots.jsonl',(s.to_json_dict() for a in result['updates'].values() for s in a))
    jsonl(out/'timed.admission.jsonl',result['timed_admissions'])
    write(out/'chronology.json',{'folds':folds,'calibration':calibrations,'scope':'Past-only prequential recovery; reused development observations.'})
    write(out/'lineage.json',{'labels':evidence,'recipe':recipe,'risk_models':'Frozen March 10 physical-feasibility probabilities; not loss probabilities.',
        'selection':'Opening locked core and capped support; checkpoint quantity <=0.5 and 48bps hurdle.'})
    manifest={**read(parent/'policies_manifest.json'),'evaluation_from':START,'evaluation_through':END,
        'files':{p.name:digest(p) for p in out.iterdir()},'recovery_recipe':recipe}
    write(out/'policies_manifest.json',manifest);write(out/'completion.json',{'complete':True})
    m,s=verify_policy(out,POLICY);timed_policy(out,m,s)

def run(root,names):
    import joblib
    from threadpoolctl import threadpool_limits
    out=root/'refits';out.mkdir(exist_ok=True)
    if not (out/'protocol.json').exists():write(out/'protocol.json',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'recipes':RECIPES,'program':bound(__file__),'refit_sessions':20,'embargo_sessions':1,
        'frozen_physical_risk':True,'IS_scope':'Deployment-policy experiments extend the selected IS state after April 1. Historical IS is a retained reference, not a new validation of alternative training schedules.',
        'independence':'All April-July tests are reused development; no renewed independent OOS.',
        'target_alignment':'Original native management labels kept unchanged to isolate staleness. Selected-management target alignment is a separate experiment.'})
    elif read(out/'protocol.json')['program']['sha256']!=digest(__file__):raise ValueError('Changed registered fitting program')
    need=any(RECIPES[n]['values']=='scheduled' or RECIPES[n]['calibration']=='adaptive' for n in names)
    data,forward,labels,sessions,observations,evidence=datasets(root,need)
    parent=root/'policies/frozen';fit_directory=Path(read(FULL/'frozen_inference_bridge.json')['models'][0]['source']['path']).parent
    frozen={};proofs=[]
    for kind in ('opening','timed'):
        frozen[kind],p=predictions(fit_directory,forward[kind],kind);proofs+=p
    initial=read(parent/'forward_lineage.json')['calibrations']
    opening_plan=read(FULL/'forward_features/plan.json');timed_plan=read(FULL/'forward_features/checkpoint_plan.json')
    ds=read(root/'preparation.json')['sessions'];source=digest(FULL/'forward_features/continuation/opportunities.jsonl');tsource=digest(FULL/'forward_features/checkpoint/opportunities.jsonl')
    bridge=assemble(forward['opening'],forward['timed'],frozen['opening'],frozen['timed'],initial,opening_plan,timed_plan,ds,source,tsource)
    m,original=verify_policy(parent,POLICY);updates=timed_policy(parent,m,original)
    def choice(s):return {c.symbol:(c.metadata.get('action_quantity_scale',1),c.selection_score) for c in s.candidates}
    for day,snap in original.items():
        for a,b in [(snap,bridge['snapshots'][day]),(updates.get(day,[snap])[-1],bridge['updates'].get(day,[bridge['snapshots'][day]])[-1])]:
            if choice(a)!=choice(b):raise ValueError('Layered selector no-drift bridge failure')
    write(out/'selection_bridge.json',{'passed':True,'day_phases':138,'parent':bound(parent/'policies_manifest.json')})
    fitcache={}
    with threadpool_limits(limits=1):
        for name in names:
            dest=out/'policies'/name
            if (dest/'completion.json').exists():continue
            recipe=RECIPES[name];result={'snapshots':{},'updates':{},'admissions':[],'timed_admissions':[]}
            history=list(observations);folds=[p['training_proof'] for p in proofs];calproof=[];raw_saved=[];timed_saved=[]
            terminal=None
            for i in range(0,len(ds),20):
                dates=ds[i:i+20];first,through=dates[0],dates[-1]
                forecasts={k:{r['opportunity_id']:deepcopy(frozen[k][r['opportunity_id']]) for r in forward[k] if first<=r['trade_date']<=through} for k in forward}
                if recipe['values']!='frozen':
                    fit_first=START if recipe['values']=='terminal' else first
                    fit_through=END if recipe['values']=='terminal' else through
                    for kind in ('opening','timed'):
                        key=(kind,recipe['window'],fit_first,fit_through)
                        if key not in fitcache:
                            values={};fp=[]
                            for scenario in ('baseline','combined'):
                                columns=cv.COLUMNS+(pv.CHECKPOINT_COLUMNS if kind=='timed' else ())
                                model,preds,proof=cv.fit_forecast(data[kind],labels[kind][scenario],sessions,fit_first,fit_through,window=recipe['window'],columns=columns)
                                path=out/'models'/('_'.join(key)+'_'+scenario+'.joblib');path.parent.mkdir(exist_ok=True)
                                joblib.dump(model,path);proof.update(scenario=scenario,model_path=str(path),model_sha256=digest(path),kind=kind)
                                fp.append(proof)
                                for r,v in preds:values.setdefault(r['opportunity_id'],{})[scenario+'_value_bps']=v
                            fitcache[key]=(values,fp)
                        values,fp=fitcache[key];folds.extend(p for p in fp if p not in folds)
                        for action,r in forecasts[kind].items():r.update(values[action]);r['valid_from']=fit_first
                cals=deepcopy(initial)
                if recipe['calibration']=='adaptive':
                    for origin,mode in [('frozen_core','offset_all'),('new_forecast','support_offset')]:
                        _,p=(corrections(history,labels['opening'],sessions,first) if mode=='offset_all' else fit_support_offsets(history,labels['opening'],sessions,first))
                        p.update(mode=mode,valid_through=through);cals[origin]=p;calproof.append(p)
                else:
                    for p in cals.values():
                        if p not in calproof:calproof.append(p)
                selected=assemble([r for r in forward['opening'] if first<=r['trade_date']<=through],
                    [r for r in forward['timed'] if first<=r['trade_date']<=through],forecasts['opening'],forecasts['timed'],cals,opening_plan,timed_plan,dates,source,tsource)
                for k in ('snapshots','updates'):result[k].update(selected[k])
                for k in ('admissions','timed_admissions'):result[k].extend(selected[k])
                for r in forward['opening']:
                    if first<=r['trade_date']<=through:history.append({**forecasts['opening'][r['opportunity_id']],
                        'trade_date':r['trade_date'],'quantity_scale':r['features']['action_quantity_scale'],'capacity_ok':capacity_ok(r)})
                raw_saved.extend(forecasts['opening'].values());timed_saved.extend(forecasts['timed'].values())
                print(json.dumps({'recipe':name,'fold':first,'through':through,'opening':len(result['admissions']),'checkpoint':len(result['timed_admissions'])}),flush=True)
            save_policy(dest,parent,result,folds,calproof,evidence,recipe)
            # Bind forecasts after policy generation without mutating its receipt.
            diagnostics=out/'forecasts'/name;diagnostics.mkdir(parents=True)
            jsonl(diagnostics/'opening.jsonl',raw_saved);jsonl(diagnostics/'timed.jsonl',timed_saved)
            write(diagnostics/'completion.json',{'complete':True,'policy':bound(dest/'policies_manifest.json'),
                'files':{p.name:digest(p) for p in diagnostics.iterdir() if p.is_file()}})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--recipes',nargs='+',choices=RECIPES,default=list(RECIPES));a=p.parse_args();run(a.root,a.recipes)
if __name__=='__main__':main()

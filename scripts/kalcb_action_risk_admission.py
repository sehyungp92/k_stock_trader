"""Add an explicit, past-only execution-feasibility forecast to quantity choice."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import action_value as av, executed_value as ev
from scripts.kalcb_action_quantity_labels import scaled_row
from scripts.kalcb_action_quantity_ranker import load_grid, choose_quantities
from scripts.kalcb_action_value_study import admit_actions
from scripts.kalcb_executed_value_study import config_for,emit,load_plan

MAXIMUM_FAILURE_PROBABILITY=.01


def fit(plan,prepared,original_labels,grid_paths,parent,output,maximum_probability=MAXIMUM_FAILURE_PROBABILITY,
        model_names=('q_ridge','q_hgb','q_hgb_rich')):
    import joblib
    from threadpoolctl import threadpool_limits
    from scripts.kalcb_executed_value_ranker import load_labels
    from scripts.kalcb_executed_value_replay import verify_policy
    provenance=json.loads((prepared/'prepared.json').read_text())
    if ev.digest(prepared/'opportunities.jsonl')!=provenance['opportunities_sha256']:
        raise ValueError('Changed causal actions')
    parent_manifest,_=verify_policy(parent,'q_ridge_robust')
    base_labels,label_proofs=load_labels(original_labels,provenance['parent_prepared_sha256'])
    grid,grid_proofs=load_grid(grid_paths,ev.digest(prepared/'prepared.json'))
    rows=[]
    with (prepared/'opportunities.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line,object_hook=lambda d:{sys.intern(k):v for k,v in d.items()})
            if row['action']['ready'] and row['guard_at_rank1']:
                for scale in (.25,.5,1.0):
                    action=scaled_row(row,scale);rows.append(action)
                    if scale==1:
                        grid['baseline'][action['opportunity_id']]=base_labels[row['opportunity_id']]
    expected={r['opportunity_id'] for r in rows}
    if any(set(grid[s])!=expected for s in ('baseline','combined')):
        raise ValueError('Incomplete quantity outcomes')
    outcomes={}
    for key in expected:
        observations=[grid[s][key] for s in ('baseline','combined')]
        if any(r['status']=='source_incomplete' or not r.get('risk_available_at') for r in observations):
            continue
        outcomes[key]={'risk_available_at':max(r['risk_available_at'] for r in observations),
            'unsafe':any(r['physical_fill_failures'] or r['deadline_violation'] or r['unresolved'] for r in observations)}
    forecasts=defaultdict(dict)
    for r in ev.read_jsonl(parent/'quantity_forecasts.jsonl'):
        forecasts[r['model']][r['opportunity_id']]=r
    models=tuple(model_names)
    suffix='_risk' if maximum_probability==.01 else '_risk_strict'
    names=[m+suffix for m in models]
    snapshots={n:[] for n in names};decisions={n:[] for n in names}
    proofs=json.loads((parent/'chronology.json').read_text())['folds']
    cfg=config_for(plan)
    if not cfg.entry_action_quantity_scale_enabled:raise ValueError('Quantity metadata must be enabled')
    seed=json.loads(Path(plan['seed']).read_text())['mutations']
    q3=json.loads(Path(plan['q3']).read_text())['mutations']
    predicate=lambda row,rank:ev.eligible(row,seed,q3,rank=rank)
    windows=ev.outer_windows(provenance['sessions'],min_train=60,block=20)
    with (output/'failure_forecasts.jsonl').open('x',encoding='utf-8') as stream,threadpool_limits(limits=1):
        for fold,(first,through) in enumerate(windows):
            model,predictions,proof=av.fit_feasibility(rows,outcomes,provenance['sessions'],first,through)
            path=output/f'feasibility_model_{fold}.joblib';joblib.dump(model,path)
            proof.update(model_path=path.name,model_sha256=ev.digest(path));proofs.append(proof)
            by_day=defaultdict(list);risk={}
            for row,p in predictions:
                risk[row['opportunity_id']]=p;by_day[row['trade_date']].append(row)
                emit(stream,{'opportunity_id':row['opportunity_id'],'failure_probability':p,'valid_from':first})
            for day in [d for d in provenance['sessions'] if first<=d<=through]:
                feasible=[r for r in by_day[day] if risk[r['opportunity_id']]<maximum_probability]
                for model_name in models:
                    name=model_name+suffix
                    baseline={r['opportunity_id']:forecasts[model_name][r['opportunity_id']]['baseline_value_bps'] for r in feasible}
                    stress={r['opportunity_id']:forecasts[model_name][r['opportunity_id']]['combined_value_bps'] for r in feasible}
                    for r in feasible:
                        if forecasts[model_name][r['opportunity_id']]['valid_from']!=first:
                            raise ValueError('Value and risk forecast folds disagree')
                    choices,_=choose_quantities(feasible,baseline,stress,robust=True)
                    selected,reasons=admit_actions(list(choices.values()),predicate,cfg,provenance['opportunities_sha256'])
                    forecast={'policy':name,'valid_from':first,'maximum_failure_probability':maximum_probability}
                    snapshots[name].append(ev.snapshot_for(day,selected,source=provenance['opportunities_sha256'],forecast=forecast).to_json_dict())
                    for row,value in selected:
                        decisions[name].append({'opportunity_id':row['base_opportunity_id'],'action_id':row['opportunity_id'],
                            'trade_date':day,'score':value,'reason':'admitted','quantity_scale':row['features']['action_quantity_scale'],
                            'failure_probability':risk[row['opportunity_id']],**forecast})
            print(json.dumps({'risk_fold':fold,'from':first,'admissions':{n:len(decisions[n]) for n in names}}),flush=True)
    for name in names:
        ev.write_json(output/(name+'.snapshots.json'),snapshots[name])
        with (output/(name+'.admission.jsonl')).open('x',encoding='utf-8') as stream:
            for row in decisions[name]:emit(stream,row)
    ev.write_json(output/'chronology.json',{'folds':proofs,'scope':'past-only value and failure forecasts; reused development observations'})
    ev.write_json(output/'policies_manifest.json',{'version':'kalcb-quantity-feasibility-v1','policies':names,
        'evaluation_from':windows[0][0],'evaluation_through':windows[-1][1],
        'files':{p.name:ev.digest(p) for p in output.iterdir() if p.is_file()},
        'contract':{'budget':8,'maximum_failure_probability':maximum_probability,
                    'quantity_flag_required':True,'production_promotion':False},
        'parent_policy_manifest_sha256':ev.digest(parent/'policies_manifest.json'),
        'prepared_sha256':ev.digest(prepared/'prepared.json'),'label_manifests':label_proofs,'grid_manifests':grid_proofs})


def main():
    from backtests.auto.shared.cache_keys import source_generation_hash
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','prepared','parent','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--labels',type=Path,action='append',required=True)
    p.add_argument('--grid',type=Path,action='append',required=True)
    p.add_argument('--maximum-failure-probability',type=float,choices=(.01,.001),default=.01)
    p.add_argument('--models',nargs='+',choices=('q_ridge','q_hgb','q_hgb_rich'),default=('q_ridge','q_hgb','q_hgb_rich'))
    a=p.parse_args();plan=load_plan(a.plan);source=source_generation_hash()
    a.output.mkdir(parents=True,exist_ok=False)
    ev.write_json(a.output/'request.json',{'arguments':vars(a),'source_generation_hash':source,'plan_sha256':ev.digest(a.plan)})
    if len(set(a.models))!=len(a.models):raise ValueError('Duplicate value model')
    fit(plan,a.prepared,a.labels,a.grid,a.parent,a.output,a.maximum_failure_probability,a.models)
    if source_generation_hash()!=source:raise RuntimeError('Source changed during risk fitting')
    ev.write_json(a.output/'completion.json',{'complete':True,'source_generation_hash':source})


if __name__=='__main__':main()

"""Past-only quantity-dependent value, stress and capacity-aware admission."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import action_value as av, executed_value as ev
from scripts.kalcb_action_quantity_labels import scaled_row
from scripts.kalcb_action_value_study import admit_actions
from scripts.kalcb_executed_value_study import config_for,emit,load_plan


def load_grid(paths, prepared_hash):
    result=defaultdict(dict);proofs={}
    for path in paths:
        manifest=json.loads((path/'grid_manifest.json').read_text())
        complete=json.loads((path/'completion.json').read_text())
        request=json.loads((path/'request.json').read_text())
        if not complete['complete'] or complete['source_generation_hash']!=request['source_generation_hash']:
            raise ValueError('Incomplete grid generation')
        if manifest['prepared_sha256']!=prepared_hash or ev.digest(path/'request.json')!=manifest['request_sha256']:
            raise ValueError('Grid input identity mismatch')
        for name,want in manifest['files'].items():
            if ev.digest(path/name)!=want:raise ValueError('Changed grid evidence')
            if name.endswith('.jsonl'):
                for row in ev.read_jsonl(path/name):
                    scope=result[manifest['scenario']]
                    if row['opportunity_id'] in scope:raise ValueError('Duplicate grid action')
                    scope[row['opportunity_id']]=row
        proofs[str(path/'grid_manifest.json')]=ev.digest(path/'grid_manifest.json')
    return result,proofs


def capacity_ok(row):
    """Conservative forecast from completed opening volume, not a fill guarantee."""
    from kis_core.trading_calendar import get_trading_calendar
    cutoff=datetime.fromisoformat(row['feature_available_at'])
    closing=get_trading_calendar().session_hours(cutoff.date())[1]
    deadline=datetime.combine(cutoff.date(),closing,cutoff.tzinfo)-timedelta(minutes=10)
    remaining=max(0,int((deadline-cutoff).total_seconds()//300))
    volumes=sorted(b['volume'] for b in row['opening_bars'])
    low_volume=volumes[min(1,len(volumes)-1)]
    # A zero integer-share capacity must remain zero. The finite denominator in
    # the model feature is an encoding convenience, not permission to assume one
    # executable share per bar in a feasibility decision.
    capacity=math.floor(low_volume*.005)*.5
    return capacity>0 and 0<row['action']['quantity']<=remaining*capacity


def choose_quantities(items, baseline, stress, *, robust=False):
    choices={};dispositions={}
    for row in items:
        key=row['opportunity_id'];base=row['base_opportunity_id'];value=baseline[key]
        if value<=0:
            dispositions[key]='nonpositive_expected_value';continue
        if robust and (stress[key]<=0 or not capacity_ok(row)):
            dispositions[key]='stress_or_capacity';continue
        previous=choices.get(base)
        # Stable ties prefer less resource use. Cash (value zero) remains available.
        if previous is None or (value,-row['features']['action_quantity_scale'])>(previous[1],-previous[0]['features']['action_quantity_scale']):
            choices[base]=(row,value)
        dispositions[key]='feasible_action'
    return choices,dispositions


def fit(plan, prepared, original_labels, grid_paths, output, controls=None):
    import joblib
    from threadpoolctl import threadpool_limits
    from scripts.kalcb_executed_value_ranker import load_labels
    provenance=json.loads((prepared/'prepared.json').read_text())
    if ev.digest(prepared/'opportunities.jsonl')!=provenance['opportunities_sha256']:
        raise ValueError('Changed action dataset')
    base_labels,label_proofs=load_labels(original_labels,provenance['parent_prepared_sha256'])
    grid,grid_proofs=load_grid(grid_paths,ev.digest(prepared/'prepared.json'))
    rows=[];base_rows={}
    with (prepared/'opportunities.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line,object_hook=lambda d:{sys.intern(k):v for k,v in d.items()})
            if row['action']['ready'] and row['guard_at_rank1']:
                base_rows[row['opportunity_id']]=row
                for scale in (.25,.5,1.0):
                    action=scaled_row(row,scale);rows.append(action)
                    if scale==1:
                        grid['baseline'][action['opportunity_id']]={**base_labels[row['opportunity_id']],
                            'opportunity_id':action['opportunity_id'],'base_opportunity_id':row['opportunity_id'],
                            'quantity_scale':scale}
    expected={r['opportunity_id'] for r in rows}
    for scenario in ('baseline','combined'):
        if set(grid[scenario])!=expected:raise ValueError(f'Incomplete {scenario} quantity grid')
    cfg=replace(config_for(plan),entry_action_quantity_scale_enabled=True)
    seed=json.loads(Path(plan['seed']).read_text())['mutations']
    q3=json.loads(Path(plan['q3']).read_text())['mutations']
    predicate=lambda row,rank:ev.eligible(row,seed,q3,rank=rank)
    specs=(av.ModelSpec('q_ridge','scaled_ridge'),av.ModelSpec('q_hgb','hgb'),av.ModelSpec('q_hgb_rich','hgb',rich=True))
    names=[s.name+suffix for s in specs for suffix in ('_expected','_robust')]
    policies={n:[] for n in names};decisions={n:[] for n in names};proofs=[]
    windows=ev.outer_windows(provenance['sessions'],min_train=60,block=20)
    with (output/'quantity_forecasts.jsonl').open('x',encoding='utf-8') as forecasts,threadpool_limits(limits=1):
        for fold,(first,through) in enumerate(windows):
            for spec in specs:
                scores={}
                for scenario in ('baseline','combined'):
                    model,predictions,proof=av.fit_forecast(rows,grid[scenario],provenance['sessions'],first,through,spec)
                    path=output/f'model_{spec.name}_{scenario}_{fold}.joblib';joblib.dump(model,path)
                    proof.update(scenario=scenario,model_path=path.name,model_sha256=ev.digest(path));proofs.append(proof)
                    scores[scenario]={r['opportunity_id']:v for r,v in predictions}
                by_day=defaultdict(list)
                for row,_ in predictions:
                    by_day[row['trade_date']].append(row)
                    emit(forecasts,{'opportunity_id':row['opportunity_id'],'model':spec.name,
                        'quantity_scale':row['features']['action_quantity_scale'],'valid_from':first,
                        'baseline_value_bps':scores['baseline'][row['opportunity_id']],
                        'combined_value_bps':scores['combined'][row['opportunity_id']],
                        'capacity_screen':capacity_ok(row)})
                for day in [d for d in provenance['sessions'] if first<=d<=through]:
                    for robust in (False,True):
                        name=spec.name+('_robust' if robust else '_expected')
                        choices,action_reasons=choose_quantities(by_day[day],scores['baseline'],scores['combined'],robust=robust)
                        # One action per symbol; rank/context preview runs again at
                        # the chosen quantity. The ordinary broker allocates cash.
                        selected,reasons=admit_actions(list(choices.values()),predicate,cfg,provenance['opportunities_sha256'])
                        forecast={'policy':name,'valid_from':first,'quantity_dependent':True,'robust':robust}
                        policies[name].append(ev.snapshot_for(day,selected,source=provenance['opportunities_sha256'],forecast=forecast).to_json_dict())
                        selected_map={r['base_opportunity_id']:(r,value) for r,value in selected}
                        for base in sorted({r['base_opportunity_id'] for r in by_day[day]}):
                            chosen=selected_map.get(base) or choices.get(base)
                            row,value=chosen if chosen else (scaled_row(base_rows[base],1),0.0)
                            decisions[name].append({'opportunity_id':base,'trade_date':day,'score':value,
                                'reason':'admitted' if base in selected_map else reasons.get(row['opportunity_id'],'no_positive_feasible_quantity'),
                                'quantity_scale':row['features']['action_quantity_scale'],
                                'action_id':row['opportunity_id'],**forecast})
            print(json.dumps({'quantity_fold':fold,'from':first,'through':through,
                'admissions':{n:sum(len(s['candidates']) for s in ss) for n,ss in policies.items()}}),flush=True)
    for name in names:
        ev.write_json(output/(name+'.snapshots.json'),policies[name])
        with (output/(name+'.admission.jsonl')).open('x',encoding='utf-8') as stream:
            for row in decisions[name]:emit(stream,row)
    control_hash = None
    if controls:
        from scripts.kalcb_executed_value_replay import verify_policy
        _, originals=verify_policy(controls,'ready_hgb')
        original_decisions=ev.read_jsonl(controls/'ready_hgb.admission.jsonl')
        control_hash=ev.digest(controls/'policies_manifest.json')
        proofs.extend(json.loads((controls/'chronology.json').read_text())['folds'])
        for name,scale in (('q_fixed_quarter',.25),('q_fixed_half',.5)):
            names.append(name);converted=[]
            for day,snapshot in sorted(originals.items()):
                items=[]
                for candidate in snapshot.to_json_dict()['candidates']:
                    candidate['metadata']={**candidate['metadata'],'action_quantity_scale':scale}
                    items.append(({'candidate':candidate,'feature_available_at':candidate['metadata'].get('feature_available_at')},candidate['selection_score']))
                converted.append(ev.snapshot_for(day.isoformat(),items,source=provenance['opportunities_sha256'],
                    forecast={'policy':name,'quantity_scale':scale,'selection_control':'ready_hgb'}).to_json_dict())
            ev.write_json(output/(name+'.snapshots.json'),converted)
            with (output/(name+'.admission.jsonl')).open('x',encoding='utf-8') as stream:
                for row in original_decisions:emit(stream,{**row,'policy':name,'quantity_scale':scale})
    ev.write_json(output/'chronology.json',{'folds':proofs,'scope':'past-only action and scenario fits on reused development observations'})
    files={p.name:ev.digest(p) for p in output.iterdir() if p.is_file()}
    ev.write_json(output/'policies_manifest.json',{'version':'kalcb-quantity-value-v1','files':files,'policies':names,
        'evaluation_from':windows[0][0],'evaluation_through':windows[-1][1],
        'contract':{'budget':8,'scales':[0,.25,.5,1],'objective':'maximum forecast baseline net value; robust adds positive combined value and causal capacity screen',
            'quantity_flag_required':True,'production_promotion':False},
        'prepared_sha256':ev.digest(prepared/'prepared.json'),'label_manifests':label_proofs,'grid_manifests':grid_proofs,
        'fixed_quantity_control_manifest_sha256':control_hash})


def main():
    from backtests.auto.shared.cache_keys import source_generation_hash
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','prepared','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--labels',type=Path,action='append',required=True)
    p.add_argument('--grid',type=Path,action='append',required=True)
    p.add_argument('--controls',type=Path)
    a=p.parse_args();plan=load_plan(a.plan);source=source_generation_hash()
    if not config_for(plan).entry_action_quantity_scale_enabled:
        raise ValueError('Quantity replay plan must explicitly enable reduction metadata')
    a.output.mkdir(parents=True,exist_ok=False)
    ev.write_json(a.output/'request.json',{'arguments':vars(a),'source_generation_hash':source,'plan_sha256':ev.digest(a.plan)})
    fit(plan,a.prepared,a.labels,a.grid,a.output,a.controls)
    if source_generation_hash()!=source:raise RuntimeError('Source changed during quantity fitting')
    ev.write_json(a.output/'completion.json',{'complete':True,'source_generation_hash':source})


if __name__=='__main__':main()

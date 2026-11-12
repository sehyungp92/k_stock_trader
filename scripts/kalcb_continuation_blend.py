"""Fixed, untuned averaging of authenticated expanding and rolling forecasts."""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import groupby
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev
from scripts.kalcb_executed_value_study import load_plan,config_for,emit


def forecast_values(directory,name):
    from scripts.kalcb_executed_value_replay import verify_policy
    manifest,_=verify_policy(directory,name)
    values={}
    with (directory/'forecasts.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['model']==name:
                if row['opportunity_id'] in values:raise ValueError('Duplicate forecast')
                values[row['opportunity_id']]=row
    return manifest,values


def combine(left,right):
    if left['opportunity_id']!=right['opportunity_id'] or left['valid_from']!=right['valid_from']:
        raise ValueError('Forecast identity or clock differs')
    if left['failure_probability']!=right['failure_probability']:
        raise ValueError('Execution risk models differ')
    return {**left,'model':'cv_blend',
        'baseline_value_bps':(left['baseline_value_bps']+right['baseline_value_bps'])/2,
        'combined_value_bps':(left['combined_value_bps']+right['combined_value_bps'])/2}


def run(plan,prepared,left,right,output):
    from scripts.kalcb_action_quantity_labels import scaled_row
    from scripts.kalcb_action_quantity_ranker import choose_quantities
    from scripts.kalcb_action_value_study import admit_actions
    lm,lv=forecast_values(left,'cv_interaction')
    rm,rv=forecast_values(right,'cv_interaction_rolling120')
    for key in ('prepared_sha256','evaluation_from','evaluation_through','risk_parent_manifest_sha256'):
        if lm[key]!=rm[key]:raise ValueError('Different component dependencies')
    if set(lv)!=set(rv):raise ValueError('Different forecast populations')
    meta=json.loads((prepared/'prepared.json').read_text())
    if ev.digest(prepared/'prepared.json')!=lm['prepared_sha256'] or ev.digest(prepared/'opportunities.jsonl')!=meta['opportunities_sha256']:
        raise ValueError('Changed candidate population')
    values={k:combine(lv[k],rv[k]) for k in lv};del lv,rv
    cfg=config_for(plan)
    seed=json.loads(Path(plan['seed']).read_text())['mutations'];q3=json.loads(Path(plan['q3']).read_text())['mutations']
    predicate=lambda r,rank:ev.eligible(r,seed,q3,rank=rank)
    scores={s:{k:v[s+'_value_bps'] for k,v in values.items()} for s in ('baseline','combined')}
    snapshots=[];admitted=[]
    with (prepared/'opportunities.jsonl').open() as stream:
        rows=(json.loads(line) for line in stream)
        for day,group in groupby(rows,key=lambda r:r['trade_date']):
            if not lm['evaluation_from']<=day<=lm['evaluation_through']:continue
            actions=[]
            for row in group:
                for scale in (.25,.5,1):
                    action=scaled_row(row,scale)
                    if values[action['opportunity_id']]['failure_probability']<.001:actions.append(action)
            choices,_=choose_quantities(actions,scores['baseline'],scores['combined'],robust=True)
            selected,_=admit_actions(list(choices.values()),predicate,cfg,meta['opportunities_sha256'])
            provenance={'policy':'cv_blend','fixed_component_weights':[.5,.5],'maximum_failure_probability':.001}
            snapshots.append(ev.snapshot_for(day,selected,source=meta['opportunities_sha256'],forecast=provenance).to_json_dict())
            for row,value in selected:
                admitted.append({'opportunity_id':row['base_opportunity_id'],'action_id':row['opportunity_id'],
                    'trade_date':day,'score':value,'reason':'admitted','quantity_scale':row['features']['action_quantity_scale'],
                    'valid_from':values[row['opportunity_id']]['valid_from'],**provenance})
    expected=[d for d in meta['sessions'] if lm['evaluation_from']<=d<=lm['evaluation_through']]
    if [s['trade_date'] for s in snapshots]!=expected:raise ValueError('Missing or unsorted decision sessions')
    ev.write_json(output/'cv_blend.snapshots.json',snapshots)
    for name,rows in (('cv_blend.admission.jsonl',admitted),('forecasts.jsonl',values.values())):
        with (output/name).open('x',encoding='utf-8') as f:
            for row in rows:emit(f,row)
    proofs=[]
    for i,parent in enumerate((left,right)):
        for proof in json.loads((parent/'chronology.json').read_text())['folds']:
            if proof['name']=='interaction':
                source=parent/proof['model_path']
                if ev.digest(source)!=proof['model_sha256']:raise ValueError('Changed component model')
                target=output/source.name
                os.link(source,target)
                proofs.append(proof)
            elif i==0 and proof['name']=='execution_failure_probability':proofs.append(proof)
    ev.write_json(output/'chronology.json',{'folds':proofs,'scope':'Fixed equal weights; no fitted ensemble coefficients; reused development data'})
    ev.write_json(output/'policies_manifest.json',{'version':'kalcb-continuation-blend-v1','policies':['cv_blend'],
        'evaluation_from':lm['evaluation_from'],'evaluation_through':lm['evaluation_through'],
        'prepared_sha256':lm['prepared_sha256'],'risk_parent_manifest_sha256':lm['risk_parent_manifest_sha256'],
        'component_manifests':{str(p/'policies_manifest.json'):ev.digest(p/'policies_manifest.json') for p in (left,right)},
        'contract':{'budget':8,'maximum_failure_probability':.001,'quantity_flag_required':True,
            'fixed_component_weights':[.5,.5],'production_promotion':False},
        'files':{p.name:ev.digest(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'admissions':len(admitted),'policy':'cv_blend'}),flush=True)


def main():
    from backtests.auto.shared.cache_keys import source_generation_hash
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','prepared','left','right','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();plan=load_plan(a.plan);source=source_generation_hash();a.output.mkdir(parents=True,exist_ok=False)
    ev.write_json(a.output/'request.json',{'arguments':vars(a),'source_generation_hash':source,'plan_sha256':ev.digest(a.plan),
        'study_role':'One fixed blend, declared after standalone window comparison began; development selection only'})
    run(plan,a.prepared,a.left,a.right,a.output)
    if source_generation_hash()!=source:raise RuntimeError('Changed executing generation')
    ev.write_json(a.output/'completion.json',{'complete':True,'source_generation_hash':source})


if __name__=='__main__':main()

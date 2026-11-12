"""Reselect quantity actions from authenticated past-only value forecasts.

Two declared comparisons isolate robust payoff ranking and the contribution of
large quantity actions. No outcome labels are accessed by this transformation.
"""
import argparse,json,os,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev
from scripts.kalcb_executed_value_study import load_plan,config_for,emit
from scripts.kalcb_continuation_blend import forecast_values

def choose(items,values,*,objective,cap):
    from scripts.kalcb_action_quantity_ranker import capacity_ok
    chosen={}
    for row in items:
        key=row['opportunity_id'];v=values[key];q=row['features']['action_quantity_scale']
        if q>cap or v['failure_probability']>=.001 or v['baseline_value_bps']<=0 or v['combined_value_bps']<=0 or not capacity_ok(row):continue
        value=v[objective+'_value_bps'];base=row['base_opportunity_id'];old=chosen.get(base)
        if old is None or (value,-q)>(old[1],-old[0]['features']['action_quantity_scale']):chosen[base]=(row,value)
    return list(chosen.values())

def run(plan,prepared,parent,parent_name,output):
    from scripts.kalcb_action_quantity_labels import scaled_row
    from scripts.kalcb_action_value_study import admit_actions
    manifest,values=forecast_values(parent,parent_name)
    meta=json.loads((prepared/'prepared.json').read_text())
    if ev.digest(prepared/'prepared.json')!=manifest['prepared_sha256'] or ev.digest(prepared/'opportunities.jsonl')!=meta['opportunities_sha256']:raise ValueError('Prepared identity differs')
    rows=defaultdict(list)
    for row in ev.read_jsonl(prepared/'opportunities.jsonl'):
        if manifest['evaluation_from']<=row['trade_date']<=manifest['evaluation_through']:
            for q in (.25,.5,1):rows[row['trade_date']].append(scaled_row(row,q))
    if {r['opportunity_id'] for rs in rows.values() for r in rs}!=set(values):raise ValueError('Forecast population differs')
    seed=json.loads(Path(plan['seed']).read_text())['mutations'];q3=json.loads(Path(plan['q3']).read_text())['mutations']
    cfg=config_for(plan);predicate=lambda r,rank:ev.eligible(r,seed,q3,rank=rank)
    policies=[]
    for suffix,objective,cap in [('stress_value','combined',1),('half_cap','baseline',.5)]:
        name=parent_name+'_'+suffix;policies.append(name);snapshots=[];admissions=[]
        for day in (d for d in meta['sessions'] if manifest['evaluation_from']<=d<=manifest['evaluation_through']):
            selected,_=admit_actions(choose(rows[day],values,objective=objective,cap=cap),predicate,cfg,meta['opportunities_sha256'])
            provenance={'policy':name,'objective':objective,'quantity_cap':cap,'maximum_failure_probability':.001}
            snapshots.append(ev.snapshot_for(day,selected,source=meta['opportunities_sha256'],forecast=provenance).to_json_dict())
            for r,score in selected:
                admissions.append({'opportunity_id':r['base_opportunity_id'],'action_id':r['opportunity_id'],
                    'trade_date':day,'score':score,'reason':'admitted','quantity_scale':r['features']['action_quantity_scale'],
                    'valid_from':values[r['opportunity_id']]['valid_from'],**provenance})
        ev.write_json(output/(name+'.snapshots.json'),snapshots)
        with (output/(name+'.admission.jsonl')).open('x',encoding='utf-8') as out:
            for r in admissions:emit(out,r)
        print(json.dumps({'policy':name,'admissions':len(admissions)}),flush=True)
    for path in parent.iterdir():
        if path.name not in ('request.json','completion.json','policies_manifest.json'):os.link(path,output/path.name)
    ev.write_json(output/'policies_manifest.json',{**manifest,'policies':policies,
        'version':'kalcb-risk-admission-v1','parent_manifest_sha256':ev.digest(parent/'policies_manifest.json'),
        'files':{p.name:ev.digest(p) for p in output.iterdir() if p.is_file()}})

def main():
    from backtests.auto.shared.cache_keys import source_generation_hash
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','prepared','parent','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--parent-name',required=True);a=p.parse_args()
    plan=load_plan(a.plan);source=source_generation_hash();a.output.mkdir(parents=True,exist_ok=False)
    ev.write_json(a.output/'request.json',{'arguments':vars(a),'source_generation_hash':source})
    run(plan,a.prepared,a.parent,a.parent_name,a.output)
    if source_generation_hash()!=source:raise ValueError('Source changed')
    ev.write_json(a.output/'completion.json',{'complete':True,'source_generation_hash':source})

if __name__=='__main__':main()

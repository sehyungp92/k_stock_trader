"""Preserve calibrated core quantities/slots while admitting causal new actions.

Consumes frozen opening-time forecasts and earlier admission decisions only.
No outcomes or trade ledgers are inputs. Broker allocation remains unchanged.
"""
import argparse,json,os,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev
from scripts.kalcb_executed_value_replay import verify_policy
from scripts.kalcb_executed_value_study import load_plan,config_for,emit
from scripts.kalcb_action_quantity_labels import scaled_row
from scripts.kalcb_action_quantity_ranker import choose_quantities
from scripts.kalcb_action_value_study import admit_actions


def merge_core_choices(choices,core,actions):
    """Lock only decisions known at that day's opening; other names stay eligible."""
    result=dict(choices)
    for decision in core:
        action=actions[decision['action_id']]
        if action['base_opportunity_id']!=decision['opportunity_id']:
            raise ValueError('Core action identity mismatch')
        result[decision['opportunity_id']]=(action,decision['score'])
    return result


def cap_new_actions(rows, core_keys, maximum_quantity):
    if maximum_quantity not in (.25,.5,1.):raise ValueError('Undeclared new-action quantity cap')
    return [r for r in rows if r['base_opportunity_id'] in core_keys
            or r['features']['action_quantity_scale']<=maximum_quantity]


def selected_forecast_origin(row, core, forecasts):
    decision=core.get(row['base_opportunity_id'])
    if decision is not None and decision['action_id']==row['opportunity_id']:
        return decision['valid_from'],'frozen_core'
    return forecasts[row['opportunity_id']]['valid_from'],'new_forecast'


def run(plan,prepared,parent,parent_name,core,core_name,output,mode,maximum_new_quantity=1.):
    manifest,_=verify_policy(parent,parent_name);core_manifest,_=verify_policy(core,core_name)
    meta=json.loads((prepared/'prepared.json').read_text())
    if ev.digest(prepared/'prepared.json')!=manifest['prepared_sha256'] or ev.digest(prepared/'opportunities.jsonl')!=meta['opportunities_sha256']:
        raise ValueError('Changed prepared data')
    for field in ('prepared_sha256','evaluation_from','evaluation_through'):
        if core_manifest[field]!=manifest[field]:raise ValueError('Different core population')
    forecasts={r['opportunity_id']:r for r in ev.read_jsonl(parent/'calibrated_forecasts.jsonl')}
    windows={r['valid_from']:r['valid_through'] for r in json.loads((parent/'chronology.json').read_text())['calibration']}
    core_days=defaultdict(list)
    for d in ev.read_jsonl(core/(core_name+'.admission.jsonl')):
        if d['reason']=='admitted':core_days[d['trade_date']].append(d)
    rows=defaultdict(list)
    for row in ev.read_jsonl(prepared/'opportunities.jsonl'):
        if manifest['evaluation_from']<=row['trade_date']<=manifest['evaluation_through']:
            rows[row['trade_date']].extend(scaled_row(row,q) for q in (.25,.5,1))
    seed=json.loads(Path(plan['seed']).read_text())['mutations'];q3=json.loads(Path(plan['q3']).read_text())['mutations']
    predicate=lambda r,rank:ev.eligible(r,seed,q3,rank=rank)
    cfg=config_for(plan);name=core_name+'_'+mode;snapshots=[];admissions=[]
    for day in (d for d in meta['sessions'] if manifest['evaluation_from']<=d<=manifest['evaluation_through']):
        actions={r['opportunity_id']:r for r in rows[day]}
        core_for_day={d['opportunity_id']:d for d in core_days[day]}
        eligible_rows=cap_new_actions(rows[day],{d['opportunity_id'] for d in core_days[day]},maximum_new_quantity)
        feasible=[r for r in eligible_rows if r['opportunity_id'] in forecasts]
        for r in feasible:
            f=forecasts[r['opportunity_id']]
            if not f['valid_from']<=day<=windows[f['valid_from']] or f['failure_probability']>=.001:
                raise ValueError('Invalid opening-time forecast')
        scores={s:{k:f[s+'_value_bps'] for k,f in forecasts.items() if k in actions} for s in ('baseline','combined')}
        choices,_=choose_quantities(feasible,scores['baseline'],scores['combined'],robust=True)
        merged=merge_core_choices(choices,core_days[day],actions)
        priorities=[d['action_id'] for d in core_days[day]] if mode=='core_first' else ()
        selected,_=admit_actions(list(merged.values()),predicate,cfg,meta['opportunities_sha256'],
            budget=manifest['contract']['budget'],priority_keys=priorities)
        provenance={'policy':name,'allocation_mode':mode,'core_manifest_sha256':ev.digest(core/'policies_manifest.json'),
            'parent_manifest_sha256':ev.digest(parent/'policies_manifest.json'),'broker_priority_changed':False,
            'maximum_new_quantity':maximum_new_quantity}
        snapshots.append(ev.snapshot_for(day,selected,source=meta['opportunities_sha256'],forecast=provenance).to_json_dict())
        for r,value in selected:
            valid_from,origin=selected_forecast_origin(r,core_for_day,forecasts)
            admissions.append({'opportunity_id':r['base_opportunity_id'],'action_id':r['opportunity_id'],
                'trade_date':day,'reason':'admitted','score':value,'quantity_scale':r['features']['action_quantity_scale'],
                'valid_from':valid_from,'forecast_origin':origin,**provenance})
    for path in parent.iterdir():
        if path.name not in ('request.json','completion.json','policies_manifest.json','chronology.json'):os.link(path,output/path.name)
    # Bind the complete incumbent lineage, including fitted models, residual
    # calibration and generation receipts. Prefixes prevent model collisions.
    for path in core.iterdir():
        if path.is_file():os.link(path,output/('core_'+path.name))
    chronology=json.loads((parent/'chronology.json').read_text())
    core_chronology=json.loads((core/'chronology.json').read_text())
    chronology['folds'] += [{**f,'model_path':'core_'+f['model_path'],'lineage':'frozen_core'}
        if 'model_path' in f else {**f,'lineage':'frozen_core'} for f in core_chronology['folds']]
    chronology.setdefault('calibration',[]).extend({**f,'lineage':'frozen_core'} for f in core_chronology.get('calibration',[]))
    ev.write_json(output/'chronology.json',chronology)
    ev.write_json(output/(name+'.snapshots.json'),snapshots)
    with (output/(name+'.admission.jsonl')).open('x',encoding='utf-8') as out:
        for r in admissions:emit(out,r)
    ev.write_json(output/'policies_manifest.json',{**manifest,'version':'kalcb-core-preserving-v1','policies':[name],
        'allocation_mode':mode,'parent_manifest_sha256':ev.digest(parent/'policies_manifest.json'),
        'core_manifest_sha256':ev.digest(core/'policies_manifest.json'),
        'files':{p.name:ev.digest(p) for p in output.iterdir() if p.is_file()}})


def main():
    from backtests.auto.shared.cache_keys import source_generation_hash
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','prepared','parent','core','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--parent-name',required=True);p.add_argument('--core-name',required=True)
    p.add_argument('--mode',choices=('quantity_locked','core_first'),required=True)
    p.add_argument('--maximum-new-quantity',type=float,choices=(.25,.5,1.),default=1.)
    a=p.parse_args();plan=load_plan(a.plan);source=source_generation_hash();a.output.mkdir(parents=True,exist_ok=False)
    ev.write_json(a.output/'request.json',{'arguments':vars(a),'source_generation_hash':source})
    run(plan,a.prepared,a.parent,a.parent_name,a.core,a.core_name,a.output,a.mode,a.maximum_new_quantity)
    if source_generation_hash()!=source:raise ValueError('Source changed')
    ev.write_json(a.output/'completion.json',{'complete':True,'source_generation_hash':source})
    verify_policy(a.output,a.core_name+'_'+a.mode)


if __name__=='__main__':main()

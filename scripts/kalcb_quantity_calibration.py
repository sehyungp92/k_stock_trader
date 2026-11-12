"""Past-only shrinkage calibration within the model's feasible quantity cohorts."""
import argparse,json,os,sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev
from scripts.kalcb_executed_value_study import load_plan,config_for,emit
from scripts.kalcb_continuation_blend import forecast_values
from scripts.kalcb_continuation_study import compact_labels

def corrections(observations,labels,sessions,first,lookback=60,prior_count=100,minimum_count=20):
    """Estimate intercept error from earlier OOF feasible actions, never fitted scores."""
    allowed=set(sessions[max(0,sessions.index(first)-1-lookback):max(0,sessions.index(first)-1)])
    cutoff=datetime.fromisoformat(first+'T00:00:00+09:00')
    errors=defaultdict(list);latest=None
    for r in observations:
        if r['trade_date'] not in allowed:continue
        if r['baseline_value_bps']<=0 or r['combined_value_bps']<=0 or r['failure_probability']>=.001 or not r['capacity_ok']:continue
        for scenario in ('baseline','combined'):
            label=labels[scenario][r['opportunity_id']]
            if not label['label_valid'] or datetime.fromisoformat(label['label_available_at'])>=cutoff:continue
            errors[(scenario,r['quantity_scale'])].append(label['net_value_equity_bps']-r[scenario+'_value_bps'])
            latest=max(latest or label['label_available_at'],label['label_available_at'])
    detail={};offsets={}
    for scenario in ('baseline','combined'):
        for q in (.25,.5,1):
            e=errors[(scenario,q)];offset=sum(e)/(len(e)+prior_count) if len(e)>=minimum_count else 0.0
            offsets[(scenario,q)]=offset
            detail[f'{scenario}/q{q:g}']={'observations':len(e),'offset_bps':offset}
    return offsets,{'valid_from':first,'latest_label_available_at':latest,
        'observed_through':max(allowed) if allowed else None,'cohorts':detail,
        'lookback_sessions':lookback,'prior_count':prior_count,'minimum_count':minimum_count,'embargo_sessions':1}

def run(plan,prepared,parent,parent_name,labels_paths,grids,output,mode='positive_offset',complete_grid=False):
    from scripts.kalcb_action_quantity_labels import scaled_row
    from scripts.kalcb_action_quantity_ranker import choose_quantities,capacity_ok
    from scripts.kalcb_action_value_study import admit_actions
    manifest,values=forecast_values(parent,parent_name)
    meta=json.loads((prepared/'prepared.json').read_text())
    if ev.digest(prepared/'prepared.json')!=manifest['prepared_sha256'] or ev.digest(prepared/'opportunities.jsonl')!=meta['opportunities_sha256']:raise ValueError('Changed prepared data')
    rows=defaultdict(list);observations=[]
    for row in ev.read_jsonl(prepared/'opportunities.jsonl'):
        if manifest['evaluation_from']<=row['trade_date']<=manifest['evaluation_through']:
            for q in (.25,.5,1):
                r=scaled_row(row,q);rows[row['trade_date']].append(r)
                observations.append({**values[r['opportunity_id']],'trade_date':r['trade_date'],
                    'quantity_scale':q,'capacity_ok':capacity_ok(r)})
    if complete_grid:
        from scripts.kalcb_management_refit import changed_grid
        if labels_paths:raise ValueError('Old management labels cannot supplement a complete grid')
        grid,gp=changed_grid(grids,prepared,plan);lp={}
        expected={r['opportunity_id'] for rs in rows.values() for r in rs}
        if any(not expected.issubset(grid[s]) for s in ('baseline','combined')):raise ValueError('Incomplete calibration grid')
    else:
        base,lp=compact_labels(labels_paths,meta['parent_prepared_sha256'],wanted={r['base_opportunity_id'] for rs in rows.values() for r in rs})
        grid,gp=compact_labels(grids,meta['original_prepared_sha256'],grid=True)
        for rs in rows.values():
            for r in rs:
                if r['features']['action_quantity_scale']==1:grid['baseline'][r['opportunity_id']]=base['baseline'][r['base_opportunity_id']]
    seed=json.loads(Path(plan['seed']).read_text())['mutations'];q3=json.loads(Path(plan['q3']).read_text())['mutations']
    cfg=config_for(plan);predicate=lambda r,rank:ev.eligible(r,seed,q3,rank=rank)
    if mode not in ('positive_offset','offset_all','affine_all','support_offset'):raise ValueError('Unknown calibration mode')
    name=parent_name+'_calibrated'+('' if mode=='positive_offset' else '_'+mode);snapshots=[];admissions=[];proofs=[]
    with (output/'calibrated_forecasts.jsonl').open('x',encoding='utf-8') as out:
        for first,through in ev.outer_windows(meta['sessions'],min_train=60,block=20):
            if mode=='affine_all':
                from backtests.strategies.kalcb.value_calibration import fit_affine,affine_value
                affine,proof=fit_affine(observations,grid,meta['sessions'],first)
            elif mode=='support_offset':
                from backtests.strategies.kalcb.value_calibration import fit_support_offsets,score_support
                offsets,proof=fit_support_offsets(observations,grid,meta['sessions'],first)
            else:offsets,proof=corrections(observations,grid,meta['sessions'],first)
            proof['valid_through']=through;proof['mode']=mode;proofs.append(proof)
            for day in (d for d in meta['sessions'] if first<=d<=through):
                scores={'baseline':{},'combined':{}};eligible=[]
                for r in rows[day]:
                    key=r['opportunity_id'];q=r['features']['action_quantity_scale'];v=values[key]
                    if v['valid_from']!=first:raise ValueError('Unexpected forecast clock')
                    if v['failure_probability']>=.001:continue
                    if mode=='positive_offset' and (v['baseline_value_bps']<=0 or v['combined_value_bps']<=0):continue
                    eligible.append(r)
                    for s in scores:
                        scores[s][key]=(affine_value(affine[s],v[s+'_value_bps'],q) if mode=='affine_all'
                            else v[s+'_value_bps']+offsets[(s,q,score_support(v))] if mode=='support_offset'
                            else v[s+'_value_bps']+offsets[(s,q)])
                    emit(out,{**v,'model':name,'baseline_value_bps':scores['baseline'][key],
                        'combined_value_bps':scores['combined'][key],'quantity_scale':q})
                choices,_=choose_quantities(eligible,scores['baseline'],scores['combined'],robust=True)
                selected,_=admit_actions(list(choices.values()),predicate,cfg,meta['opportunities_sha256'])
                provenance={'policy':name,'valid_from':first,'quantity_calibration':'past OOF residuals, 60 sessions, 100-observation prior','calibration_mode':mode}
                snapshots.append(ev.snapshot_for(day,selected,source=meta['opportunities_sha256'],forecast=provenance).to_json_dict())
                for r,value in selected:
                    admissions.append({'opportunity_id':r['base_opportunity_id'],'action_id':r['opportunity_id'],
                        'trade_date':day,'reason':'admitted','score':value,'quantity_scale':r['features']['action_quantity_scale'],**provenance})
            print(json.dumps({'fold':first,'admissions':len(admissions),'calibration':proof['cohorts']}),flush=True)
    for path in parent.iterdir():
        if path.name not in ('request.json','completion.json','policies_manifest.json','chronology.json'):os.link(path,output/path.name)
    chronology=json.loads((parent/'chronology.json').read_text());chronology['calibration']=proofs
    ev.write_json(output/'chronology.json',chronology)
    ev.write_json(output/(name+'.snapshots.json'),snapshots)
    with (output/(name+'.admission.jsonl')).open('x',encoding='utf-8') as out:
        for r in admissions:emit(out,r)
    ev.write_json(output/'policies_manifest.json',{**manifest,'version':'kalcb-quantity-calibration-v1','policies':[name],
        'parent_manifest_sha256':ev.digest(parent/'policies_manifest.json'),'calibration_label_manifests':{**lp,**gp},
        'files':{p.name:ev.digest(p) for p in output.iterdir() if p.is_file()}})

def main():
    from backtests.auto.shared.cache_keys import source_generation_hash
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','prepared','parent','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--parent-name',required=True)
    p.add_argument('--mode',choices=('positive_offset','offset_all','affine_all','support_offset'),default='positive_offset')
    p.add_argument('--labels',type=Path,action='append',default=[]);p.add_argument('--grid',type=Path,action='append',required=True)
    p.add_argument('--complete-grid',action='store_true')
    a=p.parse_args();plan=load_plan(a.plan);source=source_generation_hash();a.output.mkdir(parents=True,exist_ok=False)
    ev.write_json(a.output/'request.json',{'arguments':vars(a),'source_generation_hash':source})
    if not a.complete_grid and not a.labels:raise ValueError('Legacy calibration requires baseline labels')
    run(plan,a.prepared,a.parent,a.parent_name,a.labels,a.grid,a.output,a.mode,a.complete_grid)
    if source_generation_hash()!=source:raise ValueError('Source changed')
    ev.write_json(a.output/'completion.json',{'complete':True,'source_generation_hash':source})

if __name__=='__main__':main()

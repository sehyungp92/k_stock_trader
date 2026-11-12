"""Refit value and feasibility to a complete native changed-management grid.

All three quantities and both scenarios must describe the same management plan.
No historical q=1 labels from a different exit policy are silently substituted.
"""
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev,continuation_value as cv,action_value as av
from scripts.kalcb_executed_value_study import config_for,load_plan,emit
from scripts.kalcb_action_quantity_labels import scaled_row
from scripts.kalcb_action_quantity_ranker import load_grid,choose_quantities
from scripts.kalcb_action_value_study import admit_actions


def changed_grid(paths,prepared,plan,expected=None):
    for path in paths:
        request=json.loads((path/'request.json').read_text())
        generating=load_plan(Path(request['arguments']['plan']))
        for key in ('config','database','seed','q3'):
            if generating['input_sha256'][generating[key]]!=plan['input_sha256'][plan[key]]:
                raise ValueError('Changed-management grid differs from fitted '+key)
    grid,proofs=load_grid(paths,ev.digest(prepared/'prepared.json'))
    if expected is not None and any(set(grid[s])!=expected for s in ('baseline','combined')):
        raise ValueError('Complete changed-management action grid required')
    return grid,proofs


def failure_outcomes(grid):
    if set(grid['baseline'])!=set(grid['combined']):raise ValueError('Unpaired action outcomes')
    out={}
    for key in grid['baseline']:
        rows=[grid[s][key] for s in ('baseline','combined')]
        if any(r['status']=='source_incomplete' or not r.get('risk_available_at') for r in rows):continue
        out[key]={'risk_available_at':max(r['risk_available_at'] for r in rows),
            'unsafe':any(r['physical_fill_failures'] or r['deadline_violation'] or r['unresolved'] for r in rows)}
    return out


def fit(plan,prepared,paths,output,kind='interaction'):
    import joblib
    from threadpoolctl import threadpool_limits
    meta=json.loads((prepared/'prepared.json').read_text())
    if ev.digest(prepared/'opportunities.jsonl')!=meta['opportunities_sha256']:raise ValueError('Changed prepared actions')
    rows=[scaled_row(r,q) for r in ev.read_jsonl(prepared/'opportunities.jsonl') for q in (.25,.5,1.)]
    grid,gp=changed_grid(paths,prepared,plan,{r['opportunity_id'] for r in rows})
    outcomes=failure_outcomes(grid)
    cfg=config_for(plan)
    if not cfg.entry_action_quantity_scale_enabled:raise ValueError('Quantity metadata disabled')
    seed=json.loads(Path(plan['seed']).read_text())['mutations'];q3=json.loads(Path(plan['q3']).read_text())['mutations']
    predicate=lambda r,rank:ev.eligible(r,seed,q3,rank=rank)
    windows=ev.outer_windows(meta['sessions'],min_train=60,block=20)
    name='cv_'+kind+'_management_refit';snapshots=[];decisions=[];proofs=[]
    with (output/'forecasts.jsonl').open('x',encoding='utf-8') as stream,threadpool_limits(limits=1):
        for fold,(first,through) in enumerate(windows):
            risk_model,risk_predictions,risk_proof=av.fit_feasibility(rows,outcomes,meta['sessions'],first,through)
            p=output/f'risk_{fold}.joblib';joblib.dump(risk_model,p)
            risk_proof.update(model_path=p.name,model_sha256=ev.digest(p));proofs.append(risk_proof)
            risk={r['opportunity_id']:v for r,v in risk_predictions};scores={}
            for scenario in ('baseline','combined'):
                if kind=='paired':
                    from backtests.strategies.kalcb.paired_value import fit_forecast
                    model,predictions,proof=fit_forecast(rows,grid[scenario],meta['sessions'],first,through)
                else:model,predictions,proof=cv.fit_forecast(rows,grid[scenario],meta['sessions'],first,through,'interaction','expanding')
                p=output/f'value_{scenario}_{fold}.joblib';joblib.dump(model,p)
                proof.update(scenario=scenario,model_path=p.name,model_sha256=ev.digest(p));proofs.append(proof)
                scores[scenario]={r['opportunity_id']:v for r,v in predictions}
            by_day=defaultdict(list)
            for r,_ in predictions:
                key=r['opportunity_id']
                emit(stream,{'opportunity_id':key,'model':name,'valid_from':first,'baseline_value_bps':scores['baseline'][key],
                    'combined_value_bps':scores['combined'][key],'failure_probability':risk[key]})
                if risk[key]<.001:by_day[r['trade_date']].append(r)
            for day in (d for d in meta['sessions'] if first<=d<=through):
                choices,_=choose_quantities(by_day[day],scores['baseline'],scores['combined'],robust=True)
                selected,_=admit_actions(list(choices.values()),predicate,cfg,meta['opportunities_sha256'])
                proof={'policy':name,'valid_from':first,'maximum_failure_probability':.001,'management_labels_refitted':True}
                snapshots.append(ev.snapshot_for(day,selected,source=meta['opportunities_sha256'],forecast=proof).to_json_dict())
                decisions.extend({'opportunity_id':r['base_opportunity_id'],'action_id':r['opportunity_id'],'trade_date':day,
                    'score':v,'reason':'admitted','quantity_scale':r['features']['action_quantity_scale'],**proof} for r,v in selected)
            print(json.dumps({'fold':first,'admissions':len(decisions)}),flush=True)
    ev.write_json(output/(name+'.snapshots.json'),snapshots)
    with (output/(name+'.admission.jsonl')).open('x',encoding='utf-8') as f:
        for d in decisions:emit(f,d)
    ev.write_json(output/'chronology.json',{'folds':proofs,'scope':'Past-only expanding value and risk fits on changed-management outcomes; reused development'})
    ev.write_json(output/'policies_manifest.json',{'version':'kalcb-management-refit-v1','policies':[name],
        'evaluation_from':windows[0][0],'evaluation_through':windows[-1][1],
        'files':{p.name:ev.digest(p) for p in output.iterdir() if p.is_file()},
        'contract':{'budget':8,'maximum_failure_probability':.001,'quantity_flag_required':True,'production_promotion':False},
        'prepared_sha256':ev.digest(prepared/'prepared.json'),'label_manifests':{},'grid_manifests':gp,
        'management_config_sha256':plan['input_sha256'][plan['config']]})


def main():
    from backtests.auto.shared.cache_keys import source_generation_hash
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('plan','prepared','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--grid',type=Path,action='append',required=True)
    p.add_argument('--kind',choices=('interaction','paired'),default='interaction');a=p.parse_args()
    plan=load_plan(a.plan);source=source_generation_hash();a.output.mkdir(parents=True,exist_ok=False)
    ev.write_json(a.output/'request.json',{'arguments':vars(a),'source_generation_hash':source,'plan_sha256':ev.digest(a.plan)})
    fit(plan,a.prepared,a.grid,a.output,a.kind)
    if source_generation_hash()!=source:raise ValueError('Changed refit source')
    ev.write_json(a.output/'completion.json',{'complete':True,'source_generation_hash':source})


if __name__=='__main__':main()

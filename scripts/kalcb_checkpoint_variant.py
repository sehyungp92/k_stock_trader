"""Rebuild a declared quantity cap using existing authenticated causal forecasts."""
import argparse,json,shutil,sys
from collections import defaultdict
from datetime import date
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev
from scripts.kalcb_executed_value_replay import verify_policy,timed_policy
from scripts.kalcb_executed_value_study import load_plan,config_for,emit
from scripts.kalcb_action_quantity_labels import scaled_row
from scripts.kalcb_action_quantity_ranker import choose_quantities
from scripts.kalcb_checkpoint_fit import native_admit,update_snapshot


def meets_value_margin(row,prediction,roundtrip_bps):
    """Same economic hurdle across sizes, expressed in reference-equity bps."""
    import math
    notional=row['features']['action_notional_equity']
    if not math.isfinite(notional) or notional<=0: return False
    hurdle=notional*roundtrip_bps
    return min(prediction['baseline_value_bps'],prediction['combined_value_bps'])>hurdle


def build(plan,prepared,parent,parent_name,output,cap,cost_margin=False):
    manifest,opening=verify_policy(parent,parent_name);timed_policy(parent,manifest,opening)
    meta=json.loads((prepared/'prepared.json').read_text())
    if ev.digest(prepared/'prepared.json')!=manifest['prepared_sha256'] or ev.digest(prepared/'opportunities.jsonl')!=meta['opportunities_sha256']:
        raise ValueError('Changed checkpoint population')
    values={r['opportunity_id']:r for r in ev.read_jsonl(parent/'forecasts.jsonl') if r['model']=='interaction'}
    if not values:raise ValueError('Missing interaction forecasts')
    by_day=defaultdict(list);cfg=config_for(plan)
    hurdle_bps=2*cfg.commission_bps+cfg.tax_bps_on_sell+2*ev.SCENARIOS['combined'][1] if cost_margin else 0.
    for r in ev.read_jsonl(prepared/'opportunities.jsonl'):
        if not r['action']['ready'] or not r['guard_at_rank1']:continue
        if not manifest['evaluation_from']<=r['trade_date']<=manifest['evaluation_through']:continue
        for q in (.25,.5,1.):
            action=scaled_row(r,q);v=values[action['opportunity_id']]
            if v['trade_date']!=r['trade_date'] or v['valid_from']>r['trade_date']:raise ValueError('Forecast clock mismatch')
            if q<=cap and v['failure_probability']<.001 and (not cost_margin or meets_value_margin(action,v,hurdle_bps)):
                by_day[r['trade_date']].append(action)
    scores={s:{k:v[s+'_value_bps'] for k,v in values.items()} for s in ('baseline','combined')}
    updates=[];decisions=[]
    for day,base in opening.items():
        choices,_=choose_quantities(by_day[str(day)],scores['baseline'],scores['combined'],robust=True)
        selected=native_admit(base,choices,cfg,meta['opportunities_sha256'])
        if selected:
            updates.append(update_snapshot(base,selected,meta['opportunities_sha256']).to_json_dict())
            for r,v in selected:decisions.append({'trade_date':str(day),'symbol':r['symbol'],'available_at':r['feature_available_at'],
                'score':v,'action_id':r['opportunity_id'],'quantity_scale':r['features']['action_quantity_scale'],
                'valid_from':values[r['opportunity_id']]['valid_from']})
    for filename in manifest['files']:
        if filename not in ('request.json','timed.snapshots.jsonl','timed.admission.jsonl'):
            shutil.copy2(parent/filename,output/filename)
    name='interaction_cost_margin' if cost_margin else 'interaction_quarter'
    shutil.copy2(parent/(parent_name+'.snapshots.json'),output/(name+'.snapshots.json'))
    shutil.copy2(parent/(parent_name+'.admission.jsonl'),output/(name+'.admission.jsonl'))
    for filename,rows in [('timed.snapshots.jsonl',updates),('timed.admission.jsonl',decisions)]:
        with (output/filename).open('x') as f:
            for r in rows:emit(f,r)
    ev.write_json(output/'policies_manifest.json',{**manifest,'policies':[name],
        'parent_policy_manifest_sha256':ev.digest(parent/'policies_manifest.json'),
        'files':{p.name:ev.digest(p) for p in output.iterdir() if p.is_file()},
        'contract':{**manifest['contract'],'quantity_cap':cap,'additional_value_hurdle_bps_on_notional':hurdle_bps,
            'selection':'Same forecasts; optional one stressed roundtrip net-profit hurdle; native rank/context preview'}})
    ev.write_json(output/'completion.json',{'complete':True})
    m,s=verify_policy(output,name);timed_policy(output,m,s)
    print(json.dumps({'policy':name,'admissions':len(decisions)}),flush=True)


def main():
    from backtests.auto.shared.cache_keys import source_generation_hash
    p=argparse.ArgumentParser()
    for k in ('plan','prepared','parent','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--parent-name',default='interaction_half');p.add_argument('--cap',type=float,choices=(.25,.5),default=.25)
    p.add_argument('--cost-margin',action='store_true')
    a=p.parse_args();plan=load_plan(a.plan);source=source_generation_hash();a.output.mkdir(parents=True,exist_ok=False)
    ev.write_json(a.output/'request.json',{'arguments':vars(a),'source_generation_hash':source})
    build(plan,a.prepared,a.parent,a.parent_name,a.output,a.cap,a.cost_margin)
    if source_generation_hash()!=source:raise RuntimeError('Source changed')
    # completion is not a manifest member, so the final source receipt can replace it.
    ev.write_json(a.output/'completion.json',{'complete':True,'source_generation_hash':source})


if __name__=='__main__':main()

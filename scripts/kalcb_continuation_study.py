"""Past-only continuation state and payoff study on frozen executed actions."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev,continuation_value as cv
from scripts.kalcb_executed_value_study import config_for,connect,emit,load_plan


def compact_labels(paths, prepared_hash, *, grid=False, wanted=None):
    """Authenticate all evidence, retaining only fields consumed by value fits."""
    result=defaultdict(dict);proofs={}
    fields=('label_valid','label_available_at','net_value_equity_bps')
    for path in paths:
        manifest_path=path/('grid_manifest.json' if grid else 'labels_manifest.json')
        manifest=json.loads(manifest_path.read_text())
        complete=json.loads((path/'completion.json').read_text())
        request=json.loads((path/'request.json').read_text())
        if not complete.get('complete') or complete['source_generation_hash']!=request['source_generation_hash']:
            raise ValueError('Incomplete label generation')
        if manifest['prepared_sha256']!=prepared_hash or ev.digest(path/'request.json')!=manifest['request_sha256']:
            raise ValueError('Label input identity mismatch')
        scenario=manifest['scenario']
        if scenario not in ('baseline','combined') or (not grid and scenario!='baseline'):
            raise ValueError('Undeclared label scenario')
        for name,want in manifest['files' if grid else 'label_files'].items():
            if ev.digest(path/name)!=want:raise ValueError('Changed label evidence')
            if not name.endswith('.jsonl'):continue
            with (path/name).open(encoding='utf-8') as stream:
                for line in stream:
                    row=json.loads(line);key=row['opportunity_id']
                    if wanted is not None and key not in wanted:continue
                    if key in result[scenario]:raise ValueError('Duplicate fitted action label')
                    result[scenario][key]={k:row[k] for k in fields}
        proofs[str(manifest_path)]=ev.digest(manifest_path)
    return result,proofs


def prepare(plan,parent,output):
    meta=json.loads((parent/'prepared.json').read_text())
    if ev.digest(parent/'opportunities.jsonl')!=meta['opportunities_sha256']:raise ValueError('Changed parent actions')
    conn=connect(plan);sessions=meta['sessions'];previous={d:sessions[i-1] if i else None for i,d in enumerate(sessions)}
    current=None;prior={};count=0
    with (parent/'opportunities.jsonl').open() as stream,(output/'opportunities.jsonl').open('x',encoding='utf-8') as out:
        for line in stream:
            r=json.loads(line)
            if not r['action']['ready'] or not r['guard_at_rank1']:continue
            day=r['trade_date']
            if day!=current:
                current=day;prior=defaultdict(list)
                if previous[day]:
                    for s,ts,o,h,l,c,v in conn.execute('SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day=? ORDER BY symbol,ts',(previous[day],)):
                        prior[s].append(dict(timestamp=ts,open=o,high=h,low=l,close=c,volume=v))
            r['features']={**r['features'],**cv.opening_path(r['opening_bars'],r['feature_available_at']),
                **cv.prior_path(prior.get(r['symbol'],[]),day)}
            emit(out,r);count+=1
    conn.close()
    ev.write_json(output/'prepared.json',{**meta,'version':'kalcb-continuation-v1',
        'original_prepared_sha256':ev.digest(parent/'prepared.json'),
        'original_opportunities_sha256':meta['opportunities_sha256'],
        'opportunities_sha256':ev.digest(output/'opportunities.jsonl'),
        'continuation_rows':count,'continuation_features':cv.COLUMNS,
        'continuation_contract':'Completed opening prefix and previous-session bars only; original opening-ready native-guard population'})
    print(json.dumps({'prepared_rows':count}),flush=True)


def check_risk_clock(prediction,windows,first,day):
    start=prediction['valid_from']
    if start not in windows or start>first or not start<=day<=windows[start]:
        raise ValueError('Risk clock mismatch')


def fit(plan,prepared,labels,grids,risk_parent,output,models,window,refit_sessions=20,feature_set='continuation'):
    import joblib
    from threadpoolctl import threadpool_limits
    from scripts.kalcb_action_quantity_labels import scaled_row
    from scripts.kalcb_action_quantity_ranker import choose_quantities
    from scripts.kalcb_action_value_study import admit_actions
    from scripts.kalcb_executed_value_replay import verify_policy
    meta=json.loads((prepared/'prepared.json').read_text())
    columns=cv.COLUMNS
    interactions=cv.INTERACTIONS
    if feature_set in ('campaign','campaign_interactions'):
        from backtests.strategies.kalcb.campaign_value import COLUMNS as campaign_columns,INTERACTIONS as campaign_interactions
        if tuple(meta.get('campaign_columns',()))!=campaign_columns:raise ValueError('Missing campaign schema')
        columns=columns+campaign_columns
        if feature_set=='campaign_interactions':interactions=interactions+campaign_interactions
    if feature_set=='leadership':
        from backtests.strategies.kalcb.leadership_value import COLUMNS as leader_columns,INTERACTIONS as leader_interactions
        if tuple(meta.get('leadership_columns',()))!=leader_columns:raise ValueError('Missing leadership schema')
        columns=columns+leader_columns;interactions=interactions+leader_interactions
    if ev.digest(prepared/'opportunities.jsonl')!=meta['opportunities_sha256']:raise ValueError('Changed continuation data')
    rows=[]
    with (prepared/'opportunities.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line,object_hook=lambda d:{sys.intern(k):v for k,v in d.items()})
            r['features']={k:r['features'].get(k) for k in columns}
            for q in (.25,.5,1):rows.append(scaled_row(r,q))
    base,proofs_labels=compact_labels(labels,meta['parent_prepared_sha256'],
        wanted={r['base_opportunity_id'] for r in rows})
    grid,proofs_grid=compact_labels(grids,meta['original_prepared_sha256'],grid=True)
    for r in rows:
        if r['features']['action_quantity_scale']==1:
            grid['baseline'][r['opportunity_id']]=base['baseline'][r['base_opportunity_id']]
    expected={r['opportunity_id'] for r in rows}
    if any(set(grid[s])!=expected for s in ('baseline','combined')):raise ValueError('Incomplete action grid')
    del base
    verify_policy(risk_parent,'q_ridge_risk_strict')
    risk={r['opportunity_id']:r for r in ev.read_jsonl(risk_parent/'failure_forecasts.jsonl')}
    cfg=config_for(plan)
    if not cfg.entry_action_quantity_scale_enabled:raise ValueError('Quantity policy disabled')
    seed=json.loads(Path(plan['seed']).read_text())['mutations'];q3=json.loads(Path(plan['q3']).read_text())['mutations']
    predicate=lambda r,rank:ev.eligible(r,seed,q3,rank=rank)
    sessions=meta['sessions'];windows=ev.outer_windows(sessions,min_train=60,block=refit_sessions)
    names=['cv_'+m+('_'+feature_set if feature_set!='continuation' else '')+('_'+window if window!='expanding' else '')+('_refit'+str(refit_sessions) if refit_sessions!=20 else '') for m in models]
    snapshots={n:[] for n in names};decisions={n:[] for n in names};proofs=[];diagnostics=[]
    risk_proofs=[r for r in json.loads((risk_parent/'chronology.json').read_text())['folds'] if r['name']=='execution_failure_probability']
    risk_windows={p['valid_from']:p['valid_through'] for p in risk_proofs}
    with (output/'forecasts.jsonl').open('x',encoding='utf-8') as forecasts,threadpool_limits(limits=1):
        for fold,(first,through) in enumerate(windows):
            for kind,name in zip(models,names):
                scores={}
                for scenario in ('baseline','combined'):
                    model,predictions,proof=cv.fit_forecast(rows,grid[scenario],sessions,first,through,kind,window,columns=columns,interactions=interactions)
                    p=output/f'model_{name}_{scenario}_{fold}.joblib';joblib.dump(model,p)
                    proof.update(scenario=scenario,model_path=p.name,model_sha256=ev.digest(p));proofs.append(proof)
                    scores[scenario]={r['opportunity_id']:v for r,v in predictions}
                by_day=defaultdict(list)
                for r,_ in predictions:
                    key=r['opportunity_id']
                    check_risk_clock(risk[key],risk_windows,first,r['trade_date'])
                    emit(forecasts,{'opportunity_id':key,'model':name,'valid_from':first,
                        'baseline_value_bps':scores['baseline'][key],'combined_value_bps':scores['combined'][key],
                        'failure_probability':risk[key]['failure_probability']})
                    if risk[key]['failure_probability']<.001:by_day[r['trade_date']].append(r)
                for day in [d for d in sessions if first<=d<=through]:
                    choices,_=choose_quantities(by_day[day],scores['baseline'],scores['combined'],robust=True)
                    selected,reasons=admit_actions(list(choices.values()),predicate,cfg,meta['opportunities_sha256'])
                    provenance={'policy':name,'valid_from':first,'maximum_failure_probability':.001,
                        'value_refit_sessions':refit_sessions,'risk_refit_sessions':20}
                    snapshots[name].append(ev.snapshot_for(day,selected,source=meta['opportunities_sha256'],forecast=provenance).to_json_dict())
                    for r,value in selected:
                        decisions[name].append({'opportunity_id':r['base_opportunity_id'],'action_id':r['opportunity_id'],
                            'trade_date':day,'score':value,'reason':'admitted','quantity_scale':r['features']['action_quantity_scale'],**provenance})
                diagnostic={'fold':fold,'policy':name,'from':first,'through':through,'cumulative_admissions':len(decisions[name])}
                diagnostics.append(diagnostic);print(json.dumps(diagnostic),flush=True)
    for n in names:
        ev.write_json(output/(n+'.snapshots.json'),snapshots[n])
        with (output/(n+'.admission.jsonl')).open('x',encoding='utf-8') as f:
            for r in decisions[n]:emit(f,r)
    ev.write_json(output/'chronology.json',{'folds':proofs+risk_proofs,'scope':'Past-only fits on reused development history; no independent OOS claim'})
    ev.write_json(output/'fit_summary.json',diagnostics)
    ev.write_json(output/'policies_manifest.json',{'version':'kalcb-continuation-policy-v1','policies':names,
        'evaluation_from':windows[0][0],'evaluation_through':windows[-1][1],
        'files':{p.name:ev.digest(p) for p in output.iterdir() if p.is_file()},
        'contract':{'budget':8,'maximum_failure_probability':.001,'quantity_flag_required':True,'production_promotion':False,
            'value_refit_sessions':refit_sessions,'risk_refit_sessions':20},
        'prepared_sha256':ev.digest(prepared/'prepared.json'),'label_manifests':proofs_labels,'grid_manifests':proofs_grid,
        'risk_parent_manifest_sha256':ev.digest(risk_parent/'policies_manifest.json')})


def main():
    from backtests.auto.shared.cache_keys import source_generation_hash
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','fit'))
    for name in ('plan','prepared','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--labels',type=Path,action='append',default=[]);p.add_argument('--grid',type=Path,action='append',default=[])
    p.add_argument('--risk-parent',type=Path);p.add_argument('--models',nargs='+',choices=('interaction','mixture'),default=['interaction','mixture'])
    p.add_argument('--window',choices=('expanding','rolling120','decay60'),default='expanding')
    p.add_argument('--refit-sessions',type=int,choices=(5,20),default=20)
    p.add_argument('--feature-set',choices=('continuation','campaign','campaign_interactions','leadership'),default='continuation')
    a=p.parse_args();plan=load_plan(a.plan);source=source_generation_hash();a.output.mkdir(parents=True,exist_ok=False)
    ev.write_json(a.output/'request.json',{'arguments':vars(a),'source_generation_hash':source,'plan_sha256':ev.digest(a.plan)})
    if a.command=='prepare':prepare(plan,a.prepared,a.output)
    else:
        if not a.labels or not a.grid or not a.risk_parent:raise ValueError('Missing fitted-policy dependencies')
        if len(a.models)!=len(set(a.models)):raise ValueError('Repeated model')
        fit(plan,a.prepared,a.labels,a.grid,a.risk_parent,a.output,a.models,a.window,a.refit_sessions,a.feature_set)
    if source_generation_hash()!=source:raise RuntimeError('Code generation changed')
    ev.write_json(a.output/'completion.json',{'complete':True,'source_generation_hash':source})


if __name__=='__main__':main()

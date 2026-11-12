"""Past-only checkpoint models and timed additions to a locked opening policy."""
import argparse,json,shutil,sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime,date
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev,continuation_value as cv,action_value as av,paired_value as pv
from scripts.kalcb_executed_value_study import load_plan,config_for,emit
from scripts.kalcb_action_quantity_labels import scaled_row
from scripts.kalcb_action_quantity_ranker import choose_quantities,capacity_ok
from scripts.kalcb_management_refit import changed_grid,failure_outcomes
from scripts.kalcb_executed_value_replay import verify_policy,timed_policy


def update_snapshot(base,selected,source):
    snap=ev.snapshot_for(str(base.trade_date),selected,source=source)
    new=tuple(replace(c,metadata={**c.metadata,'candidate_rank':len(base.candidates)+i,
        'frontier_rank':len(base.candidates)+i}) for i,c in enumerate(snap.candidates,1))
    return replace(snap,candidates=base.candidates+new,metadata={**snap.metadata,
        'active_symbols':[c.symbol for c in base.candidates+new],
        'active_symbol_count':len(base.candidates+new),'candidate_pool_count':len(base.candidates+new),
        'selection_count':len(base.candidates+new),'frontier_symbol_count':len(base.candidates+new)})


def native_admit(base,choices,cfg,source):
    from strategy_common.market import MarketBar
    from strategy_kalcb.core.preview import preview_entry
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    selected=[];existing=set(base.by_symbol())
    for row,value in sorted(choices.values(),key=lambda x:(-x[1],x[0]['symbol'])):
        if row['symbol'] in existing:continue
        trial=selected+[(row,value)];snap=update_snapshot(base,trial,source)
        bars=[MarketBar(r['symbol'],datetime.fromisoformat(b['timestamp']),'5m',*[b[k] for k in ('open','high','low','close','volume')]) for r,_ in trial for b in r['opening_bars']]
        preview=preview_entry(bars,snap,cfg,KALCBPortfolioView(cash=ev.REFERENCE_EQUITY,equity=ev.REFERENCE_EQUITY),available_at=snap.generated_at)
        actual={a['symbol'] for a in preview['intents']}
        if all(r['symbol'] in actual for r,_ in trial):selected=trial
        if len(selected)>=4:break
    return selected


def fit(plan,prepared,paths,parent,name,output):
    import joblib
    from threadpoolctl import threadpool_limits
    manifest,base=verify_policy(parent,name)
    if timed_policy(parent,manifest,base):raise ValueError('Opening parent must have no later decisions')
    meta=json.loads((prepared/'prepared.json').read_text())
    if ev.digest(prepared/'opportunities.jsonl')!=meta['opportunities_sha256']:raise ValueError('Changed checkpoint data')
    rows=[scaled_row(r,q) for r in ev.read_jsonl(prepared/'opportunities.jsonl') if r['action']['ready'] and r['guard_at_rank1'] for q in pv.QUANTITIES]
    grid,gp=changed_grid(paths,prepared,plan,{r['opportunity_id'] for r in rows})
    risk_out=failure_outcomes(grid);cfg=config_for(plan);sessions=meta['sessions'];proofs=[]
    columns=cv.COLUMNS+pv.CHECKPOINT_COLUMNS
    recipes={'paired_half':('paired',.5),'paired_full':('paired',1.),'interaction_half':('interaction',.5),
        'paired_calibrated_half':('paired',.5)}
    observations=defaultdict(list);calibration=[]
    policies={n:[] for n in recipes};decisions={n:[] for n in recipes}
    with (output/'forecasts.jsonl').open('x') as f,threadpool_limits(limits=1):
        for fold,(first,through) in enumerate(ev.outer_windows(sessions,min_train=60,block=20)):
            model,predictions,proof=av.fit_feasibility(rows,risk_out,sessions,first,through)
            p=output/f'risk_{fold}.joblib';joblib.dump(model,p);proof.update(model_path=p.name,model_sha256=ev.digest(p));proofs.append(proof)
            risk={r['opportunity_id']:v for r,v in predictions}
            for kind in ('paired','interaction'):
                from backtests.strategies.kalcb.value_calibration import fit_support_offsets,score_support
                offsets,calproof=fit_support_offsets(observations[kind],grid,sessions,first)
                calproof.update(model=kind,valid_through=through);calibration.append(calproof)
                scores={}
                for scenario in ('baseline','combined'):
                    if kind=='paired':model,predictions,proof=pv.fit_forecast(rows,grid[scenario],sessions,first,through,columns=columns)
                    else:model,predictions,proof=cv.fit_forecast(rows,grid[scenario],sessions,first,through,columns=columns)
                    p=output/f'{kind}_{scenario}_{fold}.joblib';joblib.dump(model,p)
                    proof.update(scenario=scenario,model_path=p.name,model_sha256=ev.digest(p));proofs.append(proof)
                    scores[scenario]={r['opportunity_id']:v for r,v in predictions}
                daily=defaultdict(list)
                values={}
                for r,_ in predictions:
                    key=r['opportunity_id'];v={'opportunity_id':key,'model':kind,'valid_from':first,
                        'baseline_value_bps':scores['baseline'][key],'combined_value_bps':scores['combined'][key],'failure_probability':risk[key],
                        'trade_date':r['trade_date'],'quantity_scale':r['features']['action_quantity_scale'],'capacity_ok':capacity_ok(r)}
                    emit(f,v);values[key]=v;observations[kind].append(v)
                    if risk[key]<.001:daily[r['trade_date']].append(r)
                for n,(model_kind,cap) in recipes.items():
                    if model_kind!=kind:continue
                    selected_scores=scores
                    if 'calibrated' in n:
                        selected_scores={s:{key:value+offsets[(s,values[key]['quantity_scale'],score_support(values[key]))]
                            for key,value in scores[s].items()} for s in scores}
                    for day in (d for d in sessions if first<=d<=through):
                        choices,_=choose_quantities([r for r in daily[day] if r['features']['action_quantity_scale']<=cap],selected_scores['baseline'],selected_scores['combined'],robust=True)
                        selected=native_admit(base[date.fromisoformat(day)],choices,cfg,meta['opportunities_sha256'])
                        if selected:
                            snap=update_snapshot(base[date.fromisoformat(day)],selected,meta['opportunities_sha256'])
                            policies[n].append(snap.to_json_dict())
                            for r,v in selected:decisions[n].append({'trade_date':day,'symbol':r['symbol'],'available_at':r['feature_available_at'],
                                'score':v,'action_id':r['opportunity_id'],'quantity_scale':r['features']['action_quantity_scale'],'valid_from':first})
            print(json.dumps({'fold':fold,'through':through,'admissions':{n:len(d) for n,d in decisions.items()}}),flush=True)
    # Bind every model and the complete opening lineage inside each replay policy.
    opening_chronology=json.loads((parent/'chronology.json').read_text())
    ev.write_json(output/'chronology.json',{'folds':proofs,'calibration':calibration,'scope':'Reused development history; past-only models; no independent validation claim'})
    fitted_files={p.name:ev.digest(p) for p in output.iterdir() if p.is_file()}
    for n in recipes:
        target=output/n;target.mkdir()
        for filename in manifest['files']:
            shutil.copy2(parent/filename,target/('opening_'+filename))
        for filename in fitted_files:shutil.copy2(output/filename,target/filename)
        shutil.copy2(parent/(name+'.snapshots.json'),target/(n+'.snapshots.json'))
        shutil.copy2(parent/(name+'.admission.jsonl'),target/(n+'.admission.jsonl'))
        ev.write_json(target/'chronology.json',{'folds':opening_chronology['folds']+proofs,
            'calibration':opening_chronology.get('calibration',[])+calibration,'scope':'Chronological reused development only'})
        for filename,items in [('timed.snapshots.jsonl',policies[n]),('timed.admission.jsonl',decisions[n])]:
            with (target/filename).open('x') as f:
                for r in items:emit(f,r)
        ev.write_json(target/'policies_manifest.json',{'version':'kalcb-timed-checkpoint-v1','policies':[n],
            'evaluation_from':manifest['evaluation_from'],'evaluation_through':manifest['evaluation_through'],
            'files':{p.name:ev.digest(p) for p in target.iterdir() if p.is_file()},
            'timed_snapshot_file':'timed.snapshots.jsonl','timed_admission_file':'timed.admission.jsonl',
            'observation_symbols':meta['observation_symbols'],'prepared_sha256':ev.digest(prepared/'prepared.json'),
            'observation_minutes':45,
            'grid_manifests':gp,'opening_parent_manifest_sha256':ev.digest(parent/'policies_manifest.json'),
            'contract':{'budget':8,'timed_budget':12,'new_budget':4,'quantity_cap':recipes[n][1],
                'maximum_failure_probability':.001,'value_threshold':0,'production_promotion':False}})
        ev.write_json(target/'completion.json',{'complete':True})
        m,s=verify_policy(target,n);timed_policy(target,m,s)
    ev.write_json(output/'summary.json',{'admissions':{n:len(d) for n,d in decisions.items()}})


def main():
    from backtests.auto.shared.cache_keys import source_generation_hash
    p=argparse.ArgumentParser()
    for k in ('plan','prepared','parent','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--labels',type=Path,action='append',required=True)
    a=p.parse_args();plan=load_plan(a.plan);source=source_generation_hash();a.output.mkdir(parents=True,exist_ok=False)
    ev.write_json(a.output/'request.json',{'arguments':vars(a),'source_generation_hash':source})
    fit(plan,a.prepared,a.labels,a.parent,a.name,a.output)
    if source_generation_hash()!=source:raise RuntimeError('Source changed')
    ev.write_json(a.output/'completion.json',{'complete':True,'source_generation_hash':source})


if __name__=='__main__':main()

"""Ablate incumbent quantity locking and perturb the checkpoint value hurdle."""
from collections import defaultdict
from datetime import date,datetime,timezone
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_full_oos_recovery import ROOT,FULL,read,write,bound,digest,rows,jsonl,link,POLICY
from scripts.kalcb_recovery_refit import HISTORY,assemble
from scripts.kalcb_recovery_ablation import RecoveryPlugin
from scripts.kalcb_action_quantity_labels import scaled_row
from scripts.kalcb_action_quantity_ranker import choose_quantities
from scripts.kalcb_action_value_study import admit_actions
from scripts.kalcb_checkpoint_fit import native_admit,update_snapshot
from scripts.kalcb_checkpoint_variant import meets_value_margin
from scripts.kalcb_executed_value_study import config_for
from scripts.kalcb_executed_value_replay import verify_policy,timed_policy
from scripts.portfolio_frozen_inference import calibrated,predictions
from backtests.strategies.kalcb import executed_value as ev
from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.auto.shared.round_manager import RoundManager

RECIPES={'reprice_locked_core':{'unlock':True,'margin':48},
    'checkpoint_margin24':{'unlock':False,'margin':24},'checkpoint_margin0':{'unlock':False,'margin':0}}

def scaled_window(path,first,last):
    result=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if first<=r['trade_date']<=last and r['action']['ready'] and r['guard_at_rank1']:
                result.extend(scaled_row(r,q) for q in (.25,.5,1.))
    return result

def unlock_choices(items,raw,cals,plan,source):
    core=calibrated(raw,cals['frozen_core']);new=calibrated(raw,cals['new_forecast'])
    seed=read(plan['seed'])['mutations'];q3=read(plan['q3'])['mutations'];cfg=config_for(plan)
    pred=lambda row,rank:ev.eligible(row,seed,q3,rank=rank)
    scores=lambda v:{s:{k:r[s+'_value_bps'] for k,r in v.items()} for s in ('baseline','combined')}
    cs,ns=scores(core),scores(new)
    choose=lambda rs,vs:choose_quantities(rs,vs['baseline'],vs['combined'],robust=True)[0]
    admit=lambda chosen:admit_actions(list(chosen.values()),pred,cfg,source,budget=8)[0]
    feasible=[r for r in items if raw[r['opportunity_id']]['failure_probability']<.001]
    positive=[r for r in feasible if min(raw[r['opportunity_id']][s+'_value_bps'] for s in ('baseline','combined'))>0]
    initial={r['base_opportunity_id']:(r,v) for r,v in admit(choose(positive,cs))}
    expanded=choose(feasible,cs);expanded.update(initial)
    locked={r['base_opportunity_id']:(r,v) for r,v in admit(expanded)}
    eligible=[r for r in feasible if r['base_opportunity_id'] in locked or r['features']['action_quantity_scale']<=.5]
    # One ablation: current support-calibrated forecasts may resize/reject the
    # incumbent core. The half-size restriction on new support names remains.
    return admit(choose(eligible,ns)),locked

def save(parent,dest,result,recipe,evidence):
    dest.mkdir(parents=True,exist_ok=False)
    changed={POLICY+'.snapshots.json',POLICY+'.admission.jsonl','timed.snapshots.jsonl','timed.admission.jsonl','completion.json','policies_manifest.json'}
    for p in parent.iterdir():
        if p.is_file() and p.name not in changed:link(p,dest/p.name)
    write(dest/(POLICY+'.snapshots.json'),[s.to_json_dict() for s in result['snapshots'].values()])
    jsonl(dest/(POLICY+'.admission.jsonl'),result['admissions'])
    jsonl(dest/'timed.snapshots.jsonl',(s.to_json_dict() for a in result['updates'].values() for s in a))
    jsonl(dest/'timed.admission.jsonl',result['timed_admissions'])
    write(dest/'admission_experiment.json',{'recipe':recipe,'evidence':evidence,'program':bound(__file__)})
    manifest={**read(parent/'policies_manifest.json'),'files':{p.name:digest(p) for p in dest.iterdir() if p.is_file()},
        'admission_experiment':recipe}
    write(dest/'policies_manifest.json',manifest);write(dest/'completion.json',{'complete':True})
    m,s=verify_policy(dest,POLICY);timed_policy(dest,m,s)

def prepare(root):
    target=root/'admission_round';target.mkdir(exist_ok=False)
    write(target/'protocol.json',{'recipes':RECIPES,'registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'program':bound(__file__),'purpose':'One accepted core-lock removal and local checkpoint hurdle perturbations. Both IS and full OOS are replayed.'})
    study=read(root/'ablation_v3/study.json');overrides={'is':{},'oos':{}};bindings=[]
    fit_directory=Path(read(FULL/'frozen_inference_bridge.json')['models'][0]['source']['path']).parent
    for period in ('is','oos'):
        parent=Path(study['periods'][period]['policies']);m,base=verify_policy(parent,POLICY);old_timed=timed_policy(parent,m,base)
        if period=='is':
            op=HISTORY/'kalcb_continuation_20261006/prepared_v1';tp=HISTORY/'kalcb_intraday_advantage_20261007/checkpoint/prepared'
            opening_plan=read(read(fit_directory/'opening_core_request.json')['arguments']['plan'])
            timed_plan=read(HISTORY/'kalcb_intraday_advantage_20261007/checkpoint/plan.json')
            raw_all={r['opportunity_id']:r for r in rows(HISTORY/'kalcb_continuation_20261006/policies_v2/forecasts.jsonl') if r['model']=='cv_interaction'}
            timed_all={r['opportunity_id']:r for r in rows(HISTORY/'kalcb_intraday_advantage_20261007/checkpoint/policies/forecasts.jsonl') if r['model']=='interaction'}
            windows=defaultdict(list)
            for r in raw_all.values():windows[r['valid_from']].append(r['opportunity_id'][:10])
            windows=[(k,max(v)) for k,v in sorted(windows.items())]
        else:
            op=FULL/'forward_features/continuation';tp=FULL/'forward_features/checkpoint'
            opening_plan=read(FULL/'forward_features/plan.json');timed_plan=read(FULL/'forward_features/checkpoint_plan.json')
            raw_all=None;timed_all=None;windows=[('2026-04-01','2026-07-10')]
        evidence=[bound(p/'prepared.json') for p in (op,tp)]
        source=digest(op/'opportunities.jsonl');tsource=digest(tp/'opportunities.jsonl')
        if source!=read(op/'prepared.json')['opportunities_sha256'] or tsource!=read(tp/'prepared.json')['opportunities_sha256']:raise ValueError('Changed candidate population')
        collected={n:{'snapshots':{},'updates':{},'admissions':[],'timed_admissions':[]} for n in RECIPES};comparisons=0
        for first,last in windows:
            opening=scaled_window(op/'opportunities.jsonl',first,last);timed=scaled_window(tp/'opportunities.jsonl',first,last)
            if period=='oos':raw,_=predictions(fit_directory,opening,'opening');tv,_=predictions(fit_directory,timed,'timed')
            else:
                raw={r['opportunity_id']:{**raw_all[r['opportunity_id']],'quantity_scale':r['features']['action_quantity_scale']} for r in opening}
                tv={r['opportunity_id']:{**timed_all[r['opportunity_id']],'quantity_scale':r['features']['action_quantity_scale']} for r in timed}
            clock=first if period=='is' else '2026-03-10';cals={}
            for origin,prefix,mode in [('frozen_core','opening_core_','offset_all'),('new_forecast','opening_','support_offset')]:
                matches=[r for r in read(fit_directory/(prefix+'chronology.json'))['calibration'] if r['valid_from']==clock and r.get('mode')==mode and not r.get('lineage')]
                if len(matches)!=1:raise ValueError('Ambiguous historical calibration')
                cals[origin]=matches[0]
            dates=[str(d) for d in base if first<=str(d)<=last]
            bridge=assemble(opening,timed,raw,tv,cals,opening_plan,timed_plan,dates,source,tsource)
            choices=lambda s:{c.symbol:(c.metadata.get('action_quantity_scale',1),c.selection_score) for c in s.candidates}
            for day in dates:
                d=date.fromisoformat(day);snap=base[d]
                for a,b in [(snap,bridge['snapshots'][d]),(old_timed.get(d,[snap])[-1],bridge['updates'].get(d,[bridge['snapshots'][d]])[-1])]:
                    if choices(a)!=choices(b):raise ValueError('Historical selector bridge drift '+period+' '+day)
                    comparisons+=1
            opening_days=defaultdict(list);timed_days=defaultdict(list)
            for r in opening:opening_days[r['trade_date']].append(r)
            for r in timed:timed_days[r['trade_date']].append(r)
            scores={s:{k:r[s+'_value_bps'] for k,r in tv.items()} for s in ('baseline','combined')}
            for name,recipe in RECIPES.items():
                result=collected[name]
                for day in dates:
                    d=date.fromisoformat(day)
                    if recipe['unlock']:
                        selected,locked=unlock_choices(opening_days[day],raw,cals,opening_plan,source)
                        snap=ev.snapshot_for(day,selected,source=source,forecast={'policy':name,'valid_from':clock})
                        result['admissions'].extend({'opportunity_id':r['base_opportunity_id'],'action_id':r['opportunity_id'],
                            'trade_date':day,'reason':'admitted','score':v,'quantity_scale':r['features']['action_quantity_scale'],
                            'valid_from':clock,'forecast_origin':'repriced_core' if r['base_opportunity_id'] in locked else 'new_forecast'} for r,v in selected)
                    else:
                        snap=base[d];result['admissions'].extend(r for r in bridge['admissions'] if r['trade_date']==day)
                    result['snapshots'][d]=snap
                    eligible=[r for r in timed_days[day] if r['features']['action_quantity_scale']<=.5 and tv[r['opportunity_id']]['failure_probability']<.001 and meets_value_margin(r,tv[r['opportunity_id']],recipe['margin'])]
                    chosen,_=choose_quantities(eligible,scores['baseline'],scores['combined'],robust=True)
                    extra=native_admit(snap,chosen,config_for(timed_plan),tsource)
                    if extra:
                        result['updates'][d]=[update_snapshot(snap,extra,tsource)]
                        result['timed_admissions'].extend({'trade_date':day,'symbol':r['symbol'],'available_at':r['feature_available_at'],
                            'score':v,'action_id':r['opportunity_id'],'quantity_scale':r['features']['action_quantity_scale'],'valid_from':clock} for r,v in extra)
            print(json.dumps({'period':period,'fold':first,'bridge_checks':comparisons}),flush=True)
        write(target/(period+'_bridge.json'),{'passed':True,'day_phases':comparisons,'parent':bound(parent/'policies_manifest.json')})
        for name,recipe in RECIPES.items():
            dest=target/'policies'/period/name;save(parent,dest,collected[name],recipe,evidence)
            overrides[period][name]=str(dest);bindings.append(bound(dest/'policies_manifest.json'))
    spec={**study,'registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'recipes':{'control':{'config':{},'policy':'control'},**{n:{'config':{},'policy':'control'} for n in RECIPES}},
        'policy_overrides_by_period':overrides,'bound_inputs':study['bound_inputs']+bindings,
        'cached_is':{'control':study['cached_is']['control']},'source_programs':[bound(__file__),bound(Path(__file__).with_name('kalcb_recovery_ablation.py')),bound(Path(__file__).with_name('kalcb_recovery_refit.py'))],
        'workers':2,'purpose':'Test core repricing and checkpoint hurdle with exact historical selector bridge and native IS/OOS accounts.'}
    write(target/'study.json',spec)

class AdmissionPlugin(RecoveryPlugin):
    name='kalcb_full_oos_admission_perturbation'
    def paths(self,name,period):
        folder,plan,policies=super().paths(name,period)
        if name!='control':policies=Path(self.spec['policy_overrides_by_period'][period][name])
        return folder,plan,policies

def run(root):
    plugin=AdmissionPlugin(root/'admission_round/study.json');manager=RoundManager('stock',plugin.name,base_dir=plugin.root/'framework')
    PhaseRunner(plugin,manager.get_round_dir(1),round_name='Core repricing and checkpoint hurdle',round_manager=manager,round_num=1,
        max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    prepare(a.root) if a.stage=='prepare' else run(a.root)

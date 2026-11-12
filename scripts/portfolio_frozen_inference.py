"""Infer the selected layered KALCB policy without fitting validation outcomes.

The March 10 model fit and its preceding calibration are held fixed. A required
development reconstruction checks every selected action in the last saved fold.
"""
from collections import defaultdict
from datetime import date
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from backtests.strategies.kalcb import executed_value as ev, continuation_value as cv, action_value as av
from scripts.kalcb_action_quantity_labels import scaled_row
from scripts.kalcb_action_quantity_ranker import choose_quantities
from scripts.kalcb_action_value_study import admit_actions
from scripts.kalcb_core_preserving_policy import merge_core_choices
from scripts.kalcb_checkpoint_fit import native_admit,update_snapshot
from scripts.kalcb_checkpoint_variant import meets_value_margin
from scripts.kalcb_executed_value_study import config_for
from scripts.portfolio_targeted_validation import ROOT,read,bound

FIT='2026-03-10'


def model_for(directory,prefix,proof):
    import joblib
    direct=directory/(prefix+proof['model_path']);sha=proof['model_sha256']
    if direct.exists() and ev.digest(direct)==sha:return joblib.load(direct),bound(direct)
    # Scoped research bundles retained some feasibility models as additional
    # evidence rather than duplicating them under each policy prefix.
    matches=[p for p in directory.parent.rglob(Path(proof['model_path']).name) if ev.digest(p)==sha]
    if not matches:raise ValueError('Missing frozen model '+proof['model_path'])
    return joblib.load(matches[0]),bound(matches[0])


def predictions(directory,rows,kind):
    from threadpoolctl import threadpool_limits
    if not rows:return {},[]
    prefix='opening_' if kind=='opening' else ''
    chronology=read(directory/(prefix+'chronology.json'))
    proofs=[p for p in chronology['folds'] if p['valid_from']==FIT and not p.get('lineage')]
    values={r['opportunity_id']:{'opportunity_id':r['opportunity_id'],'valid_from':FIT,
            'quantity_scale':r['features']['action_quantity_scale'],'trade_date':r['trade_date']} for r in rows}
    used=[]
    for target in ('baseline','combined','failure'):
        matching=[p for p in proofs if (p['name']=='execution_failure_probability' if target=='failure'
             else p['name']=='interaction' and p.get('scenario')==target)
             and (kind=='opening' or p['model_path'].startswith('risk_' if target=='failure' else 'interaction_'))]
        if len(matching)!=1:raise ValueError('Ambiguous frozen inference proof: '+str((kind,target)))
        proof=matching[0]
        if proof['train_through']>=FIT or proof['label_available_through']>=proof['fit_at']:
            raise ValueError('Noncausal frozen model')
        model,source=model_for(directory,prefix,proof)
        columns=proof['feature_columns'];matrix=cv.matrix(rows,columns)
        with threadpool_limits(limits=1):forecast=model.predict(matrix)
        key='failure_probability' if target=='failure' else target+'_value_bps'
        for row,value in zip(rows,forecast):values[row['opportunity_id']][key]=float(value)
        used.append({'kind':kind,'target':target,'source':source,'training_proof':proof})
    return values,used


def calibrated(raw,calibration):
    output={}
    for key,row in raw.items():
        q=row['quantity_scale'];support='positive' if min(row[s+'_value_bps'] for s in ('baseline','combined'))>0 else 'nonpositive'
        value={**row,'raw_score_support':support}
        for scenario in ('baseline','combined'):
            cohort=f'{scenario}/q{q:g}'+('/'+support if calibration['mode']=='support_offset' else '')
            value[scenario+'_value_bps']+=calibration['cohorts'][cohort]['offset_bps']
        output[key]=value
    return output


def infer(directory,opening_rows,timed_rows,opening_plan,timed_plan,sessions,source,tail=(),timed_source=None):
    """One fixed model/calibration; no trade ledger or OOS label argument exists."""
    if not sessions or min(sessions)<FIT:raise ValueError('Frozen model cannot predict before its fit')
    opening_rows=[scaled_row(r,q) for r in opening_rows if r['action']['ready'] and r['guard_at_rank1']
                  for q in (.25,.5,1.) if r['trade_date'] in sessions]
    timed_rows=[scaled_row(r,q) for r in timed_rows if r['action']['ready'] and r['guard_at_rank1']
                for q in (.25,.5,1.) if r['trade_date'] in sessions]
    raw,proof=predictions(directory,opening_rows,'opening')
    timed,proof2=predictions(directory,timed_rows,'timed')
    cals={}
    for origin,prefix,mode in [('frozen_core','opening_core_','offset_all'),('new_forecast','opening_','support_offset')]:
        matches=[r for r in read(directory/(prefix+'chronology.json'))['calibration']
                 if r['valid_from']==FIT and r.get('mode')==mode and not r.get('lineage')]
        if len(matches)!=1:raise ValueError('Ambiguous frozen calibration')
        cals[origin]=copy.deepcopy(matches[0])
    core=calibrated(raw,cals['frozen_core']);new=calibrated(raw,cals['new_forecast'])
    cfg=config_for(opening_plan);tcfg=config_for(timed_plan)
    seed=read(opening_plan['seed'])['mutations'];q3=read(opening_plan['q3'])['mutations']
    predicate=lambda row,rank:ev.eligible(row,seed,q3,rank=rank)
    by_day=defaultdict(list);timed_days=defaultdict(list)
    for row in opening_rows:by_day[row['trade_date']].append(row)
    for row in timed_rows:timed_days[row['trade_date']].append(row)
    base={};updates={};admissions=[];tadmissions=[]
    scores=lambda values:{s:{k:r[s+'_value_bps'] for k,r in values.items()} for s in ('baseline','combined')}
    cs,ns,ts=scores(core),scores(new),scores(timed)
    for day in sessions:
        items=by_day[day];feasible=[r for r in items if raw[r['opportunity_id']]['failure_probability']<.001]
        original=[r for r in feasible if min(raw[r['opportunity_id']][s+'_value_bps'] for s in ('baseline','combined'))>0]
        def choose(items,values):return choose_quantities(items,values['baseline'],values['combined'],robust=True)[0]
        def admit(choices):return admit_actions(list(choices.values()),predicate,cfg,source,budget=8)[0]
        original_selected=admit(choose(original,cs))
        original_locked={r['base_opportunity_id']:(r,v) for r,v in original_selected}
        expanded=choose(feasible,cs);expanded.update(original_locked)
        selected_core=admit(expanded)
        locked={r['base_opportunity_id']:(r,v) for r,v in selected_core}
        eligible=[r for r in feasible if r['base_opportunity_id'] in locked or r['features']['action_quantity_scale']<=.5]
        supported=choose(eligible,ns);supported.update(locked)
        selected=admit(supported) if day not in tail else []
        snap=ev.snapshot_for(day,selected,source=source,forecast={'policy':'frozen_march10_interaction_cost_margin','fit_at':FIT+'T00:00:00+09:00'})
        base[date.fromisoformat(day)]=snap
        for row,value in selected:
            admissions.append({'opportunity_id':row['base_opportunity_id'],'action_id':row['opportunity_id'],
                'trade_date':day,'reason':'admitted','score':value,'quantity_scale':row['features']['action_quantity_scale'],
                'valid_from':FIT,'forecast_origin':'frozen_core' if row['base_opportunity_id'] in locked else 'new_forecast'})
        candidates=[r for r in timed_days[day] if r['features']['action_quantity_scale']<=.5
            and timed[r['opportunity_id']]['failure_probability']<.001
            and meets_value_margin(r,timed[r['opportunity_id']],48)]
        extra=native_admit(snap,choose(candidates,ts),tcfg,timed_source or source) if day not in tail else []
        if extra:
            updates[date.fromisoformat(day)]=[update_snapshot(snap,extra,timed_source or source)]
            for row,value in extra:tadmissions.append({'trade_date':day,'symbol':row['symbol'],
                'available_at':row['feature_available_at'],'score':value,'action_id':row['opportunity_id'],
                'quantity_scale':row['features']['action_quantity_scale'],'valid_from':FIT})
    return {'snapshots':base,'updates':updates,'admissions':admissions,'timed_admissions':tadmissions,
            'raw':raw,'core':core,'new':new,'timed':timed,'calibrations':cals,'model_proofs':proof+proof2}


def bridge(root):
    from scripts.kalcb_executed_value_replay import verify_policy,timed_policy
    study=read(root/'development/study.json');p=Path(study['code'])/study['kalcb_policies']
    pm,snapshots=verify_policy(p,study['kalcb_policy']);updates=timed_policy(p,pm,snapshots)
    original_plan=read(Path(read(p/'opening_core_request.json')['arguments']['plan']))
    timed_plan=read(Path(read(p/'request.json')['arguments']['plan']))
    prepared=p.parent/'additional_evidence_9';checkpoint=p.parent/'prepared'
    sessions=[str(d) for d in snapshots if str(d)>=FIT]
    result=infer(p,ev.read_jsonl(prepared/'opportunities.jsonl'),ev.read_jsonl(checkpoint/'opportunities.jsonl'),
                 original_plan,timed_plan,sessions,'inference_bridge')
    def choices(snap):return {c.symbol:(float(c.metadata.get('action_quantity_scale',1)),c.selection_score) for c in snap.candidates}
    errors=[]
    for day,snap in result['snapshots'].items():
        for phase,a,b in [('opening',choices(snap),choices(snapshots[day])),
            ('timed',choices(result['updates'].get(day,[snap])[-1]),choices(updates.get(day,[snapshots[day]])[-1]))]:
            if a.keys()!=b.keys() or any(a[s][0]!=b[s][0] or abs(a[s][1]-b[s][1])>1e-8 for s in a.keys()&b.keys()):
                errors.append({'day':str(day),'phase':phase,'inferred':a,'saved':b})
    payload={'pass':not errors,'sessions':len(sessions),'errors':errors,'study':bound(root/'development/study.json'),
        'program':bound(Path(__file__)),'models':result['model_proofs'],'scope':'Last development fold action membership, quantity and scores; no OOS outcomes.'}
    write_json=root/'frozen_inference_bridge.json';ev.write_json(write_json,payload)
    print(json.dumps({'inference_bridge_pass':not errors,'sessions':len(sessions),'mismatched_day_phases':len(errors)}),flush=True)
    if errors:raise ValueError('Frozen inference differs from saved selected policy')


if __name__=='__main__':bridge(ROOT)

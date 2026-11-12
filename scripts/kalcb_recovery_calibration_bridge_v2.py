"""Predeclared calibration-update damping using already causal saved forecasts.

This changes the April deployment rule only. Historical selected IS remains a
reference, not a new validation of this alternative calibration update rule.
"""
from copy import deepcopy
from datetime import date, datetime, timezone
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_full_oos_recovery import ROOT,FULL,POLICY,read,write,rows,bound,digest
from scripts.kalcb_recovery_refit import ready,assemble,save_policy
from scripts.kalcb_executed_value_replay import verify_policy,timed_policy


def blend(frozen, adaptive, weight, through):
    if not 0 <= weight <= 1:
        raise ValueError('Invalid adaptive calibration weight')
    if frozen['mode'] != adaptive['mode'] or frozen['cohorts'].keys() != adaptive['cohorts'].keys():
        raise ValueError('Calibration cohort mismatch')
    result = deepcopy(adaptive)
    for key, cohort in result['cohorts'].items():
        cohort['offset_bps'] = ((1-weight)*frozen['cohorts'][key]['offset_bps']
                                + weight*adaptive['cohorts'][key]['offset_bps'])
    result['valid_through'] = through
    result['damping'] = {'adaptive_weight':weight,'frozen':frozen,'adaptive':adaptive}
    for field in ('latest_label_available_at','observed_through'):
        values = [c[field] for c in (frozen,adaptive) if c.get(field)]
        if values: result[field] = max(values)
    return result


def run(root, definitions):
    target=root/'calibration_damping_v2';target.mkdir(exist_ok=False)
    write(target/'protocol.json',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'definitions':definitions,'program':bound(__file__),
        'hypothesis':'Updated value forecasts reduce losses; adaptive residual correction may suppress activity. Test a fixed halfway update, with unchanged causal forecasts and execution rules.',
        'scope':'OOS deployment experiment; selected IS is a reference. No extra fitting, no outcome-conditioned admission.'})
    days=read(root/'preparation.json')['sessions']
    data={kind:[r for r in ready(FULL/'forward_features'/stage) if r['trade_date'] in days]
          for kind,stage in [('opening','continuation'),('timed','checkpoint')]}
    frozen=read(root/'policies/frozen/forward_lineage.json')['calibrations']
    op=read(FULL/'forward_features/plan.json');tp=read(FULL/'forward_features/checkpoint_plan.json')
    source=digest(FULL/'forward_features/continuation/opportunities.jsonl')
    tsource=digest(FULL/'forward_features/checkpoint/opportunities.jsonl')
    bridges=[]
    for name,recipe in definitions.items():
        parent=Path(recipe['parent_policy']);forecast=Path(recipe['forecasts'])
        manifest,original=verify_policy(parent,POLICY);original_updates=timed_policy(parent,manifest,original)
        chronology=read(parent/'chronology.json');lineage=read(parent/'lineage.json')
        values={kind:{r['opportunity_id']:r for r in rows(forecast/(kind+'.jsonl'))} for kind in data}
        for kind,rs in data.items():
            if values[kind].keys() != {r['opportunity_id'] for r in rs}:raise ValueError('Forecast coverage mismatch')
            for r in rs:
                if values[kind][r['opportunity_id']]['valid_from'] > r['trade_date']:raise ValueError('Future forecast')
        inherited=read(FULL/'frozen_inference_bridge.json')['models']
        model_evidence={}
        for fold in chronology['folds']:
            model=Path(fold['model_path'])
            if not model.is_absolute():
                matches=[m['source'] for m in inherited if m['source']['sha256']==fold['model_sha256']]
                if len(matches)!=1:raise ValueError('Ambiguous inherited model identity')
                model=Path(matches[0]['path'])
            if digest(model)!=fold['model_sha256']:raise ValueError('Changed fitted model')
            fold['model_path']=str(model.resolve())
            model_evidence[str(model.resolve())]=fold['model_sha256']
        clocks=sorted({r['valid_from'] for r in values['opening'].values()})
        collected={'snapshots':{},'updates':{},'admissions':[],'timed_admissions':[]};proofs=[];checks=0
        for i,clock in enumerate(clocks):
            dates=[d for d in days if clock<=d and (i+1==len(clocks) or d<clocks[i+1])]
            rs={kind:[r for r in data[kind] if r['trade_date'] in dates] for kind in data}
            raw={kind:{r['opportunity_id']:values[kind][r['opportunity_id']] for r in rs[kind]} for kind in data}
            adaptive={}
            for origin,mode in [('frozen_core','offset_all'),('new_forecast','support_offset')]:
                candidates=[c for c in chronology['calibration'] if c['mode']==mode and c['valid_from']<=clock]
                c=max(candidates,key=lambda p:p['valid_from'])
                if c['valid_from'] != clock:raise ValueError('Missing calibration for forecast clock')
                adaptive[origin]=c
            bridge=assemble(rs['opening'],rs['timed'],raw['opening'],raw['timed'],adaptive,op,tp,dates,source,tsource)
            for d in dates:
                key=date.fromisoformat(d)
                for a,b in [(original[key],bridge['snapshots'][key]),
                            (original_updates.get(key,[original[key]])[-1],bridge['updates'].get(key,[bridge['snapshots'][key]])[-1])]:
                    if a.to_json_dict()!=b.to_json_dict():raise ValueError('Unchanged selector bridge drift '+d)
                    checks+=1
            damped={k:blend(frozen[k],c,recipe['adaptive_weight'],dates[-1]) for k,c in adaptive.items()}
            proofs.extend(damped.values())
            result=assemble(rs['opening'],rs['timed'],raw['opening'],raw['timed'],damped,op,tp,dates,source,tsource)
            for k in ('snapshots','updates'):collected[k].update(result[k])
            for k in ('admissions','timed_admissions'):collected[k].extend(result[k])
        dest=target/'policies'/name
        evidence={**lineage['labels'],**model_evidence,str(parent/'policies_manifest.json'):digest(parent/'policies_manifest.json'),
                  **{str(forecast/(k+'.jsonl')):digest(forecast/(k+'.jsonl')) for k in data}}
        save_policy(dest,parent,collected,chronology['folds'],proofs,evidence,recipe)
        generated=read(dest/'lineage.json');generated['risk_models']=lineage['risk_models']
        generated['calibration_damping']={'adaptive_weight':recipe['adaptive_weight'],'source':bound(parent/'policies_manifest.json'),
            'full_selector_bridge_checks':checks,'IS_interpretation':'Historical selected IS reference; alternate rule begins at the April deployment boundary.'}
        write(dest/'lineage.json',generated);m=read(dest/'policies_manifest.json');m['files']['lineage.json']=digest(dest/'lineage.json');write(dest/'policies_manifest.json',m)
        m,s=verify_policy(dest,POLICY);timed_policy(dest,m,s)
        receipt={'recipe':name,'bridge_checks':checks,'opening':len(collected['admissions']),'checkpoint':len(collected['timed_admissions']),
                 'policy':bound(dest/'policies_manifest.json')}
        bridges.append(receipt);print(receipt,flush=True)
    write(target/'completion.json',{'complete':True,'bridges':bridges,'protocol':bound(target/'protocol.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--definitions',type=Path,required=True);a=p.parse_args()
    run(a.root,read(a.definitions))

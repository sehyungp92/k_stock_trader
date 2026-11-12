"""Cross-family selection and matched robustness for the bounded OLR study."""
from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import olr_round1_optimization_v1 as s
from scripts import olr_round1_optimization_results_v1 as results
from scripts import olr_continuous_panel_study_v1 as feature
n=s.n


def gather():
    rows=results.pairs(True)
    for r in rows:r.update(key='legacy:'+r['candidate_id'],family='legacy',root=str(s.ROOT))
    baseline=next(r for r in rows if r['candidate_id']==s.forensic.BASE)
    assert (feature.ROOT/'completion.json').exists()
    for c in n.read(feature.ROOT/'candidate_plan.json')['candidates']:
        row={'candidate_id':c['id'],'label':'Continuous panel + '+c['label'],'key':'continuous:'+c['id'],'family':'continuous','root':str(feature.ROOT)}
        for w in ('train','oos'):
            folder=feature.ROOT/'cases'/c['id']/w/'100000000_baseline'
            a=n.read(folder/'assessment.json');s.forensic.authenticate(folder)
            row[w]={k:a[k] for k in ('return_pct','drawdown_pct','trades')}
            row[w].update(return_gain_pp=a['return_pct']-baseline[w]['return_pct'],drawdown_gain_pp=a['drawdown_pct']-baseline[w]['drawdown_pct'],folder=str(folder))
        row['both_returns_improved']=all(row[w]['return_gain_pp']>=0 for w in ('train','oos'))
        row['eligible']=row['train']['return_gain_pp']>=-3 and row['oos']['return_gain_pp']>=2 and row['oos']['return_pct']>0 and all(row[w]['drawdown_gain_pp']<=2 for w in ('train','oos'))
        row['activity_preserved']=all(row[w]['trades']>=baseline[w]['trades'] for w in ('train','oos'))
        rows.append(row)
    n.write(s.OUT/'all_paired_results.json',rows)
    return rows


def rank(r):
    return (r['both_returns_improved'],min(r[w]['return_gain_pp'] for w in ('train','oos')),r['oos']['return_pct'],-r['oos']['drawdown_pct'])


def evaluate(row,window,scenario='combined',capped=False):
    root=Path(row['root']);cid=row['candidate_id']
    if row['family']=='legacy':
        old=results.locate(cid,window,scenario,capped)
        if old:
            s.forensic.authenticate(old);return n.read(old/'assessment.json')
    catalog={c['id']:c for c in n.read(root/'candidate_plan.json')['candidates']}
    value=s.mechanism.evaluate(root,catalog[cid],window,scenario=scenario,capped=capped)
    s.forensic.authenticate(Path(value['output']))
    return value


def main():
    assert (s.ROOT/'center_completion.json').exists()
    rows=gather();eligible=[r for r in rows if r['eligible']]
    preservation=max(eligible,key=rank);highest=max(eligible,key=lambda r:(r['oos']['return_pct'],r['train']['return_pct'],-r['oos']['drawdown_pct']))
    finalists=list({r['key']:r for r in (preservation,highest)}.values())
    baseline=next(r for r in rows if r['key']=='legacy:'+s.forensic.BASE)
    reg={'registered_utc':datetime.now(timezone.utc).isoformat(),'finalist_keys':[r['key'] for r in finalists],
        'selection':'IS-preservation ranking and highest OOS return among nominal gate passers; same ranking/gates for both feature families.',
        'all_pairs_sha256':n.sha(s.OUT/'all_paired_results.json'),'scenario':'combined','participation':.005,'olr_slippage_bps':15,
        'limits':{'is_return_max_drop_pp':3,'oos_return_gain_min_pp':2,'oos_return_positive':True,'drawdown_max_increase_pp':2},'promotion':False}
    p=s.ROOT/'cross_family_robustness_registration.json'
    if p.exists():assert n.read(p)['finalist_keys']==reg['finalist_keys']
    else:n.write(p,reg)
    stress=[]
    for row in [baseline,*finalists]:
        for w in ('train','oos'):
            print('Stress',row['key'],w,flush=True)
            result=evaluate(row,w);stress.append({**result,'key':row['key'],'family':row['family'],'root':row['root']})
            n.write(s.OUT/'matched_stress_results.json',stress)
            print({k:result.get(k) for k in ('return_pct','drawdown_pct','trades','qualified')},flush=True)
    control={r['window']:r for r in stress if r['key']==baseline['key']};accepted=[]
    for row in finalists:
        pair={r['window']:r for r in stress if r['key']==row['key']}
        row['stress_eligible']=(pair['train']['return_pct']-control['train']['return_pct']>=-3
            and pair['oos']['return_pct']>0 and pair['oos']['return_pct']-control['oos']['return_pct']>=2
            and all(pair[w]['drawdown_pct']-control[w]['drawdown_pct']<=2 for w in ('train','oos')))
        if row['stress_eligible']:accepted.append(row)
    selected=max(accepted,key=rank) if accepted else None
    n.write(s.OUT/'qualification_decision.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'finalists':finalists,'selected':selected,
        'nominal_and_stress_pass':selected is not None,'independent_validation':False,'promotion':False})
    if selected:
        capped=[]
        for row in (baseline,selected):
            for w in ('train','oos'):
                print('No upward compounding',row['key'],w,flush=True)
                result=evaluate(row,w,scenario='baseline',capped=True)
                capped.append({**result,'key':row['key'],'family':row['family'],'root':row['root']})
                n.write(s.OUT/'no_upward_compounding_results.json',capped)
        catalog={c['id']:c for c in n.read(Path(selected['root'])/'candidate_plan.json')['candidates']}
        n.write(s.OUT/'selected_research_policy.json',{'status':'research_only_not_live_promoted','selected':selected,
            'candidate':catalog[selected['candidate_id']],'history_policy':'Full IS intraday observations retained across April 1.',
            'current_afternoon_panel':'continuous prior observations' if selected['family']=='continuous' else 'Original same-session panel; previous-session rolling features fully continuous.',
            'selection_limit':'Former OOS consumed in development; later independent qualification still required.'})
    n.write(s.ROOT/'qualification_completion.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'selected_key':selected['key'] if selected else None,'contingency_required':selected is None})


if __name__=='__main__':main()

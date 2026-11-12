"""Authenticate paired results, select bounded finalists, and run robustness."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import olr_round1_optimization_v1 as s
n=s.n


def locate(cid,w,scenario='baseline',capped=False):
    suffix='100000000_'+scenario+('_capped' if capped else '')
    for root in (s.ROOT,s.PRIOR,s.PARENT):
        p=root/'cases'/cid/w/suffix
        if (p/'assessment.json').exists():return p
    return None


def pairs(authenticate=False):
    plan=n.read(s.ROOT/'candidate_plan.json');baseline=plan['baseline_id'];result=[]
    b={w:n.read(locate(baseline,w)/'assessment.json') for w in ('train','oos')}
    for c in plan['candidates']:
        row={'candidate_id':c['id'],'label':c['label']};qualified=True
        for w in ('train','oos'):
            folder=locate(c['id'],w)
            if folder is None:qualified=False;break
            a=n.read(folder/'assessment.json')
            if not a.get('qualified'):qualified=False;break
            if authenticate:s.forensic.authenticate(folder)
            row[w]={'return_pct':a['return_pct'],'drawdown_pct':a['drawdown_pct'],'trades':a['trades'],
                'return_gain_pp':a['return_pct']-b[w]['return_pct'],'drawdown_gain_pp':a['drawdown_pct']-b[w]['drawdown_pct'],'folder':str(folder)}
        if not qualified:continue
        row['both_returns_improved']=all(row[w]['return_gain_pp']>=0 for w in b)
        row['eligible']=(row['train']['return_gain_pp']>=-3 and row['oos']['return_gain_pp']>=2 and row['oos']['return_pct']>0
            and all(row[w]['drawdown_gain_pp']<=2 for w in b))
        row['activity_preserved']=all(row[w]['trades']>=b[w]['trades'] for w in b)
        result.append(row)
    n.write(s.OUT/'paired_results.json',result)
    return result


def robustness():
    assert n.read(s.ROOT/'grid_completion.json')['all_qualified']
    if (s.ROOT/'center_registration.json').exists():assert (s.ROOT/'center_completion.json').exists()
    rows=pairs(True);eligible=[r for r in rows if r['eligible']]
    assert eligible,'No eligible recorded successor'
    preservation=max(eligible,key=lambda r:(r['both_returns_improved'],min(r[w]['return_gain_pp'] for w in ('train','oos')),r['oos']['return_pct'],-r['oos']['drawdown_pct']))
    strongest_oos=max(eligible,key=lambda r:(r['oos']['return_pct'],r['train']['return_pct'],-r['oos']['drawdown_pct']))
    ids=list(dict.fromkeys([preservation['candidate_id'],strongest_oos['candidate_id']]))
    plan=n.read(s.ROOT/'candidate_plan.json');catalog={c['id']:c for c in plan['candidates']};base=plan['baseline_id']
    reg={'registered_utc':datetime.now(timezone.utc).isoformat(),'selected':ids,'baseline_id':base,
        'paired_results_sha256':n.sha(s.OUT/'paired_results.json'),'scenario':'combined','participation':.005,'slippage_bps':15,
        'limits':{'is_return_max_drop_pp':3,'positive_oos_return':True,'oos_return_gain_pp':2,'drawdown_max_increase_pp':2},
        'selection_rule':n.read(s.ROOT/'registration.json')['selection'],'promotion':False}
    rp=s.ROOT/'robustness_registration.json'
    if rp.exists():assert n.read(rp)['selected']==ids
    else:n.write(rp,reg)
    output=[]
    for cid in [base,*ids]:
        for w in ('train','oos'):
            cached=locate(cid,w,'combined')
            if cached:
                s.forensic.authenticate(cached);row=n.read(cached/'assessment.json')
            else:row=s.mechanism.evaluate(s.ROOT,catalog[cid],w,scenario='combined')
            s.forensic.authenticate(Path(row['output']));output.append(row)
            n.write(s.OUT/'stress_results.json',output)
            print({k:row.get(k) for k in ('candidate_id','window','return_pct','drawdown_pct','qualified')},flush=True)
    control={r['window']:r for r in output if r['candidate_id']==base}
    accepted=[]
    for cid in ids:
        r={x['window']:x for x in output if x['candidate_id']==cid}
        if (r['train']['return_pct']-control['train']['return_pct']>=-3 and r['oos']['return_pct']>0
            and r['oos']['return_pct']-control['oos']['return_pct']>=2
            and all(r[w]['drawdown_pct']-control[w]['drawdown_pct']<=2 for w in r)):accepted.append(cid)
    candidates=[r for r in rows if r['candidate_id'] in accepted]
    selected=max(candidates,key=lambda r:(r['both_returns_improved'],min(r[w]['return_gain_pp'] for w in ('train','oos')),r['oos']['return_pct']))['candidate_id'] if candidates else None
    n.write(s.ROOT/'robustness_completion.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'selected':selected,'stress_accepted':accepted,'promotion':False})
    if selected:
        capped=[]
        for cid in (base,selected):
            for w in ('train','oos'):
                old=locate(cid,w,capped=True)
                row=n.read(old/'assessment.json') if old else s.mechanism.evaluate(s.ROOT,catalog[cid],w,capped=True)
                s.forensic.authenticate(Path(row['output']));capped.append(row)
                n.write(s.OUT/'no_upward_compounding_results.json',capped)
                print({'no_upward_compounding':True,**{k:row.get(k) for k in ('candidate_id','window','return_pct','drawdown_pct','qualified')}},flush=True)
        n.write(s.ROOT/'compounding_completion.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'selected':selected,'native_replays':4})


def register_center():
    target=s.ROOT/'center_registration.json'
    if target.exists():return
    plan=n.read(s.ROOT/'candidate_plan.json');base=plan['baseline'];candidates=[]
    for floor in (-.001,-.002):
        m=copy.deepcopy(base);m[s.EXIT].update(failed_followthrough_bars=1,failed_followthrough_close_r=-1.,failed_followthrough_mfe_r=4.)
        m[s.ENTRY]['min_vwap_ret']=floor;key=n.sig(m)
        candidates.append({'id':'c'+key[:16],'label':f'guard_1.0_confirmation_{floor}','kind':'adaptive_factorial_center',
            'mutations':m,'mutations_hash':key,'aliases':[],'paths':[list(p) for p,_,_ in n.mutation_edits(base,m)]})
    n.write(target,{'registered_utc':datetime.now(timezone.utc).isoformat(),'candidates':candidates,
        'reason':'The audited 1.0R center improves both returns versus the earlier 1.2R guard. Complete the center cells of the same confirmation factorial; no new confirmation thresholds.',
        'consumed_guard_results':{w:n.sha(locate('c8d12020033107bb0',w)/'assessment.json') for w in ('train','oos')},
        'adaptive_development':True,'maximum_additional_candidates':2,'both_windows_required':True,'promotion':False})
    for name in ('candidate_plan.json','mechanism_plan.json'):
        p=n.read(s.ROOT/name);p['candidates'].extend(candidates);n.write(s.ROOT/name,p)


def center():
    assert (s.ROOT/'grid_completion.json').exists()
    candidates=n.read(s.ROOT/'center_registration.json')['candidates']
    s.mechanism.batch(s.ROOT,candidates,workers=2,phase='factorial_center')
    for c in candidates:
        for w in ('train','oos'):s.forensic.authenticate(locate(c['id'],w))
    n.write(s.ROOT/'center_completion.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'native_replays':4,'promotion':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['read','register-center','center','robustness']);a=p.parse_args()
    if a.action=='robustness':robustness()
    elif a.action=='register-center':register_center()
    elif a.action=='center':center()
    else:
        for r in pairs():print(r['label'],r['train']['return_pct'],r['oos']['return_pct'],r['eligible'])

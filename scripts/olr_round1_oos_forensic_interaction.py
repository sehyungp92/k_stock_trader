"""One adaptive, fully specified complementary-risk factorial follow-up."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import olr_round1_oos_forensic_replays as s
n=s.native


def main():
    assert n.read(s.ROOT/'completion.json')['full_native_audits']==6
    plan=n.read(s.ROOT/'candidate_plan.json'); baseline_id=plan['baseline_id']
    base={w:n.read(s.PARENT/'cases'/baseline_id/w/'100000000_baseline/assessment.json') for w in ('train','oos')}
    eligible=[]
    for cid in s.IDS:
        p={w:n.read(s.ROOT/'cases'/cid/w/'100000000_baseline/assessment.json') for w in base}
        gains={w:p[w]['return_pct']-base[w]['return_pct'] for w in base}
        if (all(p[w]['qualified'] for w in base) and gains['train']>=-3 and gains['oos']>=2
                and all(p[w]['trades']>=base[w]['trades'] and p[w]['drawdown_pct']<=base[w]['drawdown_pct']+2 for w in base)):
            eligible.append((min(gains.values()),gains['oos'],cid))
    if not eligible:
        n.write(s.ROOT/'interaction_disposition.json',{'no_eligible_guard':True});return
    chosen=max(eligible)[2]
    parentplan=n.read(s.PARENT/'mechanism_plan.json')
    known={c['id']:c for c in parentplan['candidates']}
    wider='cd72a95a0fa95c30f'
    mutations=copy.deepcopy(known[chosen]['mutations'])
    mutations['olr.trade_plan.entry']=copy.deepcopy(known[wider]['mutations']['olr.trade_plan.entry'])
    key=n.sig(mutations)
    candidate={'id':'c'+key[:16],'label':'wider_entry_extension_plus_'+known[chosen]['label'],
        'kind':'adaptive_mechanism_interaction','mutations_hash':key,'mutations':mutations,
        'reason':'Wider entry extension raises IS but harms OOS; selected guard contains OOS losses with bounded IS cost. Test their joint account outcome with all four factorial cells retained.',
        'aliases':[],'paths':[['olr.trade_plan.entry','max_vwap_extension_pct'],['olr.trade_plan.exit']]}
    registration={'registered_utc':datetime.now(timezone.utc).isoformat(),
        'candidate':candidate,'baseline_id':baseline_id,'wider_only_id':wider,'guard_only_id':chosen,
        'selection_rule':'Highest minimum paired return gain among the three guards satisfying inherited gates.',
        'consumed_results_sha256':n.sha(s.ROOT/'results.json'),
        'adaptive_development':True,'single_extra_configuration':True,'both_windows_required':True,
        'hypothesis':candidate['reason'],'acceptance':n.read(s.ROOT/'protocol.json')['selection'],
        'no_further_threshold_tuning_in_this_followup':True,'promotion':False}
    rp=s.ROOT/'interaction_registration.json'
    if rp.exists():assert n.read(rp)['candidate']==candidate
    else:
        n.write(rp,registration)
        plan['candidates'].append(candidate);n.write(s.ROOT/'candidate_plan.json',plan)
        m=n.read(s.ROOT/'mechanism_plan.json');m['candidates'].append(candidate);n.write(s.ROOT/'mechanism_plan.json',m)
    results=[]
    for w in ('train','oos'):
        print('Interaction',w,candidate['id'],flush=True)
        a=s.mechanism.evaluate(s.ROOT,candidate,w);results.append(a)
        n.write(s.ROOT/'interaction_results.json',results)
        print({k:a.get(k) for k in ('window','qualified','return_pct','drawdown_pct','trades')},flush=True)
        if not a.get('qualified'):raise RuntimeError('Unqualified interaction retained for diagnosis')
    n.write(s.ROOT/'interaction_completion.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),
        'candidate_id':candidate['id'],'full_native_audits':2,'promotion':False})


if __name__=='__main__':main()

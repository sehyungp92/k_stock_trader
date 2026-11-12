"""Matched combined-stress validation of the best eligible predeclared guard."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import olr_round1_oos_forensic_replays as study

n=study.native
ROOT=study.ROOT


def main():
    assert n.read(ROOT/'completion.json')['full_native_audits']==6
    base={w:n.read(study.PARENT/'cases'/n.read(ROOT/'candidate_plan.json')['baseline_id']/w/'100000000_baseline/assessment.json') for w in ('train','oos')}
    tested=list(study.IDS)
    if (ROOT/'interaction_completion.json').exists():
        tested.append(n.read(ROOT/'interaction_completion.json')['candidate_id'])
    pairs={cid:{w:n.read(ROOT/'cases'/cid/w/'100000000_baseline/assessment.json') for w in base} for cid in tested}
    eligible=[]
    for cid,p in pairs.items():
        gains={w:p[w]['return_pct']-base[w]['return_pct'] for w in base}
        if (all(p[w]['qualified'] for w in base) and gains['train']>=-3 and gains['oos']>=2
            and all(p[w]['trades']>=base[w]['trades'] and p[w]['drawdown_pct']<=base[w]['drawdown_pct']+2 for w in base)):
            eligible.append((min(gains.values()),gains['oos'],cid))
    if not eligible:
        n.write(ROOT/'stress_disposition.json',{'no_eligible_guard':True,'no_stress_selection':True})
        print('No eligible guard; no successor selected.',flush=True)
        return
    selected=max(eligible)[2]
    plan=n.read(ROOT/'candidate_plan.json');catalog={c['id']:c for c in plan['candidates']}
    stress_candidates=[selected]
    # The user prioritizes both-window return, whereas the inherited study also
    # imposes a 2pp drawdown-increase ceiling. Retain an economically improving
    # interaction as a separately labeled risk trade-off even if that ceiling
    # excludes it from the primary selection. Never change the original gates.
    if (ROOT/'interaction_completion.json').exists():
        interaction=n.read(ROOT/'interaction_completion.json')['candidate_id']
        if interaction!=selected and all(pairs[interaction][w]['return_pct']>=base[w]['return_pct'] for w in base):
            stress_candidates.append(interaction)
    registration={'registered_utc':datetime.now(timezone.utc).isoformat(),'selected':selected,
        'selection_rule':'Among predeclared guards and the one registered interaction passing all inherited gates, maximize minimum IS/OOS return gain; tie by OOS gain.',
        'base_results_sha256':n.sha(ROOT/'results.json'),
        'stress_candidates':stress_candidates,
        'secondary_scope':'Also stress the one interaction if it improves both baseline-window returns, even when it fails the inherited drawdown ceiling; preserve that failure label.',
        'scenario':'combined','participation':0.005,'olr_slippage_bps':15,
        'commission_bps':2,'tax_bps_on_sell':20,'windows':n.read(ROOT/'protocol.json')['windows'],
        'matched_control':plan['baseline_id'],'capital':100_000_000,
        'development_only':True,'promotion':False}
    if not (ROOT/'stress_registration.json').exists():n.write(ROOT/'stress_registration.json',registration)
    else:assert n.read(ROOT/'stress_registration.json')['selected']==selected
    rows=[]
    for cid in (plan['baseline_id'],*stress_candidates):
        for window in ('train','oos'):
            print('Combined stress',cid,window,flush=True)
            row=study.mechanism.evaluate(ROOT,catalog[cid],window,scenario='combined')
            rows.append(row);n.write(ROOT/'stress_results.json',rows)
            print({k:row.get(k) for k in ('candidate_id','window','qualified','return_pct','drawdown_pct','trades')},flush=True)
            if not row.get('qualified'):raise RuntimeError('Unqualified stress replay retained for review')
    n.write(ROOT/'stress_completion.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),
        'selected':selected,'stress_candidates':stress_candidates,'full_native_audits':len(rows),'promotion':False})
    n.write(ROOT.parent/'research_candidate.json',{
        'candidate':catalog[selected],'stress_registration':str(ROOT/'stress_registration.json'),
        'deployment_status':'research_only; no independent validation; no canonical configuration change'})


if __name__=='__main__':main()

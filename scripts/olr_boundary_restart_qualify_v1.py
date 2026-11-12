"""Bounded neighborhood, matched controls, and stress for restart-correct OLR."""
from __future__ import annotations
import argparse,copy,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import olr_boundary_restart_results_v1 as r
b=r.b;s=b.s;n=b.n


def rank(v):return (v['both_returns_improved'],min(v[w]['return_gain_pp'] for w in ('train','oos')),v['oos']['return_pct'],-v['oos']['drawdown_pct'])


def amend_feature_adoption():
    """Register a bounded fallback without relaxing any return or DD gate."""
    path=b.ROOT/'feature_adoption_amendment.json'
    if path.exists():return
    p=b.ROOT/'cases/c662122ea4134f37b/train/100000000_baseline'
    s.forensic.authenticate(p)
    n.write(path,{'registered_utc':s.utc(),'trigger_assessment_sha256':n.sha(p/'assessment.json'),
        'trigger':'Corrected current-panel + 1R IS return 50.8067% violates the original 3pp preservation limit; literal state-carry + 1R IS return 54.5355% preserves it.',
        'distinction':'Native market-history continuity is mandatory. Activation of a changed sector-volume scoring formula is a separate economic treatment.',
        'rule':'Prefer any fully corrected-panel policy that meets the original gates. If none does, permit the exact original scoring formula with verified literal state carry. Keep the feature repair available but disabled in the selected scoring version; label this explicitly.',
        'search_scope':'Only the already registered 0.8R and 1.2R neighborhood around the selected 1R center, with its same panel mode and history policy. No new entry, ranking, sizing or exit family.',
        'controls':'Use exact same panel mode and history policy for the unguarded comparator.',
        'return_drawdown_stress_gates_unchanged':True,'adaptive_development':True,'promotion':False})


def control_for(v,catalog):
    if v['history_policy']!='literal_carry':return next(c for c in catalog.values() if c['label']=='broad_declared_universe_full_panel')
    if v['mutations'].get(b.x.PANEL_KEY):return r.matched_control()
    return next(c for c in catalog.values() if c['label']=='literal_carry_original')


def register_protocol():
    p=b.ROOT/'qualification_protocol.json'
    if p.exists():return
    n.write(p,{'registered_utc':s.utc(),'required':'Continuous prior signal panel plus literal state carry or declared-universe broad history.',
        'gates':n.read(b.ROOT/'registration.json')['acceptance'],
        'neighborhood':'At most two additional guard thresholds, 0.8R and 1.2R, around the preferred eligible 1.0R history-complete policy. Same observation policy and all other parameters.',
        'matched_control':'Add unguarded full-panel literal-carry control if needed to attribute the preferred literal-carry successor.',
        'finalists':'At most two eligible policies: preservation ranking winner and highest OOS-return winner. Prefer both returns improved, then maximize the smaller return gain, then OOS return.',
        'stress':'Combined actual bar participation 0.5%, OLR slippage 15bp. Evaluate finalists and their unguarded history-matched controls; compare same return/DD gates against original audited stressed benchmark.',
        'compounding':'No upward compounding for final selection and its continuous unguarded reference, keeping native cash and fill constraints.',
        'independent_validation':False,'promotion':False})


def neighborhood():
    register_protocol();assert (b.ROOT/'primary_completion.json').exists()
    rows=r.pairs();eligible=[v for v in rows if v['eligible']]
    if not eligible:
        n.write(b.ROOT/'neighborhood_completion.json',{'eligible_center':None,'native_replays':0});return
    center=max(eligible,key=rank);plan=n.read(b.ROOT/'candidate_plan.json');catalog={c['id']:c for c in plan['candidates']}
    if center['history_policy']=='literal_carry' and center['mutations'].get(b.x.PANEL_KEY):
        control=r.matched_control()
        for w in ('train','oos'):b.evaluate(control,w)
    path=b.ROOT/'guard_neighborhood_registration.json'
    if path.exists():candidates=n.read(path)['candidates']
    else:
        candidates=[]
        for threshold in (.8,1.2):
            m=copy.deepcopy(center['mutations']);m[s.EXIT].update(failed_followthrough_bars=1,failed_followthrough_close_r=-threshold,failed_followthrough_mfe_r=4.)
            policy=center['history_policy'];key=n.sig(m)
            c={'id':'c'+n.sig({'mutations':m,'history_policy':policy})[:16],
                'label':policy+('_full_panel' if m.get(b.x.PANEL_KEY) else '_original_scoring')+'_guard_'+str(threshold),'history_policy':policy,'mutations':m,'mutations_hash':key,
                'aliases':[],'kind':'registered_guard_neighborhood','paths':[list(p) for p,_,_ in n.mutation_edits(plan['baseline'],m)]}
            candidates.append(c)
        n.write(path,{'registered_utc':s.utc(),'center':center['candidate_id'],'candidates':candidates,
            'consumed_pairs_sha256':n.sha(s.OUT/'boundary_restart_paired_results.json'),'adaptive_development':True,'promotion':False})
        for name in ('candidate_plan.json','mechanism_plan.json'):
            data=n.read(b.ROOT/name);data['candidates'].extend(candidates);n.write(b.ROOT/name,data)
    for c in candidates:
        for w in ('train','oos'):b.evaluate(c,w)
    r.pairs();n.write(b.ROOT/'neighborhood_completion.json',{'completed_utc':s.utc(),'center':center['candidate_id'],'native_replays':4})


def qualify():
    register_protocol();rows=r.pairs();eligible=[v for v in rows if v['eligible']]
    if not eligible:
        n.write(s.OUT/'boundary_restart_qualification.json',{'selected':None,'reason':'No history-complete nominal policy satisfies the preservation gates.','promotion':False});return
    plan=n.read(b.ROOT/'candidate_plan.json');catalog={c['id']:c for c in plan['candidates']}
    preservation=max(eligible,key=rank);highest=max(eligible,key=lambda v:(v['oos']['return_pct'],v['train']['return_pct']))
    finalists=list({v['candidate_id']:v for v in (preservation,highest)}.values())
    controls={}
    for v in finalists:
        controls[v['history_policy']]=control_for(v,catalog)
    catalog.update({c['id']:c for c in controls.values()})
    n.write(b.ROOT/'stress_finalist_registration.json',{'registered_utc':s.utc(),'finalists':[v['candidate_id'] for v in finalists],
        'controls':{k:c['id'] for k,c in controls.items()},'paired_results_sha256':n.sha(s.OUT/'boundary_restart_paired_results.json')})
    stress=[]
    for w in ('train','oos'):
        f=s.PRIOR/'cases'/s.forensic.BASE/w/'100000000_combined';s.forensic.authenticate(f)
        stress.append({**n.read(f/'assessment.json'),'comparison':'original'})
    ids=list(dict.fromkeys([*[c['id'] for c in controls.values()],*[v['candidate_id'] for v in finalists]]))
    for cid in ids:
        for w in ('train','oos'):
            a=b.evaluate(catalog[cid],w,scenario='combined');r.audit(catalog[cid],w,'combined')
            stress.append({**a,'comparison':'continuous_control' if cid in {c['id'] for c in controls.values()} else 'finalist'})
            n.write(s.OUT/'boundary_restart_stress_results.json',stress)
    original={a['window']:a for a in stress if a['comparison']=='original'};accepted=[]
    for v in finalists:
        pair={a['window']:a for a in stress if a['candidate_id']==v['candidate_id']}
        v['stress_eligible']=pair['train']['return_pct']-original['train']['return_pct']>=-3 and pair['oos']['return_pct']>0 and pair['oos']['return_pct']-original['oos']['return_pct']>=2 and all(pair[w]['drawdown_pct']-original[w]['drawdown_pct']<=2 for w in ('train','oos'))
        if v['stress_eligible']:accepted.append(v)
    selected=max(accepted,key=rank) if accepted else None
    n.write(s.OUT/'boundary_restart_qualification.json',{'completed_utc':s.utc(),'selected':selected,'finalists':finalists,'promotion':False,'independent_validation':False})
    if selected:
        capped=[];reference=controls[selected['history_policy']]
        for c in (reference,catalog[selected['candidate_id']]):
            for w in ('train','oos'):
                a=b.evaluate(c,w,capped=True);r.audit(c,w,capped=True);capped.append(a)
                n.write(s.OUT/'boundary_restart_no_upward_compounding.json',capped)
        n.write(s.OUT/'selected_research_policy.json',{'status':'research_only_not_live_promoted','candidate':catalog[selected['candidate_id']],
            'selection':selected,'continuous_reference':reference,'generation_root':str(b.ENGINES),'research_root':str(b.ROOT),
            'mutation_edits_from_original':n.mutation_edits(plan['baseline'],catalog[selected['candidate_id']]['mutations']),
            'original_acceptance_gates':n.read(b.ROOT/'registration.json')['acceptance'],
            'scoring_adoption_amendment':str(b.ROOT/'feature_adoption_amendment.json') if (b.ROOT/'feature_adoption_amendment.json').exists() else None,
            'source_hashes':n.read(b.ROOT/'preparation_complete.json')['source_hashes'],
            'builder':str(b.x.BUILDER),'builder_sha256':n.sha(b.x.BUILDER),
            'runtime_requirement':'Use the bound source generation and declared warm-start path. A bare exit-parameter edit does not implement history continuity.',
            'history_policy':selected['history_policy'],'current_sector_panel':('Continuous prior completed-session history' if selected['mutations'].get(b.x.PANEL_KEY) else 'Original scoring formula retained; corrected sector-volume scoring disabled after failing preservation gate'),
            'independent_validation':False,'promotion':False})
    n.write(b.ROOT/'qualification_completion.json',{'completed_utc':s.utc(),'selected':selected['candidate_id'] if selected else None,'promotion':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','amend-feature-adoption','neighborhood','qualify','all']);a=p.parse_args()
    if a.action=='all':neighborhood();qualify()
    else:{'register':register_protocol,'amend-feature-adoption':amend_feature_adoption,'neighborhood':neighborhood,'qualify':qualify}[a.action]()

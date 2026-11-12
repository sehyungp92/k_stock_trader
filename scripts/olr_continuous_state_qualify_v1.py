"""Select only history-complete policies and verify matched execution stress."""
from __future__ import annotations
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import olr_continuous_state_results_v1 as r
x=r.x;s=x.s;n=x.n


def rank(row):
    return (row['both_returns_improved'],min(row[w]['return_gain_pp'] for w in ('train','oos')),row['oos']['return_pct'],-row['oos']['drawdown_pct'])


def register():
    target=x.ROOT/'qualification_protocol.json'
    if target.exists():return
    n.write(target,{'registered_utc':s.utc(),'no_warm_native_outcomes_consumed':True,
        'required_history_flags':{'capacity_broad_auction_history':True,x.PANEL_KEY:True},
        'eligible':'Both periods native-qualified; IS return no more than 3pp below original; OOS positive and at least 2pp above original; neither maximum intraday drawdown more than 2pp worse.',
        'selection':'Prefer both returns improved, then maximize the smaller return gain, then OOS return, then lower OOS drawdown.',
        'finalists':'At most two: preservation ranking winner and highest OOS return among eligible history-complete policies.',
        'stress':'Combined 0.5% actual volume participation and 15bp slippage, versus original matched control and fully history-complete unguarded control. Same acceptance gates versus original.',
        'compounding':'Selected policy and original cold control under no upward compounding, to diagnose dependence on earlier capital gains relative to the original user benchmark.',
        'uncertainty':'Calendar subperiods, bootstrap, tail concentration, and local threshold neighbors are diagnostic; former OOS remains development data.',
        'promotion':False})


def run():
    register();rows=r.pairs();plan=n.read(x.ROOT/'candidate_plan.json');catalog={c['id']:c for c in plan['candidates']}
    eligible=[v for v in rows if v['eligible'] and v['mutation'].get('capacity_broad_auction_history') and v['mutation'].get(x.PANEL_KEY)]
    if not eligible:
        n.write(s.OUT/'continuous_state_qualification.json',{'completed_utc':s.utc(),'selected':None,'reason':'No history-complete nominal candidate satisfies original-return preservation gates.','promotion':False});return
    preservation=max(eligible,key=rank);highest=max(eligible,key=lambda v:(v['oos']['return_pct'],v['train']['return_pct']))
    finalists=list({v['candidate_id']:v for v in (preservation,highest)}.values())
    n.write(x.ROOT/'stress_finalist_registration.json',{'registered_utc':s.utc(),'candidate_ids':[v['candidate_id'] for v in finalists],
        'paired_results_sha256':n.sha(s.OUT/'continuous_state_paired_results.json'),'protocol_sha256':n.sha(x.ROOT/'qualification_protocol.json')})
    corrected=next(c for c in plan['candidates'] if c['label']=='both_histories_continuous');stress=[]
    # Original stressed results were independently completed before this study.
    for w in ('train','oos'):
        f=s.PRIOR/'cases'/s.forensic.BASE/w/'100000000_combined'
        if not (f/'assessment.json').exists():
            f=s.PARENT/'cases'/s.forensic.BASE/w/'100000000_combined'
        s.forensic.authenticate(f);a=n.read(f/'assessment.json');stress.append({**a,'comparison':'original'})
    for cid in dict.fromkeys([corrected['id'],*[v['candidate_id'] for v in finalists]]):
        for w in ('train','oos'):
            a=s.mechanism.evaluate(x.ROOT,catalog[cid],w,scenario='combined')
            assert a['qualified'],a
            r.audit(Path(a['output']),w,True);stress.append({**a,'comparison':'corrected_control' if cid==corrected['id'] else 'finalist'})
            n.write(s.OUT/'continuous_state_stress_results.json',stress)
            print({k:a[k] for k in ('candidate_id','window','return_pct','drawdown_pct','trades')},flush=True)
    original={v['window']:v for v in stress if v['comparison']=='original'};accepted=[]
    for v in finalists:
        pair={a['window']:a for a in stress if a['candidate_id']==v['candidate_id'] and a['comparison']!='original'}
        v['stress_eligible']=pair['train']['return_pct']-original['train']['return_pct']>=-3 and pair['oos']['return_pct']>0 and pair['oos']['return_pct']-original['oos']['return_pct']>=2 and all(pair[w]['drawdown_pct']-original[w]['drawdown_pct']<=2 for w in ('train','oos'))
        if v['stress_eligible']:accepted.append(v)
    selected=max(accepted,key=rank) if accepted else None
    decision={'completed_utc':s.utc(),'finalists':finalists,'selected':selected,'promotion':False,'independent_validation':False}
    n.write(s.OUT/'continuous_state_qualification.json',decision)
    if selected:
        capped=[]
        for cid in dict.fromkeys([plan['baseline_id'],selected['candidate_id']]):
            for w in ('train','oos'):
                a=s.mechanism.evaluate(x.ROOT,catalog[cid],w,capped=True)
                assert a['qualified'],a
                r.audit(Path(a['output']),w,catalog[cid]['mutations'].get('capacity_broad_auction_history',False));capped.append(a)
                n.write(s.OUT/'continuous_state_no_upward_compounding.json',capped)
                print({'capped':True,**{k:a[k] for k in ('candidate_id','window','return_pct','drawdown_pct','trades')}},flush=True)
        n.write(s.OUT/'selected_research_policy.json',{'status':'research_only_not_live_promoted','candidate':catalog[selected['candidate_id']],
            'selected':selected,'generation_root':str(x.ENGINES),'protocol':str(x.ROOT/'qualification_protocol.json'),
            'builder':str(x.BUILDER),'builder_sha256':n.sha(x.BUILDER),'source_hashes':n.read(x.ROOT/'preparation_complete.json')['source_hashes'],
            'required_runtime':'Use the bound continuous-state source generation and builder, preserve prior market history and native broad auction observations; copying the exit parameters alone does not implement this policy.',
            'history_validation':{w:n.sha(x.ROOT/f'boundary_validation_{w}.json') for w in ('train','oos')},
            'independent_validation':False})
    n.write(x.ROOT/'qualification_completion.json',{'completed_utc':s.utc(),'selected':selected['candidate_id'] if selected else None,'promotion':False})


if __name__=='__main__':register() if len(sys.argv)>1 and sys.argv[1]=='register' else run()

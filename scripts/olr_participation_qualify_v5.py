"""Serial two-policy, three-scenario native qualification with unweakened gates."""
from collections import defaultdict
import gzip,json,traceback
from pathlib import Path
from scripts import olr_participation_study_v5 as t
from scripts import olr_participation_prepare_v5 as preparation
from scripts import olr_opening_risk_qualify_v2 as engine
from scripts.olr_repair_qualify_v4 import frequency_economics
n,s,b=t.n,t.s,t.b
_audit=engine.audit;_gates=engine.gates
WINDOWS=('train','oos')


def references():
    p=n.read(t.previous.OUT/'selected_research_policy.json')
    assert p['results_sha256']==n.sha(t.previous.OUT/'results.json')
    return n.read(t.previous.OUT/'results.json')['references']


def envelope_audit(folder):
    cohorts=defaultdict(list); activity=[]
    with gzip.open(folder/'native_decisions.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            event=json.loads(line);m=event.get('metadata',{})
            if event['decision_code']=='ENTRY_SUBMITTED' and m.get('empty_participation_rescue_v5'):
                p=m['participation_envelope_v5'];q=int(m['qty']);unit=float(p['unit_cost'])
                assert q>0 and q<=p['clipped_proposed_qty']<=p['native_proposed_qty']
                assert q*unit<=p['name_limit']+1e-7 and abs(p['name_limit']-.05*p['equity'])<1e-6
                assert abs(p['cohort_limit']-.10*p['equity'])<1e-6
                assert abs(p['remaining']-max(0,p['cohort_limit']-p['held_marked']-p['pending_reserved']))<1e-6
                assert m['allocation_redistributed_qty']==0
                assert abs(m['reserved_notional']-q*unit)<1e-6
                assert m['scenario_loss_budget']['approved_loss']<=.04*m['scenario_loss_budget']['equity']+1e-5
                cohorts[event['timestamp']].append((q*unit,p))
                activity.append(dict(timestamp=event['timestamp'],symbol=event['symbol'],qty=q,reserved=q*unit,equity=p['equity']))
    for stamp,rows in cohorts.items():
        p=rows[0][1]
        assert all(all(r[k]==p[k] for k in ('equity','held_marked','pending_reserved','remaining')) for _,r in rows)
        assert sum(v for v,_ in rows)<=p['remaining']+1e-5,(stamp,rows)
    trades=n.read(folder/'trades.json')
    tagged=[x for x in trades if x['route_metadata'].get('empty_participation_rescue_v5')]
    for x in tagged:
        m=x['route_metadata'];p=m['participation_envelope_v5']
        assert x['qty']<=m['allocation_native_qty']<=p['clipped_proposed_qty']
    return dict(passed=True,submitted_orders=len(activity),cohorts=len(cohorts),filled_lifecycles=len(tagged),
        filled_entry_notional=sum(x['qty']*x['entry_price'] for x in tagged),net_pnl=sum(x['net_pnl'] for x in tagged),
        max_realized_entry_nav_fraction=max((x['qty']*x['entry_price']/x['route_metadata']['participation_envelope_v5']['equity'] for x in tagged),default=0),
        decision_admission_limits=True,not_a_continuous_mark_to_market_cap=True,activity=activity)


def audit(c,w,scenario='baseline',capped=False):
    proof=_audit(c,w,scenario,capped)
    proof['participation_envelope']=envelope_audit(b.output(c,w,scenario,capped))
    if not c['mutations'][t.FLAG]: assert proof['participation_envelope']['submitted_orders']==0
    return proof


def gates(a,z,control,refs,stress=False):
    tests=_gates(a,z,control,refs,stress)
    if stress:
        tests.update(is_frequency=a['trades']>=control['stress']['train']['trades'],
            oos_frequency=z['trades']>=control['stress']['oos']['trades'],
            is_entry_days=a['entry_days']>=control['stress']['train']['entry_days'],
            oos_entry_days=z['entry_days']>=control['stress']['oos']['entry_days'])
    else:
        g=n.read(t.ROOT/'registration.json')['gates'];e=frequency_economics(z,control['windows']['oos'])
        tests.update(is_entry_days=a['entry_days']>=g['is_entry_days_min'],oos_entry_days=z['entry_days']>=g['oos_entry_days_min'],
            oos_meaningful_added_notional=e['net_incremental_notional']>=1e6 and e['median_added_notional']>=1e6)
    return tests


def configure():
    engine.t=t;engine.preparation=preparation;engine.audit=audit;engine.gates=gates;engine.reference_data=references;t.activate()


def collect():
    configure();data=engine.collect();control=data['candidates'][0]
    for row in data['candidates']:
        if len(row['windows'])==len(control['windows'])==2:
            row['frequency_economics']={w:frequency_economics(row['windows'][w],control['windows'][w]) for w in WINDOWS}
        if len(row['capped'])==len(control['capped'])==2:
            a,z=row['capped']['train'],row['capped']['oos'];ca,cz=control['capped']['train'],control['capped']['oos']
            row['capped_gates']=dict(is_return=a['return_pct']>=ca['return_pct']-1.5,
                oos_return=z['return_pct']>=cz['return_pct'],oos_drawdown=z['drawdown_pct']<cz['drawdown_pct'],
                is_frequency=a['trades']>=ca['trades'],oos_frequency=z['trades']>=cz['trades'],
                is_entry_days=a['entry_days']>=ca['entry_days'],oos_entry_days=z['entry_days']>=cz['entry_days'])
            row['capped_qualified']=all(row['capped_gates'].values())
    return data


def check_preflight():
    receipt=n.read(t.OUT/'preflight_receipt.json');tests=n.read(t.OUT/'test_receipt.json')
    assert receipt['passed'] and receipt['registration_sha256']==n.sha(t.ROOT/'registration.json')
    assert receipt['sources']==n.read(t.ROOT/'preparation_complete.json')['sources']
    assert receipt['script_sha256']==n.sha(t.REPO/'scripts/olr_participation_preflight_v5.py')
    assert tests['passed']>=67 and tests['failures']==tests['skipped']==tests['exit_code']==0
    for name,digest in {**tests['test_hashes'],**tests['frozen_sources']}.items():
        assert n.sha(t.REPO/name)==digest
    for name,key in [('focused_tests.log','log_sha256'),('focused_tests.xml','junit_sha256')]:
        assert n.sha(t.OUT/name)==tests[key]
    for row in receipt['evidence']:
        assert row['incumbent_membership_rank_size_inputs_and_features_exact']
        assert n.sha(row['candidate_snapshot_path'])==row['candidate_snapshot_sha256']


def qualify():
    configure();check_preflight()
    harness=n.read(t.OUT/'research_harness.json')
    for name,digest in harness['files'].items(): assert n.sha(t.REPO/name)==digest
    catalog=n.read(t.ROOT/'candidate_plan.json')['candidates']
    assert len(catalog)==2
    for scenario,capped in [('baseline',False),('combined',False),('baseline',True)]:
        for c in catalog:
            for w in WINDOWS:engine.evaluate(c,w,scenario,capped)
    data=collect();qualified=[r for r in data['candidates'][1:] if all(r.get(k,False) for k in ('nominal_qualified','stress_qualified','capped_qualified'))]
    selected=qualified[0] if qualified else data['candidates'][0]
    n.write(t.OUT/'results.json',data)
    profile=dict(status='qualified_research_successor' if qualified else 'retain_selected_no_qualified_upgrade',
        selected_id=selected['id'],selected_label=selected['label'],mutations=selected['mutations'],
        results_sha256=n.sha(t.OUT/'results.json'),registration_sha256=n.sha(t.ROOT/'registration.json'),
        source_generation_hashes=n.read(t.ROOT/'preparation_complete.json')['sources'],history_policy='literal_carry',promotion=False,independent_validation=False)
    n.write(t.OUT/'selected_research_policy.json',profile)
    n.write(t.OUT/'qualified_alternatives.json',dict(policies=qualified))
    n.write(t.ROOT/'qualification_complete.json',dict(utc=s.utc(),selected_id=selected['id'],results_sha256=profile['results_sha256'],profile_sha256=n.sha(t.OUT/'selected_research_policy.json')))
    print(profile,flush=True)


if __name__=='__main__':
    try:qualify()
    except Exception:
        n.write(t.ROOT/'qualification_failure.json',dict(utc=s.utc(),traceback=traceback.format_exc()));raise

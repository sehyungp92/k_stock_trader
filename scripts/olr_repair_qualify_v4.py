"""Serial native qualification; role-specific component tests precede promotion."""
import copy,gzip,json,traceback
from pathlib import Path
from statistics import median
from scripts import olr_repair_study_v4 as t
from scripts import olr_repair_prepare_v4 as preparation
from scripts import olr_opening_risk_qualify_v2 as engine
n,s,b=t.n,t.s,t.b
_audit=engine.audit;_gates=engine.gates
WINDOWS=('train','oos')

def references():
    profile=n.read(t.previous.OUT/'selected_research_policy.json')
    assert profile['results_sha256']==n.sha(t.previous.OUT/'results.json')
    return n.read(t.previous.OUT/'results.json')['references']

def audit(c,w,scenario='baseline',capped=False):
    proof=_audit(c,w,scenario,capped);folder=b.output(c,w,scenario,capped)
    assert (folder/'native_decisions.jsonl.gz').is_file()
    if c['mutations']['capacity_liquidation_tail']:
        receipt=n.read(folder/'liquidation_restart_receipt.json')
        assert receipt['strictly_prior_sessions_only'] and receipt['independent_source_prefix_equal']
        if w=='oos':
            prior=n.read(b.output(c,'train',scenario,capped)/'capacity_terminal_state.json')
            assert receipt['actual_is_terminal_restored'] and receipt['through']=='2026-03-31'
            assert receipt['state_hash']==n.sig(prior['liquidation_tail_history_v3'])
        proof['liquidation_restart']=receipt
        for trade in n.read(folder/'trades.json'):
            m=trade['route_metadata'];ev=m['liquidation_tail_evidence'];q=m['allocation_before_redistribution']+m['allocation_redistributed_qty']
            assert ev['through']<trade['entry_decision_time'][:10]
            assert abs(q*m['estimated_entry_price']-ev['notional_upper'])<1e-5
            r=m['risk_repair_evidence'];assert r['total_quote_quantity']==q
            assert m['scenario_loss_budget']['approved_loss']<=m['scenario_loss_budget']['equity']*.04+1e-5
            if c['mutations'][t.RISK]=='separated':
                assert ev['fraction']>=ev['symbol_tail_mean'] and ev['peer_names']>=3 and ev['peer_dates']>=20
            if c['mutations'][t.RISK]=='portfolio':
                joint=r['joint_stress'];assert joint['supported'] and joint['common_dates']>=20
                assert joint['total_loss']<=r['equity']*.08+1e-5
                assert all(v<=r['equity']*.06+1e-5 for v in joint['sector_losses'].values())
    return proof

def frequency_economics(a,control):
    key=lambda x:(x['symbol'],x['entry_decision_time'])
    old={key(x):x for x in n.read(Path(control['folder'])/'trades.json')}
    new={key(x):x for x in n.read(Path(a['folder'])/'trades.json')}
    added=[x for k,x in new.items() if k not in old];removed=[x for k,x in old.items() if k not in new]
    notionals=[x['qty']*x['entry_price'] for x in added]
    return dict(added=len(added),removed=len(removed),added_notional=sum(notionals),
        net_incremental_notional=sum(notionals)-sum(x['qty']*x['entry_price'] for x in removed),
        median_added_notional=median(notionals) if notionals else 0.,added_net_pnl=sum(x['net_pnl'] for x in added))

def gates(a,z,control,refs,stress=False):
    out=_gates(a,z,control,refs,stress)
    if stress:
        out.update(is_frequency=a['trades']>=control['stress']['train']['trades'],oos_frequency=z['trades']>=control['stress']['oos']['trades'])
    else:
        g=n.read(t.ROOT/'registration.json')['gates'];e=frequency_economics(z,control['windows']['oos'])
        out.update(is_entry_days=a['entry_days']>=g['is_entry_days_min'],oos_entry_days=z['entry_days']>=g['oos_entry_days_min'],
            oos_meaningful_added_notional=e['net_incremental_notional']>=1e6 and e['median_added_notional']>=1e6)
    return out

def configure():
    engine.t=t;engine.preparation=preparation;engine.audit=audit;engine.gates=gates;engine.reference_data=references;t.activate()

def collect():
    configure();data=engine.collect()
    if len(data['candidates'][0]['windows'])==2:
        for r in data['candidates']:
            if len(r['windows'])==2:r['frequency_economics']={w:frequency_economics(r['windows'][w],data['candidates'][0]['windows'][w]) for w in WINDOWS}
    return data

def rank(row):return engine.rank(row)
def preserved(row):return all(row['gates'][k] for k in ('is_original_return','is_guard_return','is_selected_return','is_guard_drawdown'))
def active(row,control):
    keys=('symbol','qty','entry_decision_time','entry_fill_time','entry_price','exit_fill_time','exit_price','net_pnl')
    econ=lambda p:[{k:x[k] for k in keys} for x in n.read(Path(p)/'trades.json')]
    return any(econ(row['windows'][w]['folder'])!=econ(control['windows'][w]['folder']) for w in WINDOWS)
def risk_rank(row,control):return (control['windows']['oos']['drawdown_pct']-row['windows']['oos']['drawdown_pct'],row['windows']['oos']['return_pct'],row['windows']['train']['return_pct'])

def combination(data):
    bylabel={r['label']:r for r in data['candidates']};control=bylabel['selected_control'];parents=[]
    structural=bylabel['structural_partial_half']
    if preserved(structural):parents.append(structural)
    risks=[r for r in data['candidates'] if r['mutations'][t.RISK]!='off' and preserved(r) and active(r,control)]
    if risks:parents.append(max(risks,key=lambda r:risk_rank(r,control)))
    rescue=bylabel['empty_afternoon_flow_rescue']
    if preserved(rescue) and rescue['windows']['oos']['trades']>control['windows']['oos']['trades']:parents.append(rescue)
    if len(parents)<2:parents=[]
    ids=[r['id'] for r in parents];receipt=t.ROOT/'combination_selection.json'
    if receipt.exists():assert n.read(receipt)['parents']==ids
    else:n.write(receipt,dict(utc=s.utc(),parents=ids,consumed_nominal_results_hash=n.sig(data),rule=n.read(t.ROOT/'registration.json')['gates']['combination']))
    if not parents:return None
    plan=n.read(t.ROOT/'candidate_plan.json');changes={}
    for row in parents:
        for path,value,deleted in n.mutation_edits(plan['baseline'],row['mutations']):
            assert len(path)==1 and not deleted
            if path[0] in changes:assert changes[path[0]]==value
            changes[path[0]]=value
    c=t.candidate('registered_component_union',plan['baseline'],changes)
    if c['id'] not in {x['id'] for x in plan['candidates']}:
        for filename in ('candidate_plan.json','mechanism_plan.json'):
            p=n.read(t.ROOT/filename);p['candidates'].append(c);n.write(t.ROOT/filename,p)
    return c

def check_tests():
    receipt=n.read(t.OUT/'test_receipt.json')
    assert receipt['passed']>=67 and receipt['failures']==receipt['skipped']==receipt['exit_code']==0
    assert len(receipt['frozen_sources'])==2*len(t.PATCHED)
    for name,digest in receipt['test_hashes'].items():assert n.sha(t.REPO/name)==digest
    for name,digest in receipt['frozen_sources'].items():assert n.sha(Path(name))==digest
    assert n.sha(t.OUT/'focused_tests.log')==receipt['log_sha256']
    assert n.sha(t.OUT/'focused_tests.xml')==receipt['junit_sha256']

def check_preflight():
    receipt=n.read(t.OUT/'preflight_receipt.json')
    assert receipt['passed']
    assert receipt['registration_sha256']==n.sha(t.ROOT/'registration.json')
    assert receipt['sources']==n.read(t.ROOT/'preparation_complete.json')['sources']
    assert receipt['script_sha256']==n.sha(t.REPO/'scripts/olr_repair_preflight_v4.py')
    assert {r['window'] for r in receipt['evidence']}==set(WINDOWS)
    for record in receipt['evidence']:
        assert record['incumbent_membership_rank_size_inputs_and_features_exact']
        assert n.sha(Path(record['candidate_snapshot_path']))==record['candidate_snapshot_sha256']

def qualify():
    configure();preparation.prepare();check_tests()
    check_preflight()
    catalog=n.read(t.ROOT/'candidate_plan.json')['candidates'];control=catalog[0];structural=catalog[1]
    # Resolve the missing robustness evidence first, without a full-frequency
    # requirement on a risk-only component.
    for c in (control,structural):
        for scenario,capped in [('baseline',False),('combined',False),('baseline',True)]:
            for w in WINDOWS:engine.evaluate(c,w,scenario,capped)
    for c in catalog[2:6]:
        for w in WINDOWS:engine.evaluate(c,w)
    data=collect();combo=combination(data)
    if combo:
        for w in WINDOWS:engine.evaluate(combo,w)
        data=collect()
    risks=sorted((r for r in data['candidates'][2:6] if r['mutations'][t.RISK]!='off' and preserved(r) and active(r,data['candidates'][0])),key=lambda r:risk_rank(r,data['candidates'][0]),reverse=True)[:2]
    eligible=[r for r in data['candidates'][1:] if r['nominal_qualified']]
    leaders=[max(eligible,key=rank),max(eligible,key=lambda r:(engine.frequency(r),rank(r)))] if eligible else []
    ids=list(dict.fromkeys([r['id'] for r in [*risks,*leaders]]))
    receipt=t.ROOT/'stress_selection.json'
    if receipt.exists():assert n.read(receipt)['additional_candidates']==ids
    else:n.write(receipt,dict(utc=s.utc(),always=[control['id'],structural['id']],additional_candidates=ids,consumed_results_hash=n.sig(data),rule=n.read(t.ROOT/'registration.json')['gates']['qualification']))
    catalog={c['id']:c for c in n.read(t.ROOT/'candidate_plan.json')['candidates']}
    for cid in ids:
        for w in WINDOWS:engine.evaluate(catalog[cid],w,'combined')
    data=collect();qualified=[r for r in data['candidates'][1:] if r.get('nominal_qualified') and r.get('stress_qualified')]
    selected=max(qualified,key=rank) if qualified else data['candidates'][0]
    for w in WINDOWS:engine.evaluate(catalog[selected['id']],w,'baseline',True)
    data=collect();n.write(t.OUT/'results.json',data)
    selected=next(r for r in data['candidates'] if r['id']==selected['id'])
    profile=dict(status='qualified_research_successor' if qualified else 'retain_selected_no_qualified_upgrade',
        selected_id=selected['id'],selected_label=selected['label'],mutations=catalog[selected['id']]['mutations'],
        results_sha256=n.sha(t.OUT/'results.json'),registration_sha256=n.sha(t.ROOT/'registration.json'),
        source_generation_hashes=n.read(t.ROOT/'preparation_complete.json')['sources'],history_policy='literal_carry',promotion=False,independent_validation=False)
    n.write(t.OUT/'selected_research_policy.json',profile)
    n.write(t.OUT/'qualified_alternatives.json',dict(policies=[dict(candidate_id=r['id'],label=r['label'],mutations=r['mutations'],nominal=r['windows'],stress=r['stress']) for r in qualified]))
    n.write(t.ROOT/'qualification_complete.json',dict(utc=s.utc(),selected_id=selected['id'],results_sha256=profile['results_sha256'],profile_sha256=n.sha(t.OUT/'selected_research_policy.json')))
    print(profile,flush=True)

if __name__=='__main__':
    try:qualify()
    except Exception:
        n.write(t.ROOT/'qualification_failure.json',dict(utc=s.utc(),traceback=traceback.format_exc()));raise

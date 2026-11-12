"""Recheck native evidence and delivery without rewriting selected results."""
import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse,unquote
from scripts import olr_repair_study_v4 as t
from scripts import olr_repair_qualify_v4 as q
n,s,b=t.n,t.s,t.b


def verify(chart_reviewed=False):
    q.configure();q.check_tests();q.check_preflight()
    data=n.read(t.OUT/'results.json');profile=n.read(t.OUT/'selected_research_policy.json')
    assert profile['results_sha256']==n.sha(t.OUT/'results.json')
    assert profile['registration_sha256']==data['registration_sha256']==n.sha(t.ROOT/'registration.json')
    assert n.sha(t.OUT/'registration.json')==profile['registration_sha256']
    done=n.read(t.ROOT/'qualification_complete.json')
    assert done['results_sha256']==profile['results_sha256']
    assert done['profile_sha256']==n.sha(t.OUT/'selected_research_policy.json')
    complete=n.read(t.OUT/'completion.json')
    for name,digest in complete['files'].items():assert n.sha(t.OUT/name)==digest,name
    assert not profile['promotion'] and not profile['independent_validation']
    assert not data['promotion'] and not data['independent_validation']
    plan=n.read(t.ROOT/'candidate_plan.json');catalog={c['id']:c for c in plan['candidates']}
    reg=n.read(t.ROOT/'registration.json')
    assert plan['candidates'][:6]==reg['candidates']
    assert len(catalog) in (6,7) and len(data['candidates'])==len(catalog)
    assert [r['id'] for r in data['candidates']]==list(catalog)
    control=data['candidates'][0];structural=data['candidates'][1]
    selected=next(r for r in data['candidates'] if r['id']==profile['selected_id'])
    assert profile['mutations']==catalog[selected['id']]['mutations']
    assert all(not control['mutations'][k] for k in ('capacity_liquidation_tail','capacity_structural_partial','capacity_late_confirmation',t.FLOW))
    assert control['mutations'][t.RISK]=='off'
    proofs=[]
    for row in data['candidates']:
        assert row['mutations']==catalog[row['id']]['mutations']
        assert len(row['windows'])==2
        assert row['gates']==q.gates(row['windows']['train'],row['windows']['oos'],control,data['references'])
        assert row['nominal_qualified']==all(row['gates'].values())
        for w in q.WINDOWS:
            assert row['frequency_economics'][w]==q.frequency_economics(row['windows'][w],control['windows'][w])
        if row['stress']:
            assert row['stress_gates']==q.gates(row['stress']['train'],row['stress']['oos'],control,data['references'],True)
            assert row['stress_qualified']==all(row['stress_gates'].values())
        for group,scenario,capped in [('windows','baseline',False),('stress','combined',False),('capped','baseline',True)]:
            assert len(row[group]) in (0,2)
            for w,summary in row[group].items():
                proof=q.audit(catalog[row['id']],w,scenario,capped)
                assert proof['completion_sha256']==summary['completion_sha256']
                assert proof['source_generation_hash']==profile['source_generation_hashes'][w]
                assert summary['daily_dates'][0]==('2025-08-07' if w=='train' else '2026-04-01')
                assert summary['daily_dates'][-1]==('2026-03-31' if w=='train' else '2026-07-10')
                assert summary['sessions']==(156 if w=='train' else 69)
                if w=='oos':assert proof['literal_is_terminal_state_restored']
                proofs.append(dict(candidate=row['id'],window=w,scenario=scenario,capped=capped,completion_sha256=proof['completion_sha256']))
    qualified=[r for r in data['candidates'][1:] if r.get('nominal_qualified') and r.get('stress_qualified')]
    if qualified:
        assert selected['id']==max(qualified,key=q.rank)['id']
        assert profile['status']=='qualified_research_successor'
    else:assert selected['id']==control['id'] and profile['status']=='retain_selected_no_qualified_upgrade'
    assert len(control['stress'])==len(control['capped'])==len(structural['stress'])==len(structural['capped'])==len(selected['capped'])==2
    alternatives=n.read(t.OUT/'qualified_alternatives.json')['policies']
    assert [r['candidate_id'] for r in alternatives]==[r['id'] for r in qualified]
    for a,r in zip(alternatives,qualified):
        assert a['mutations']==r['mutations'] and a['nominal']==r['windows'] and a['stress']==r['stress']
    # Reproduce the registered component and stress-selection rules read-only.
    parents=[]
    if q.preserved(structural):parents.append(structural)
    risks=[r for r in data['candidates'][2:6] if r['mutations'][t.RISK]!='off' and q.preserved(r) and q.active(r,control)]
    if risks:parents.append(max(risks,key=lambda r:q.risk_rank(r,control)))
    rescue=data['candidates'][5]
    if q.preserved(rescue) and rescue['windows']['oos']['trades']>control['windows']['oos']['trades']:parents.append(rescue)
    if len(parents)<2:parents=[]
    assert n.read(t.ROOT/'combination_selection.json')['parents']==[r['id'] for r in parents]
    if parents:
        edits={}
        for r in parents:
            for path,value,deleted in n.mutation_edits(plan['baseline'],r['mutations']):
                assert len(path)==1 and not deleted
                if path[0] in edits:assert edits[path[0]]==value
                edits[path[0]]=value
        assert plan['candidates'][6]==t.candidate('registered_component_union',plan['baseline'],edits)
    else:assert len(catalog)==6
    components=sorted(risks,key=lambda r:q.risk_rank(r,control),reverse=True)[:2]
    eligible=[r for r in data['candidates'][1:] if r['nominal_qualified']]
    leaders=[max(eligible,key=q.rank),max(eligible,key=lambda r:(q.engine.frequency(r),q.rank(r)))] if eligible else []
    extra_ids=list(dict.fromkeys(r['id'] for r in [*components,*leaders]))
    stress=n.read(t.ROOT/'stress_selection.json')
    assert stress['always']==[control['id'],structural['id']] and stress['additional_candidates']==extra_ids
    assert {r['id'] for r in data['candidates'] if r['stress']}=={control['id'],structural['id'],*extra_ids}
    for w in q.WINDOWS:
        for scenario,capped in [('baseline',False),('combined',False),('baseline',True)]:
            parity=n.read(t.ROOT/f'control_parity_{w}_{scenario}_{capped}.json')
            assert parity['passed'] and parity['daily_and_intraday_marks_exact']
            assert n.sha(Path(parity['parent'])/'completion.json')==parity['parent_completion']
            assert n.sha(Path(parity['new'])/'completion.json')==parity['new_completion']
    prepared=n.read(t.ROOT/'preparation_complete.json')
    assert prepared['inherited_causal_panels_unchanged'] and prepared['sources']==profile['source_generation_hashes']
    for record in prepared['changes']:
        w=record['window'];ready=n.read(t.ENGINES/w/'ready.json');study=ready['study']
        parent=n.read(t.PARENT/'runtime'/t.REFERENCE/w/'baseline/study.json')
        assert record['liquidation_panel_sha256']==n.sha(Path(study['code'])/study['liquidation_observations'])==n.sha(Path(parent['code'])/parent['liquidation_observations'])
    for record in n.read(t.OUT/'implementation_receipt.json')['changed_sources']:
        w,name=record['window'],record['name']
        assert n.sha(t.ENGINES/w/'inputs/files'/name)==record['after']
        assert n.sha(t.OUT/'review_bundle'/w/name)==record['after']
    assert reg['reference_profile_sha256']==n.sha(t.previous.OUT/'selected_research_policy.json')
    assert reg['reference_results_sha256']==n.sha(t.previous.OUT/'results.json')
    assert data['references']==q.references()
    rescue=n.read(t.OUT/'rescue_signal_review.json')
    for w,source in rescue['sources'].items():
        runtime=t.ROOT/f'runtime/c66727c62cde50425/{w}/baseline/study.json';study=n.read(runtime)
        assert source['runtime_sha256']==n.sha(runtime)
        assert source['completion_sha256']==n.sha(t.ROOT/f'cases/c66727c62cde50425/{w}/100000000_baseline/completion.json')
        assert source['snapshot_sha256']==n.sha(Path(study['code'])/study['olr_snapshots'])
        assert source['execution_sha256']==n.sha(Path(study['code'])/'strategy_olr/execution.py')
        assert source['manifest_sha256']==n.sha(study['manifest'])
    assert {(r['window'],r['date'],r['symbol']) for r in rescue['rows']}=={
        (r['window'],x['date'],x['symbol']) for r in n.read(t.OUT/'preflight_receipt.json')['evidence'] for x in r['new_candidates']}
    for r in rescue['rows']:
        trades=n.read(t.ROOT/f"cases/c66727c62cde50425/{r['window']}/100000000_baseline/trades.json")
        actual=[x for x in trades if x['symbol']==r['symbol'] and x['entry_decision_time'][:10]==r['date']]
        assert len(actual)==r['actual_entries']
        assert sum(x['qty']*x['entry_price'] for x in actual)==r['actual_notional']
        assert sum(x['net_pnl'] for x in actual)==r['actual_net_pnl']
        if actual:assert actual[0]['entry_decision_time']==r['six_bar_signal_time']
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.local=[]
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('src','href') and value and not urlparse(value).scheme:self.local.append(unquote(urlparse(value).path))
    links=Links();links.feed((t.OUT/'report.html').read_text(encoding='utf-8'))
    assert links.local
    for name in links.local:assert (t.OUT/name).is_file(),name
    from PIL import Image
    with Image.open(t.OUT/'results.png') as im:dimensions=list(im.size);im.verify()
    output=dict(passed=bool(chart_reviewed),automated_checks_passed=True,verified_utc=s.utc(),native_windows=len(proofs),
        native_proofs=proofs,focused_tests_passed=n.read(t.OUT/'test_receipt.json')['passed'],chart_visually_reviewed=bool(chart_reviewed),
        chart_dimensions=dimensions,chart_sha256=n.sha(t.OUT/'results.png'),local_links=links.local,
        policy_sha256=n.sha(t.OUT/'selected_research_policy.json'),completion_sha256=n.sha(t.OUT/'completion.json'),promotion=False,independent_validation=False)
    n.write(t.OUT/'delivery_checks.json',output)
    print({k:output[k] for k in ('passed','automated_checks_passed','native_windows','focused_tests_passed','chart_visually_reviewed')})
    assert chart_reviewed,'Automated checks complete; visual review still required before final delivery'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--chart-reviewed',action='store_true')
    verify(parser.parse_args().chart_reviewed)

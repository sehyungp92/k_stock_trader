"""Recheck qualification, native histories and delivery without rewriting results."""
import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse,unquote
from scripts import olr_precision_study_v3 as t
from scripts import olr_precision_qualify_v3 as q
n,s,b=t.n,t.s,t.b


def verify(chart_reviewed=False):
    q.configure();data=n.read(t.OUT/'results.json');profile=n.read(t.OUT/'selected_research_policy.json')
    assert profile['results_sha256']==n.sha(t.OUT/'results.json')
    assert profile['registration_sha256']==data['registration_sha256']==n.sha(t.ROOT/'registration.json')
    qualified_receipt=n.read(t.ROOT/'qualification_complete.json')
    assert qualified_receipt['results_sha256']==profile['results_sha256']
    assert qualified_receipt['profile_sha256']==n.sha(t.OUT/'selected_research_policy.json')
    complete=n.read(t.OUT/'completion.json')
    for name,expected in complete['files'].items():assert n.sha(t.OUT/name)==expected,name
    assert not profile['promotion'] and not profile['independent_validation']
    catalog={c['id']:c for c in n.read(t.ROOT/'candidate_plan.json')['candidates']}
    selected=next(r for r in data['candidates'] if r['id']==profile['selected_id']);control=data['candidates'][0]
    assert profile['mutations']==catalog[selected['id']]['mutations']
    assert len(catalog) in (4,5) and len(data['candidates'])==len(catalog)
    assert all(not control['mutations'][key] for key in ('capacity_liquidation_tail','capacity_structural_partial','capacity_late_confirmation'))
    proofs=[]
    for row in data['candidates']:
        assert len(row['windows'])==2
        assert row['gates']==q.gates(row['windows']['train'],row['windows']['oos'],control,data['references'])
        assert row['nominal_qualified']==all(row['gates'].values())
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
                proofs.append({'candidate':row['id'],'window':w,'scenario':scenario,'capped':capped,'completion_sha256':proof['completion_sha256']})
    qualified=[r for r in data['candidates'][1:] if r.get('nominal_qualified') and r.get('stress_qualified')]
    if qualified:
        assert selected['id']==max(qualified,key=q.rank)['id']
        assert profile['status']=='qualified_research_successor'
    else:assert selected['id']==control['id'] and profile['status']=='retain_selected_no_qualified_upgrade'
    assert len(control['stress'])==len(control['capped'])==len(selected['capped'])==2
    for w in ('train','oos'):
        for scenario,capped in [('baseline',False),('combined',False),('baseline',True)]:
            parity=n.read(t.ROOT/f'control_parity_{w}_{scenario}_{capped}.json')
            assert parity['passed'] and parity['daily_and_intraday_marks_exact']
            assert n.sha(Path(parity['parent'])/'completion.json')==parity['parent_completion']
            assert n.sha(Path(parity['new'])/'completion.json')==parity['new_completion']
    prepared=n.read(t.ROOT/'preparation_complete.json')
    assert prepared['all_overlapping_observation_rows_exact'] and prepared['is_through']=='2026-03-31'
    assert prepared['sources']==profile['source_generation_hashes']
    for record in n.read(t.OUT/'implementation_receipt.json')['changed_sources']:
        w,name=record['window'],record['name']
        assert n.sha(t.ENGINES/w/'inputs/files'/name)==record['after']
        assert n.sha(t.OUT/'review_bundle'/w/name)==record['after']
    receipt=n.read(t.OUT/'test_receipt.json')
    assert receipt['passed']>=50 and receipt['failures']==receipt['skipped']==receipt['exit_code']==0
    assert len(receipt['frozen_sources'])==2*len(t.PATCHED)
    for name,digest in receipt['test_hashes'].items():assert n.sha(t.REPO/name)==digest
    for name,digest in receipt['frozen_sources'].items():assert n.sha(Path(name))==digest
    assert n.sha(t.OUT/'focused_tests.log')==receipt['log_sha256']
    assert n.sha(t.OUT/'focused_tests.xml')==receipt['junit_sha256']
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
    output={'passed':bool(chart_reviewed),'automated_checks_passed':True,'verified_utc':s.utc(),
        'native_windows':len(proofs),'native_proofs':proofs,'focused_tests_passed':receipt['passed'],
        'chart_visually_reviewed':bool(chart_reviewed),'chart_dimensions':dimensions,
        'chart_sha256':n.sha(t.OUT/'results.png'),'local_links':links.local,
        'policy_sha256':n.sha(t.OUT/'selected_research_policy.json'),'completion_sha256':n.sha(t.OUT/'completion.json'),
        'promotion':False,'independent_validation':False}
    n.write(t.OUT/'delivery_checks.json',output)
    print({k:output[k] for k in ('passed','automated_checks_passed','native_windows','focused_tests_passed','chart_visually_reviewed')})
    assert chart_reviewed,'Automated checks complete; visual review is still required before final delivery'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--chart-reviewed',action='store_true')
    verify(parser.parse_args().chart_reviewed)

"""Bind the delivered report and policy to authenticated native evidence."""
from __future__ import annotations
import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse, unquote
from scripts import olr_targeted_risk_study_v1 as t
from scripts import olr_boundary_restart_results_v1 as history
n,s,b=t.n,t.s,t.b


def verify(chart_reviewed=False):
    t.activate()
    decision=n.read(t.OUT/'selected_research_policy.json')
    data=n.read(t.OUT/'results.json')
    complete=n.read(t.OUT/'completion.json')
    assert decision['results_sha256']==n.sha(t.OUT/'results.json')
    assert decision['registration_sha256']==n.sha(t.ROOT/'registration.json')
    assert decision['stress_selection_sha256']==n.sha(t.ROOT/'stress_selection.json')
    assert not decision['promotion'] and not decision['independent_validation']
    assert complete['selected']==decision['selected_id']
    for name,expected in complete['files'].items():
        assert n.sha(t.OUT/name)==expected,name
    catalog={c['id']:c for c in n.read(t.ROOT/'candidate_plan.json')['candidates']}
    selected=next(row for row in data['candidates'] if row['id']==decision['selected_id'])
    assert decision['mutations']==catalog[selected['id']]['mutations']
    if decision['status']=='qualified_research_successor':
        assert selected['nominal_qualified'] and selected['stress_qualified']
    proofs=[]
    assert len(data['candidates'])==7
    for row in data['candidates']:
        assert len(row['windows'])==2
        for group,scenario,capped in [('windows','baseline',False),('stress','combined',False),('capped','baseline',True)]:
            assert len(row[group]) in (0,2)
            for w,summary in row[group].items():
                proof=history.audit(catalog[row['id']],w,scenario,capped)
                assert proof['completion_sha256']==summary['completion_sha256']
                assert proof['source_generation_hash']==decision['source_generation_hashes'][w]
                assert summary['daily_dates'][0]==('2025-08-07' if w=='train' else '2026-04-01')
                assert summary['daily_dates'][-1]==('2026-03-31' if w=='train' else '2026-07-10')
                assert summary['sessions']==(156 if w=='train' else 69)
                if w=='oos':assert proof['literal_is_terminal_state_restored']
                proofs.append({'candidate':row['id'],'window':w,'scenario':scenario,'capped':capped,
                    'completion_sha256':proof['completion_sha256'],'history_carry':w=='oos'})
    for group in ('original_saved','original_stress'):
        for summary in data[group].values():
            assert s.forensic.authenticate(Path(summary['folder']))['completion_sha256']==summary['completion_sha256']
    receipt=n.read(t.OUT/'test_receipt.json')
    assert receipt['unique_tests_passed']==49
    assert n.sha(t.REPO/'tests/strategy_olr/test_priority_loss_budget.py')==receipt['test_sha256']
    for name,expected in receipt['implementation_hashes'].items():
        assert n.sha(t.REPO/name)==expected,('Workspace changed after tests',name)
    for row in n.read(t.OUT/'implementation_receipt.json')['changed_sources']:
        for w in ('train','oos'):
            assert n.sha(t.ENGINES/w/'inputs/files'/row['name'])==row['after']
        assert n.sha(t.OUT/'review_bundle'/row['name'])==row['after']
    for w in ('train','oos'):
        assert n.read(t.ROOT/('control_parity_'+w+'.json'))['passed']
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.local=[]
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('src','href') and value and not urlparse(value).scheme:
                    self.local.append(unquote(urlparse(value).path))
    links=Links();links.feed((t.OUT/'report.html').read_text(encoding='utf-8'))
    assert links.local
    for name in links.local:assert (t.OUT/name).is_file(),name
    from PIL import Image
    with Image.open(t.OUT/'results.png') as im:
        dimensions=list(im.size);im.verify()
    result={'verified_utc':s.utc(),'passed':True,'native_windows':len(proofs),'native_proofs':proofs,
        'original_reference_windows':4,'unique_tests_passed':49,'chart_dimensions':dimensions,
        'chart_visually_reviewed':chart_reviewed,'local_report_links_checked':links.local,
        'completion_sha256':n.sha(t.OUT/'completion.json'),'policy_sha256':n.sha(t.OUT/'selected_research_policy.json'),
        'scope':'Artifact integrity, source identities, native audits, preservation gates and continuous history. No independent later-period validation or live promotion.'}
    n.write(t.OUT/'delivery_checks.json',result)
    print({k:result[k] for k in ('passed','native_windows','unique_tests_passed','chart_visually_reviewed')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--chart-reviewed',action='store_true')
    verify(parser.parse_args().chart_reviewed)

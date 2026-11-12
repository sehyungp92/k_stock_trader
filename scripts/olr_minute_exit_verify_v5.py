"""Independently recollect saved native outputs; never rewrite results/profile."""
import argparse
from pathlib import Path
from scripts import olr_minute_exit_qualify_v5 as q
t=q.t;n,s,b=t.n,t.s,t.b


def main(chart_reviewed=False):
    assert chart_reviewed,'Review report.html/report.txt and results.png first'
    q.configure();q.check_preflight();harness=q.verify_harness()
    data=n.read(t.OUT/'results.json');profile=n.read(t.OUT/'selected_research_policy.json');done=n.read(t.ROOT/'qualification_complete.json')
    assert profile['results_sha256']==done['results_sha256']==n.sha(t.OUT/'results.json')
    assert done['profile_sha256']==n.sha(t.OUT/'selected_research_policy.json')
    assert profile['registration_sha256']==data['registration_sha256']==n.sha(t.ROOT/'registration.json')
    fresh=q.collect();assert fresh==data,'Stored evidence differs from authenticated recomputation'
    assert len(data['candidates'])==2 and all(len(r[g])==2 for r in data['candidates'] for _,_,g in q.SCENARIOS)
    eligible=[r for r in data['candidates'][1:] if r['full_objective_qualified']]
    old=n.read(t.previous.OUT/'selected_research_policy.json');expected=eligible[0] if eligible else old
    assert profile['selected_id']==expected.get('id',expected.get('selected_id'))
    assert profile['mutations']==expected['mutations']
    assert profile['risk_component_ids']==[r['id'] for r in data['candidates'][1:] if r['risk_component_qualified']]
    assert not profile['promotion'] and not profile['independent_validation']
    if not eligible:assert profile['source_generation_hashes']==old['source_generation_hashes']
    report=n.read(t.OUT/'report_receipt.json')
    assert report['results_sha256']==profile['results_sha256'] and report['profile_sha256']==done['profile_sha256']
    assert report['script_sha256']==n.sha(t.REPO/'scripts/olr_minute_exit_report_v5.py')
    for name,digest in report['files'].items():assert n.sha(t.OUT/name)==digest,name
    for w in ('train','oos'):
        ready=n.read(t.ENGINES/w/'ready.json')
        assert n.source_hash(Path(ready['files']))==ready['study']['source_generation_hash']
        for name,digest in ready['manifest']['files'].items():assert n.sha(Path(ready['files'])/name)==digest,name
    n.write(t.OUT/'delivery_checks.json',dict(passed=True,utc=s.utc(),native_windows=12,chart_reviewed=True,
        tests=n.read(harness['tests_receipt'])['passed'],qualification_tests=harness['qualification_tests'],
        result_hash=profile['results_sha256'],profile_hash=done['profile_sha256'],selected_id=profile['selected_id'],
        report_receipt_sha256=n.sha(t.OUT/'report_receipt.json'),verifier_sha256=n.sha(__file__),
        native_history_source_clock_capacity_ledger_and_every_gate_verified=True,exact_recollection_equal=True))
    print(dict(passed=True,native_windows=12,selected_id=profile['selected_id']))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--chart-reviewed',action='store_true');main(ap.parse_args().chart_reviewed)

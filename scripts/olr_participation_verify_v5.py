"""Authenticate completed comparisons and delivery, preserving frozen identities."""
import argparse
import json
from pathlib import Path
from scripts import olr_participation_study_v5 as t
from scripts import olr_participation_qualify_v5 as q
from scripts import olr_participation_prepare_v5 as prep
n,s,b=t.n,t.s,t.b


def serialized_evidence(value):
    """Match the writer's JSON types and only the collector's unordered sets.

    Keep every field and duplicate; execution, ledger and all_matches order stays
    significant. The old collector uses set differences for added/removed trades.
    """
    value=json.loads(json.dumps(value,sort_keys=True))
    for candidate in value['candidates']:
        for comparison in candidate.get('trade_comparison',{}).values():
            for name in ('added','removed'):
                comparison[name].sort(key=lambda row:json.dumps(row,sort_keys=True))
    return value


def verify_harness():
    original=n.read(t.OUT/'research_harness.json')['files']
    receipt=n.read(t.OUT/'reporting_changes.json')
    assert receipt['results_sha256']==n.sha(t.OUT/'results.json')
    assert receipt['profile_sha256']==n.sha(t.OUT/'selected_research_policy.json')
    assert set(receipt['files'])=={'scripts/olr_participation_verify_v5.py'}
    for name,digest in original.items():
        if name in receipt['files']:
            change=receipt['files'][name]
            archive=(t.OUT/change['archive']).resolve()
            assert archive.is_relative_to(t.OUT.resolve())
            assert n.sha(archive)==change['original_sha256']==digest
            assert n.sha(t.REPO/name)==change['current_sha256']
        else:
            assert n.sha(t.REPO/name)==digest,name


def main(chart_reviewed=False):
    assert chart_reviewed,'Visually review results.png and report.html/report.txt first'
    q.configure();q.check_preflight()
    data=n.read(t.OUT/'results.json');profile=n.read(t.OUT/'selected_research_policy.json');done=n.read(t.ROOT/'qualification_complete.json')
    assert n.sha(t.OUT/'results.json')==profile['results_sha256']==done['results_sha256']
    assert n.sha(t.OUT/'selected_research_policy.json')==done['profile_sha256']
    assert data['registration_sha256']==profile['registration_sha256']==n.sha(t.ROOT/'registration.json')
    assert n.sha(t.previous.OUT/'results.json')==n.read(t.ROOT/'registration.json')['reference_results_sha256']
    assert n.sha(t.previous.OUT/'selected_research_policy.json')==n.read(t.ROOT/'registration.json')['reference_profile_sha256']
    catalog=n.read(t.ROOT/'candidate_plan.json')['candidates'];assert len(catalog)==len(data['candidates'])==2
    checks=[]
    for c in catalog:
        for scenario,capped in [('baseline',False),('combined',False),('baseline',True)]:
            for w in ('train','oos'):
                proof=q.audit(c,w,scenario,capped)
                if c['label']=='selected_control':prep.parity(c,w,scenario,capped)
                checks.append(dict(candidate=c['id'],window=w,scenario=scenario,capped=capped,proof=proof))
    fresh=q.collect()
    assert serialized_evidence(fresh)==serialized_evidence(data),'Stored results differ from independently recollected evidence'
    qualified=[r for r in fresh['candidates'][1:] if all(r.get(k,False) for k in ('nominal_qualified','stress_qualified','capped_qualified'))]
    expected=qualified[0]['id'] if qualified else catalog[0]['id'];assert profile['selected_id']==expected
    report=n.read(t.OUT/'report_receipt.json');assert report['results_sha256']==profile['results_sha256']
    for name,digest in report['files'].items():assert n.sha(t.OUT/name)==digest
    assert report['script_sha256']==n.sha(t.REPO/'scripts/olr_participation_report_v5.py')
    verify_harness()
    for w in ('train','oos'):
        ready=n.read(t.ENGINES/w/'ready.json')
        for name,digest in ready['manifest']['files'].items():assert n.sha(Path(ready['files'])/name)==digest
    n.write(t.OUT/'delivery_checks.json',dict(passed=True,utc=s.utc(),native_windows=len(checks),chart_reviewed=True,
        tests=n.read(t.OUT/'test_receipt.json')['passed'],selected_id=expected,checks=checks,
        result_hash=profile['results_sha256'],profile_hash=done['profile_sha256'],source_history_native_and_gate_checks=True,
        reporting_changes_sha256=n.sha(t.OUT/'reporting_changes.json'),
        evidence_comparison='Exact JSON field equality; sort only set-derived added/removed trade-comparison lists'))
    print(dict(passed=True,native_windows=len(checks),selected_id=expected))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--chart-reviewed',action='store_true');args=ap.parse_args();main(args.chart_reviewed)

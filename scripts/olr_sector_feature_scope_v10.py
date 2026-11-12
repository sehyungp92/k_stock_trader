"""Additive exact endpoint correction; original frozen audit stays unchanged."""
from pathlib import Path
import argparse
import difflib
import hashlib
import importlib.util
import json
import sys
from datetime import date

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010/structural_v6/sector_feature_audit_v1')
ORIGINAL=REPO/'scripts/olr_sector_feature_audit_v10.py'
TEST=REPO/'tests/strategy_olr/test_sector_feature_scope_v10.py'
OLD="assert len(days)==215 and days[-1]==date(2026,3,31)"
NEW="assert len(days)==215 and days[-1]==date(2026,3,27)"


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def corrected(text):
    assert text.count(OLD)==1
    return text.replace(OLD,NEW,1)


def scope(index):
    base=index['identity']['base'];days=base['context_dates']
    assert days==base['output_dates']==sorted(set(days))
    assert len(days)==215 and days[0]=='2025-05-12' and days[-1]=='2026-03-27'
    assert set(index['files'])=={'static.pickle',*(d+'.pickle' for d in days)}
    return days


def module():
    spec=importlib.util.spec_from_file_location('sector_scope_effective',ORIGINAL)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m
    # __file__ remains original for its frozen paths; only the bound literal changes.
    exec(compile(corrected(ORIGINAL.read_text(encoding='utf-8')),str(ORIGINAL),'exec'),m.__dict__)
    return m


def bind():
    m=module();p=m.authenticate();assert not (ROOT/'scope_repair.json').exists()
    index=m.read(Path(p['source']['cache'])/'index.json');scope(index)
    archive=ROOT/'audit_failed_attempt_0';preserved=m.read(archive/'preservation.json')
    for path,record in preserved['files'].items():
        assert digest(record['saved'])==record['sha256']==digest(path)
    import xml.etree.ElementTree as ET
    suites=list(ET.parse(ROOT/'scope_tests.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==3
    assert not any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped'))
    copy=ROOT/'scope_repair/effective_audit.py';copy.parent.mkdir()
    with copy.open('x',encoding='utf-8',newline='') as f:f.write(corrected(ORIGINAL.read_text(encoding='utf-8')))
    paths=[Path(__file__),TEST,ROOT/'scope_tests.xml',ROOT/'scope_tests.log',archive/'preservation.json',copy]
    m.write(ROOT/'scope_repair.json',dict(utc=m.utc(),original_protocol_sha256=digest(ROOT/'protocol.json'),
        correction='Only assert endpoint March31 -> March27. Existing215 payloads unchanged. March30/31 remain excluded original entry tail, not fabricated zero-opportunity days. This corrects the protocol scope description too.',
        substitutions={OLD:NEW},files={str(p):digest(p) for p in paths}))
    print('Exact endpoint correction bound before feature outcomes.')


def authenticate(m):
    m.authenticate();repair=m.read(ROOT/'scope_repair.json')
    assert repair['original_protocol_sha256']==digest(ROOT/'protocol.json')
    for path,h in repair['files'].items():assert digest(path)==h,path
    assert (ROOT/'scope_repair/effective_audit.py').read_text(encoding='utf-8')==corrected(ORIGINAL.read_text(encoding='utf-8'))
    scope(m.read(Path(m.read(ROOT/'protocol.json')['source']['cache'])/'index.json'))


def report(m):
    from html import escape
    done=m.read(ROOT/'completion.json');effective=m.read(ROOT/'scope_complete.json')
    assert effective['completion_sha256']==digest(ROOT/'completion.json')
    assert effective['repair_sha256']==digest(ROOT/'scope_repair.json')
    assert done['ledger_sha256']==digest(ROOT/'ledger.json')
    rows=m.read(ROOT/'ledger.json');m.verify_ledger(rows);summary=m.summarize(rows)
    summary.update(source_first_day=rows[0]['day'],source_last_day=rows[-1]['day'])
    # Keep original scalar summary schema for the frozen independent verifier.
    m.write(m.OUT/'summary.json',m.summarize(rows))
    m.write(m.OUT/'selected_research_policy.json',dict(selected_id='c4f49be01ed09b8dc',promoted=False,
        live_activation=False,summary_sha256=digest(m.OUT/'summary.json')))
    paragraphs=['OLR causal sector-volume feature contract audit',
        'No trading decisions, payoff labels, threshold changes or policy promotion.',
        'The existing single-day constructor discards historical volumes and defaults relative volume to1. The causal adapter matches the unchanged native full-history constructor, with independently checked prior lists, ratios, peer blends and participation.',
        'The denominator is the arithmetic mean of preceding20 positive completed-prefix volumes, not a median. Missing baselines remain null separately from native neutral1. Qualified ratios describe observed peers; full fixed-universe peer coverage is separately recorded.',
        json.dumps(summary,indent=2),
        'Scope erratum: the original frozen narrative/check incorrectly said March31. The authenticated215 source snapshots end March27. An additive one-literal endpoint correction followed the preserved failure before any feature ledger; March30/31 are the original two-session entry tail. All source/cache/native bytes remain unchanged.',
        'Full cached bars matched authenticated database slices before close-anchored14:30 prefix qualification. January2 uses the late10:00 opening and unchanged14:30 cutoff. No interpolation or new source reconstruction occurred.',
        'The existing static panel is a source-bound reference. The previous uncalibrated historical-feature trading substitution failed; correctness/nondegeneracy is not alpha or permission to repeat it. Rejected rank calibration and score-only cohorts remain closed.',
        'Further economic work requires its own evidence-based hypothesis and preregistration. Selected c4f49be01ed09b8dc remains unchanged; the full return/frequency/drawdown objective is not met.',
        'Limits: fixed universe/taxonomy, missing peer coverage, positive-volume-conditioned history, varying calendar ages and session durations, sparse auctions/inferred clocks and conservative execution proxies. April-July is repeatedly inspected development and is unused in this audit. No independent-validation claim.']
    text='\n\n'.join(paragraphs)+'\n';m.OUT.mkdir(exist_ok=True,parents=True)
    for name,content in [('report.txt',text),('report.html','<!doctype html><meta charset="utf-8"><title>OLR feature audit</title><style>body{max-width:1000px;margin:40px auto;font:16px system-ui;line-height:1.6}pre{white-space:pre-wrap}</style><pre>'+escape(text)+'</pre>')]:
        with (m.OUT/name).open('x',encoding='utf-8') as f:f.write(content)
    bundle=m.OUT/'review_bundle';bundle.mkdir();patch=[]
    sources=[ORIGINAL,REPO/m.NAMES[1],Path(__file__),TEST]
    for source in sources:
        rel=source.relative_to(REPO);target=bundle/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        patch.extend(difflib.unified_diff([],source.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile='b/'+rel.as_posix()))
    (bundle/'implementation.patch').write_text(''.join(patch),encoding='utf-8')
    paths=[m.OUT/n for n in ('summary.json','selected_research_policy.json','report.txt','report.html','review_bundle/implementation.patch')]
    paths += [ROOT/'scope_complete.json',ROOT/'scope_repair.json']+list(bundle.rglob('*.py'))
    m.write(m.OUT/'report_receipt.json',dict(utc=m.utc(),completion_sha256=digest(ROOT/'completion.json'),files={str(p):digest(p) for p in paths}))
    print(json.dumps(summary))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['bind','audit','report','verify']);phase=p.parse_args().phase
    if phase=='bind':bind();return
    m=module();authenticate(m)
    if phase=='audit':
        reg,ready,guard,clock=m.setup();m.audit(reg,ready,guard,clock)
        m.write(ROOT/'scope_complete.json',dict(utc=m.utc(),completion_sha256=digest(ROOT/'completion.json'),repair_sha256=digest(ROOT/'scope_repair.json')))
    elif phase=='report':report(m)
    else:
        m.verify()
        m.write(m.OUT/'scope_delivery.json',dict(passed=True,repair_sha256=digest(ROOT/'scope_repair.json'),checks_sha256=digest(m.OUT/'delivery_checks.json')))


if __name__=='__main__':main()

"""Read-only explanation of rejected coverage; no refit or gate modification."""
from pathlib import Path
from datetime import date,datetime,timezone
import hashlib
import importlib.util
import json
import pickle
import sys

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010/structural_v6/sector_information_v1/support_diagnosis')
OUT=REPO/'data/backtests/analysis/olr_structural_20261012/sector_information_v1'


def main():
    spec=importlib.util.spec_from_file_location('support_feature_contract',REPO/'scripts/olr_sector_feature_audit_v10.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    reg,ready,guard,clock=m.setup();m.authenticate()
    assert m.sha(OUT/'results.json')=='1209a50d8330b45656d584a6edec24fad1b4eb973f454e4ab3e500b1006b2860'
    assert m.sha(OUT/'delivery_checks.json')=='1c2fc3e3e37424a1ebf2cc37b5909ba347b65e14747e3bbee17e311d05b9dd93'
    done=m.read(m.ROOT/'completion.json');assert done['ledger_sha256']==m.sha(m.ROOT/'ledger.json')
    records=m.read(OUT/'records.json');rows=m.read(m.ROOT/'ledger.json');lookup={(r['day'],r['symbol']):r for r in rows}
    requested=[];keys=set()
    for r in records:
        if '2025-11-01'<=r['day']<='2025-12-31' and r['x'] is None:
            f=lookup[r['day'],r['symbol']];peer=f['peer_support'];assert peer
            missing=peer['missing_sector_peers'] if peer['weight']==1 else [v['symbol'] for v in rows if v['day']==r['day'] and not v['source_complete']]
            requested.append(dict(day=r['day'],symbol=r['symbol'],missing_peer_keys=[[r['day'],s] for s in missing]))
            keys.update((r['day'],s) for s in missing)
    from backtests.core.input_manifest import InputManifest
    from strategy_common.session_coverage import session_bar_coverage
    from strategy_common.source_quality import missing_prefix,expected_labels
    files=Path(ready['files']);cache=Path(reg['source']['cache']);index=m.read(cache/'index.json');details=[];payload_hashes={}
    assert not (ROOT/'diagnosis.json').exists()
    with InputManifest(ready['study']['manifest']).activate():
        for day,symbol in sorted(keys):
            path=cache/(day+'.pickle');assert m.sha(path)==index['files'][path.name]
            payload_hashes[str(path)]=m.sha(path)
            with path.open('rb') as f:payload=pickle.load(f)
            bars=payload[3].get((date.fromisoformat(day),symbol),())
            cutoff=datetime.fromisoformat(lookup[day,symbol]['available_at'])
            stamps=[b.timestamp for b in bars]
            coverage=session_bar_coverage(stamps,date.fromisoformat(day),5,symbol=symbol)
            missing=missing_prefix(symbol,date.fromisoformat(day),stamps,cutoff)
            from datetime import timedelta
            expected=sum(s+timedelta(minutes=5)<=cutoff for s in expected_labels(symbol,date.fromisoformat(day)))
            inactive=coverage.get('reason')=='sourced_symbol_suspension' and coverage['expected_intervals']==0 and not bars
            details.append(dict(day=day,symbol=symbol,cutoff=cutoff.isoformat(),calendar_result=coverage,
                raw_bar_count=len(bars),expected_prefix_intervals=expected,missing_prefix_labels=[s.isoformat() for s in missing],
                classification='documented_suspension_zero_expected_observations' if inactive else 'unresolved_expected_prefix_missing',
                feature_ledger_source_complete=lookup[day,symbol]['source_complete']))
    classified={(r['day'],r['symbol']):r for r in details}
    for r in requested:
        r['all_absent_peers_documented_inactive']=all(classified[tuple(k)]['classification']=='documented_suspension_zero_expected_observations' for k in r['missing_peer_keys'])
    notice=m.read(ROOT/'primary_notice.json')
    assert notice['publication_date']=='2025-10-22' and notice['suspension_start']=='2025-10-30' and notice['suspension_end']=='2025-11-21'
    value=dict(utc=datetime.now(timezone.utc).isoformat(),description='Descriptive diagnosis after the coverage failure; preliminary four-key counts and primary notice already inspected. Not a preregistered economic experiment.',
        opportunities=requested,missing_peers=details,primary_notice=notice,
        unsupported_opportunities=len(requested),suspension_only_opportunities=sum(r['all_absent_peers_documented_inactive'] for r in requested),
        completed_information_test_unchanged=True,new_fits=0,promotion=False,
        interpretation='The proposed full-fixed-universe support requirement conflated expected inactivity with missing active-peer observations. This is a design defect in the proposed diagnostic support gate, not failed alpha evidence. Preserve its rejection. Any active-eligibility contract needs separate validation/registration with dated notices before affected decisions; never infer inactivity from absent prices or observed payoffs.')
    m.write(ROOT/'diagnosis.json',value)
    lines=['OLR sector-information coverage postmortem',value['interpretation'],
        f"Unsupported incumbent observations: {len(requested)}; {value['suspension_only_opportunities']} depend only on a documented inactive peer.",
        'Frozen native calendar classifies207940 on Nov3/13/17 as a sourced suspension with zero expected observations. The source notice was published Oct22, before the affected periodOct30-Nov21.',
        'The other two unsupported incumbents,035900 and041510 on Dec24, use market shrinkage with298050 missing an expected prefix. No sourced inactivity for that row is established here; do not impute or exempt it.',
        'This does not change the completed information protocol, coverage gate, source ledger, native feature values or selected policy. No real regression was fitted. A new active-peer contract would be a separately registered measurement design, not a relabeling of this result.',
        'Primary source: '+notice['url']]
    text='\n\n'.join(lines)+'\n'
    with (OUT/'support_postmortem.txt').open('x',encoding='utf-8') as f:f.write(text)
    paths=[Path(__file__),ROOT/'primary_notice.json',ROOT/'diagnosis.json',OUT/'support_postmortem.txt',OUT/'records.json',OUT/'results.json',OUT/'delivery_checks.json',m.ROOT/'ledger.json',m.ROOT/'completion.json']
    m.write(ROOT/'receipt.json',dict(utc=m.utc(),files={str(p):m.sha(p) for p in paths},cached_payloads=payload_hashes,
        native_imports=m.imports(guard),native_files=str(files),calendar_sha256=m.sha(files/'kis_core/data/krx_sessions.yaml'),new_fits=0))
    print(text)


if __name__=='__main__':main()

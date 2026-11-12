"""Independent supply arithmetic and original-snapshot reconciliation."""
from collections import Counter
from pathlib import Path
from html import escape
import difflib
import shutil
from scripts import olr_opportunity_v7 as c


def main():
    assert not (c.OUT/'inventory_checks.json').exists(), 'Never rerun a completed inventory verifier'
    registration = c.read(c.ROOT / 'registration.json')
    proof = c.read(c.ROOT / 'ledger_complete.json')
    assert proof['registration_sha256'] == c.sha(c.ROOT / 'registration.json')
    assert proof['ledger_sha256'] == c.sha(c.ROOT / 'opportunities_train.json')
    assert proof['summary_sha256'] == c.sha(c.OUT / 'supply_summary.json')
    repair = c.read(c.ROOT / 'clock_repair.json')
    assert proof['clock_repair_sha256'] == c.sha(c.ROOT / 'clock_repair.json')
    assert proof['current_sector_panel_recomputed_from_completed_prefix']
    assert repair['original_registration_sha256'] == c.sha(c.ROOT / 'registration.json')
    for name,digest in repair['files'].items():
        assert c.sha(c.REPO/name) == digest
    for name,digest in repair['evidence'].items():
        assert c.sha(name) == digest
    source = (c.REPO/'scripts/olr_opportunity_worker_v7.py').read_text(encoding='utf8')
    assert repair['original_worker_sha256'] == c.sha(c.REPO/'scripts/olr_opportunity_worker_v7.py')
    for change in repair['substitutions']:
        assert source.count(change['old']) == 1
        source = source.replace(change['old'],change['new'])
    assert source == (c.REPO/'scripts/olr_opportunity_clock_worker_v7.py').read_text(encoding='utf8')
    for name, digest in registration['files'].items():
        assert c.sha(c.REPO / name) == digest, name
    for name, digest in registration['inputs'].items():
        assert c.sha(name) == digest, name
    ready = c.read(registration['source']['ready'])
    for name, digest in proof['imports'].items():
        assert ready['manifest']['files'][name] == c.sha(Path(ready['files']) / name) == digest
    original = {r['trade_date']:r for r in c.read(registration['source']['snapshot'])}
    rows = c.read(c.ROOT / 'opportunities_train.json')
    assert len(rows) == 103*len(original) == len({(r['day'],r['symbol']) for r in rows})
    assert Counter(r['day'] for r in rows) == {day:103 for day in original}
    for r in rows:
        snap = original[r['day']]
        selected = {x['symbol'] for x in snap['candidates']}
        assert r['selected'] == (r['symbol'] in selected)
        assert r['daily_rejections'] == snap['metadata'].get('rejected_symbols',{}).get(r['symbol'],[])
        assert r['afternoon_final_rejections'] == snap['metadata'].get('afternoon_rejected_symbols',{}).get(r['symbol'],[])
        reason_set = set(r['afternoon_final_rejections'])
        only_score = bool(reason_set) and reason_set.issubset({
            'afternoon_score_below_floor','afternoon_score_above_cap',
            'afternoon_score_in_reject_band','afternoon_score_band_rule_miss'})
        if r['selected']:
            expected = 'incumbent'
            assert not r['afternoon_hard_rejections'] and not reason_set
        elif r['morning_selected'] and r['features'] is not None and not r['afternoon_hard_rejections'] and only_score:
            expected = 'score_mapping_only'
        else:
            expected = 'other_exclusion'
        assert r['cohort'] == expected
        if expected != 'other_exclusion':
            assert not r['daily_rejections'] and not r['source_prefix_excluded']
        assert r['available_at'][:10] == r['day']
        assert not any(k in r for k in ('return','net_pnl','future_price','outcome'))
    selected = [r for r in rows if '2025-08-07' <= r['day'] <= '2026-03-31' and r['cohort']=='incumbent']
    extra = [r for r in rows if '2025-08-07' <= r['day'] <= '2026-03-31' and r['cohort']=='score_mapping_only']
    days = {r['day'] for r in extra}
    empty_days = days - {r['day'] for r in selected}
    blocks = []
    for lo,hi in [('2025-08-07','2025-10-31'),('2025-11-01','2025-12-31'),('2026-01-01','2026-03-31')]:
        sample = [r for r in extra if lo <= r['day'] <= hi]
        blocks.append(dict(start=lo,end=hi,candidates=len(sample),days=len({r['day'] for r in sample})))
    expected_gates = dict(at_least_60_candidates=len(extra)>=60,at_least_30_days=len(days)>=30,
        every_block_10_candidates_5_days=all(b['candidates']>=10 and b['days']>=5 for b in blocks),
        at_least_10_previously_empty_days=len(empty_days)>=10)
    summary = c.read(c.OUT / 'supply_summary.json')
    assert summary['support_checks'] == expected_gates and summary['blocks'] == blocks
    assert summary['cohort_counts']['incumbent'] == len(selected)
    assert summary['cohort_counts'].get('score_mapping_only',0) == len(extra)
    assert summary['payoff_study_eligible'] == all(expected_gates.values())
    assert not summary['portfolio_experiment_eligible'] and not summary['pnl_consumed']
    panel_path = c.ROOT.parent / 'train_panel.json'
    assert c.sha(panel_path) == 'bfd75304ee8bebf1a6eb0adfa0fdf70fc5f98b932b8b409592f2bd907c00e951'
    calendar_days = {d for d in c.read(panel_path)['sessions'] if d >= '2025-08-07'}
    missing = sorted(calendar_days-set(original))
    assert all(x['raw_selected_exact'] and x['rejection_metadata_exact'] and x['future_intraday_invariance'] for x in proof['daily_checks'])
    receipt = dict(utc=c.utc(),passed=True,ledger_sha256=proof['ledger_sha256'],
        summary_sha256=proof['summary_sha256'],verifier_sha256=c.sha(__file__),
        registration_sha256=c.sha(c.ROOT/'registration.json'), proof_sha256=c.sha(c.ROOT/'ledger_complete.json'),
        source_calendar_sha256=c.sha(panel_path), source_sessions_without_saved_entry_snapshot=missing,
        native_imports_authenticated=len(proof['imports']), independent_supply_arithmetic=True,
        original_snapshot_rejections_reconciled=True, no_trading_payoff_labels=True,
        payoff_study_eligible=all(expected_gates.values()), portfolio_experiment_eligible=False)
    report = [
        'OLR training opportunity inventory — supply only',
        f'Original selected candidates: {len(selected)} on {len({r["day"] for r in selected})} dates.',
        f'Score-map-only exclusions: {len(extra)} on {len(days)} dates; {len(empty_days)} previously empty dates.',
        'Every additional candidate passed original daily selection and non-score afternoon guards. No F participation relaxation or late confirmation is included.',
        *[f'{b["start"]} to {b["end"]}: {b["candidates"]} additional candidates on {b["days"]} dates.' for b in blocks],
        'Supply gates: '+str(expected_gates),
        'Eligible for a separately registered payoff study: '+str(all(expected_gates.values())),
        'Portfolio experiment eligible: False. No returns, alpha, fill counts or new strategy performance are established.',
        'A preserved failed inventory exposed an eligibility-clock defect on the late-opening 2026-01-02 session: completeness originally required bars through 15:30 for a 14:30 OLR decision. The additive diagnostic wrapper aligns completeness with the close-anchored decision time and recomputes current sector features from the completed prefix. Six frozen-native clock contracts passed. Original selections remain exact; no performance impact is attributed to this correction.',
        'Source calendar sessions without saved entry snapshots (not imputed as zero-opportunity days): '+str(missing),
        f'All {len(rows)} symbol-days, original selected candidates and rejection metadata reconciled; {len(proof["imports"])} native module hashes authenticated.',
        'April–July was not read for this inventory and remains repeatedly inspected development data. Fixed universe/taxonomy, sparse auction history, inferred clocks and conservative execution proxies remain.',
        'Selected policy c4f49be01ed09b8dc unchanged. No live activation. The broader objective remains incomplete.'
    ]
    (c.OUT/'inventory_report.txt').write_text('\n\n'.join(report)+'\n',encoding='utf8')
    html='<!doctype html><html lang="en"><meta charset="utf-8"><title>OLR opportunity inventory</title><style>body{font:17px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:0 24px;color:#182c3e}p{margin:18px 0}</style><h1>'+escape(report[0])+'</h1>'+''.join('<p>'+escape(p)+'</p>' for p in report[1:])+'</html>'
    (c.OUT/'inventory_report.html').write_text(html,encoding='utf8')
    bundle=c.OUT/'review_bundle';bundle.mkdir(exist_ok=False)
    sources={};patch=[]
    names=sorted(set(registration['files'])|set(repair['files'])|{'scripts/olr_opportunity_verify_v7.py'})
    for name in names:
        source=c.REPO/name;target=bundle/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);sources[name]=dict(source=str(source),sha256=c.sha(target))
        patch.extend(difflib.unified_diff([],source.read_text(encoding='utf8').splitlines(True),fromfile='/dev/null',tofile='b/'+name))
    (bundle/'implementation.patch').write_text(''.join(patch),encoding='utf8')
    c.write(bundle/'sources.json',sources)
    receipt['report_files']={name:c.sha(c.OUT/name) for name in ('inventory_report.txt','inventory_report.html','review_bundle/implementation.patch','review_bundle/sources.json')}
    receipt['clock_repair_sha256']=c.sha(c.ROOT/'clock_repair.json')
    c.write(c.OUT / 'inventory_checks.json', receipt)
    print('\n'.join(report))


if __name__ == '__main__':
    main()

"""Verify reviewed delivery against frozen inputs and saved native accounting."""
from pathlib import Path
import argparse
import math
from scripts import olr_structural_native_v6 as e
from scripts import olr_structural_results_v6 as results
from scripts import olr_structural_feedback_v6 as feedback
from scripts.olr_structural_v6 import REPO, read, sha, write, utc


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) < 1e-7, (a, b)


def drawdown(values):
    peak, worst = 1e8, 0.
    for value in values:
        assert math.isfinite(value) and value > 0
        peak = max(peak, value)
        worst = max(worst, (peak-value)/peak*100)
    return worst


def verify_training(data, registered):
    old = data['references']['retained']
    control, treatment = data['candidates']
    expected = {}
    for group in ('windows', 'stress', 'capped'):
        a, z = control[group]['train'], treatment[group]['train']
        checks = {'matched_is_return': z['return_pct'] >= a['return_pct'] - 1.5,
                  'is_selected_return': z['return_pct'] >= old[group]['train']['return_pct'] - 1.5}
        if group != 'capped':
            original = data['references']['original_stress' if group == 'stress' else 'original']['train']
            guard = data['references']['guard'][group]['train']
            checks.update(is_original_return=z['return_pct'] >= original['return_pct'] - registered['is_return_loss_max_pp_original'],
                is_guard_return=z['return_pct'] >= guard['return_pct'] - registered['is_return_loss_max_pp_guard'],
                is_guard_drawdown=z['drawdown_pct'] <= guard['drawdown_pct'] + registered['is_dd_increase_max_pp_guard'])
        expected[group] = checks
    assert data['training_decision']['checks'] == expected
    eligible = all(all(row.values()) for row in expected.values())
    assert data['training_decision']['oos_eligible'] == eligible == data['oos_evaluated']
    assert data['completed_cases'] == (12 if eligible else 6)
    return expected


def verify_saved_accounting(data):
    cases = {}
    for candidate in data['candidates']:
        for scenario, capped, group in e.GROUPS:
            for window, row in candidate[group].items():
                folder = Path(row['folder'])
                e.s.forensic.authenticate(folder)
                result, nav = read(folder / 'result.json'), read(folder / 'minute_nav_evidence.json')
                trades, daily = read(folder / 'trades.json'), read(folder / 'daily.json')
                close(sum(t['net_pnl'] for t in trades)/1e6, row['return_pct'])
                close(100*(daily[-1]['equity']/1e8-1), row['return_pct'])
                close(nav['final_equity'], daily[-1]['equity'])
                close(drawdown([r['equity'] for r in nav['common_original_grid']]), row['drawdown_pct'])
                close(100*(1-nav['maximum']['trough_equity']/nav['maximum']['peak_equity']), row['complete_fine_drawdown_pct'])
                assert row['complete_fine_drawdown_pct'] + 1e-8 >= row['drawdown_pct']
                assert row['unique_entries'] == row['trades'] == len(trades)
                assert row['entry_days'] == len({t['entry_fill_time'][:10] for t in trades})
                close(row['entry_notional'], sum(t['qty']*t['entry_price'] for t in trades))
                assert result['audit_status'] == 'passed' and result['execution_validity_pass']
                assert not result['terminal_inventory'] and not result['terminal_pending']
                if candidate['structural_mode'] == 'off':
                    parity = read(folder / 'control_parity.json')
                    prior = Path(parity['original'])
                    e.s.forensic.authenticate(prior)
                    for name in ('daily.json','equity.json'):
                        assert read(folder/name) == read(prior/name)
                    fields = ('symbol','qty','entry_decision_time','entry_fill_time','entry_price',
                              'exit_fill_time','exit_price','gross_pnl','commission','net_pnl','exit_reason','mfe','mae','r_multiple')
                    economics = lambda ts: [{k:t.get(k) for k in fields} for t in ts]
                    assert economics(trades) == economics(read(prior/'trades.json'))
                cases[str(folder)] = sha(folder / 'completion.json')
    assert len(cases) == data['completed_cases']
    actual = {str(p.parent) for p in (e.ROOT / 'cases').rglob('completion.json')}
    assert actual == set(cases), 'Unexpected completed case outside the bounded protocol'
    return cases


def verify(chart_reviewed):
    e.configure()
    assert chart_reviewed, 'Inspect results.png and record delivery_review.json first'
    assert not (e.OUT / 'delivery_checks.json').exists(), 'Never rerun a successful completed verifier'
    harness = e.verify_frozen()
    receipt, review = read(e.OUT / 'report_receipt.json'), read(e.OUT / 'delivery_review.json')
    assert review['chart_visually_reviewed'] and review['report_text_reviewed']
    assert review['report_receipt_sha256'] == sha(e.OUT / 'report_receipt.json')
    for name, digest in receipt['files'].items():
        assert sha(e.OUT / name) == digest, name
    for name in ('report.html','report.txt','results.png'):
        assert review['files'][name] == sha(e.OUT / name)
    assert receipt['frozen_harness_sha256'] == sha(e.ROOT / 'frozen_harness.json')
    assert receipt['qualification_complete_sha256'] == sha(e.ROOT / 'qualification_complete.json')
    data, profile = read(e.OUT / 'results.json'), read(e.OUT / 'selected_research_policy.json')
    parent = e.PARENT_OUT / 'selected_research_policy.json'
    assert sha(parent) == 'aa2d612718023b915e1b87f65ae9f3a8cf5575edaf30e564b4aadbd9ece3c9a9'
    old = read(parent)
    assert profile['mutations'] == old['mutations'] and profile['source_generation_hashes'] == old['source_generation_hashes']
    assert profile['selected_id'] == data['selected_id'] == 'c4f49be01ed09b8dc'
    assert not profile['promotion'] and not profile['structural_migration_adopted'] and not data['promotion']
    assert not profile['independent_validation'] and not data['independent_validation']
    assert profile['results_sha256'] == sha(e.OUT / 'results.json')
    assert read(e.OUT / 'results_collection_receipt.json')['collector_sha256'] == sha(REPO / 'scripts/olr_structural_results_v6.py')
    done = read(e.ROOT / 'qualification_complete.json')
    assert done['training_decision_sha256'] == sha(e.OUT / 'training_decision.json')
    gates = verify_training(data, read(e.ROOT / 'registration.json')['prior_gates'])
    cases = verify_saved_accounting(data)
    # Recollect only this new saved-output study. No source materialization,
    # old collector, native replay, or completed verifier is invoked.
    assert results.collect() == data
    assert feedback.collect() == read(e.OUT / 'admission_feedback.json')
    manifest = read(e.OUT / 'review_bundle/source_manifest.json')
    for name, value in manifest['files'].items():
        assert sha(e.OUT / 'review_bundle' / name) == value['sha256']
        assert sha(value['source']) == value['sha256']
    tests = read(e.ROOT / 'isolated_test_native_imports.json')
    assert tests['exit_code'] == 0 and tests['adversarial_import_checks'] == 3
    ready = read(e.t.ENGINES / 'train/ready.json')
    for name, digest in tests['imports'].items():
        assert ready['manifest']['files'][name] == sha(Path(ready['files'])/name) == digest
    assert harness['tests'] == dict(tests=64, failures=0, errors=0, skipped=0)
    write(e.OUT / 'delivery_checks.json', dict(utc=utc(), passed=True, cases=cases,
        result_sha256=sha(e.OUT / 'results.json'), profile_sha256=sha(e.OUT / 'selected_research_policy.json'),
        report_receipt_sha256=sha(e.OUT / 'report_receipt.json'), review_sha256=sha(e.OUT / 'delivery_review.json'),
        verifier_sha256=sha(__file__), exact_new_saved_output_recollection=True,
        independent_training_gate_arithmetic=gates, saved_native_accounting_verified=True,
        chart_reviewed=True, selected_id=profile['selected_id'], promotion=False))
    print('Structural foundation delivery verified; broader alpha/frequency objective remains active.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chart-reviewed', action='store_true')
    verify(parser.parse_args().chart_reviewed)

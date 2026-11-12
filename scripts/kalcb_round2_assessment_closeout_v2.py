"""Verify C16 completed measurements and explicitly preserve uncomputed scope."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import sys
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_round2_assessment_v1 import read, write, bound, digest

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('E:/CodexResearch/k_stock_trader/kalcb_round2_assessment_20261017/v1')


def main():
    originals = []
    for name in ('kalcb-exp-0041.md', 'kalcb-exp-0041-analysis-binding.md', 'kalcb-exp-0041-capital-binding.md'):
        current = REPO/'docs/optimisation/kalcb/experiments'/name
        saved = ROOT/'registered_documents'/name
        assert digest(current) == digest(saved)
        originals.append(dict(current=bound(current), original=bound(saved)))
    outputs = []
    for directory in ('analysis', 'candidate_evidence_v2', 'anatomy', 'minute_source_audit',
                      'minute_marks', 'support', 'exposure_v2', 'figures',
                      'minute_anatomy', 'reservations_v2', 'decision_atr_v1', 'native_rank_v1',
                      'noise_r2b', 'capital_summary', 'capital_minute', 'journal_support'):
        path = ROOT/directory/'completion.json'
        result = read(path); assert result['complete'], directory
        outputs.append(bound(path))
    scaling=list(csv.DictReader((ROOT/'score_scaling/cells.csv').open()))
    assert len(scaling)==8
    outputs.extend(bound(ROOT/'score_scaling'/name) for name in ('cells.csv','baseline_score.json','observed_minute_reference.json','binding.json'))
    assert read(ROOT/'capital_summary/display_completion.json')['complete']
    assert read(ROOT/'score_checks_v4.json')['passed']
    assert digest(REPO/'scripts/kalcb_round2_score_checks_v4.py') == read(ROOT/'score_checks_v4.json')['program_sha256']
    matrix = read(ROOT/'noise_r2b/completion.json')
    assert matrix['total_executions_including_original_pilot_and_preserved_failure'] == 27
    assert len(matrix['cases']) == 23 and all(c['qualified'] and not c['guard'] and c['exit_code']==0 for c in matrix['cases'])
    assert not read(ROOT/'capital_summary/completion.json')['unqualified_cells']
    minute = list(csv.DictReader((ROOT/'capital_minute/summary.csv').open()))
    assert len(minute)==16 and all(float(r['critical_max_staleness_minutes'])==0 for r in minute)
    assert all(float(r['max_native_nav_error_krw'])<.02 for r in minute)
    assert not any(r['observed_hard_limit_breach']=='True' for r in minute)
    processes = []; resources = []
    for directory in ('noise', 'noise_r2b'):
        for path in sorted((ROOT/directory).glob('*_launch.json')):
            record = read(path); pid=record['pid']; created=record['created']
            alive = psutil.pid_exists(pid) and abs(psutil.Process(pid).create_time()-created)<.01
            assert not alive, (pid, path)
            processes.append(dict(receipt=bound(path), original_process_active=False))
        resources.extend(read(path) for path in (ROOT/directory).glob('*_process.json'))
    pilot=read(ROOT/'pilot_process.json')
    alive = psutil.pid_exists(pilot['pid']) and abs(psutil.Process(pilot['pid']).create_time()-pilot['create_time'])<.01
    assert not alive
    assert len(processes)==26 and len(resources)==26
    write(ROOT/'closeout.json', dict(verified_at_utc=datetime.now(timezone.utc).isoformat(),
        program=bound(__file__), frozen_originals_unchanged=originals, completed_measurement_receipts=outputs,
        exact_original_repeats=8, exact_capital_adapter_bridges=2, qualified_capital_accounts=16,
        preserved_unverified_capital_generation=1, total_native_executions=27,
        monitored_accounts=26, monitored_wall_seconds=sum(x['seconds'] for x in resources),
        maximum_monitored_account_seconds=max(x['seconds'] for x in resources),
        maximum_monitored_rss_bytes=max(x['peak_rss'] for x in resources),
        guard_breaches=[x for x in resources if x['guard']], known_native_processes=processes,
        pilot_process_active=False, all_native_producers_exited=True,
        finer_capital_max_observed_dd_pct=max(float(x['retained_minute_dd_pct']) for x in minute),
        finer_source_complete=False, new_treatment_accounts=0, new_component_ablation_accounts=0,
        native_oracle_accounts=0, matched_random_entry_accounts=0,
        current_source_rejected_candidate_native_labels_complete=False,
        all_requested_measurements_complete=False,
        scope='Evidence-cited assessment delivered with explicit missing native-label/oracle/placebo measurements. No economic promotion or stopped-study restart.'))


if __name__=='__main__': main()

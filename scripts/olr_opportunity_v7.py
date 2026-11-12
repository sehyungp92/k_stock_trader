"""Training-only opportunity inventory, before any payoff or policy fitting."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
BASE = Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010')
PARENT = BASE / 'minute_exit_v5/clockfix_generation_1'
ROOT = BASE / 'structural_v6/opportunity_v1'
OUT = REPO / 'data/backtests/analysis/olr_structural_20261012/opportunity_v1'
BLOCKS = [('2025-08-07', '2025-10-31'), ('2025-11-01', '2025-12-31'), ('2026-01-01', '2026-03-31')]
SCORE_REASONS = frozenset(('afternoon_score_below_floor', 'afternoon_score_above_cap',
                         'afternoon_score_in_reject_band', 'afternoon_score_band_rule_miss'))


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(p, value):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, indent=2, default=str, allow_nan=False)
        f.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def cohort(selected, morning, context, hard_reasons, final_reasons):
    if selected:
        if not morning or not context or hard_reasons or final_reasons:
            raise ValueError('Selected candidate has a failed upstream gate')
        return 'incumbent'
    if morning and context and not hard_reasons and final_reasons and set(final_reasons) <= SCORE_REASONS:
        return 'score_mapping_only'
    return 'other_exclusion'


def summarize(rows, sessions):
    scored = [r for r in rows if BLOCKS[0][0] <= r['day'] <= BLOCKS[-1][1]]
    extra = [r for r in scored if r['cohort'] == 'score_mapping_only']
    selected_days = {r['day'] for r in scored if r['cohort'] == 'incumbent'}
    blocks = []
    for lo, hi in BLOCKS:
        sample = [r for r in extra if lo <= r['day'] <= hi]
        blocks.append(dict(start=lo, end=hi, candidates=len(sample), days=len({r['day'] for r in sample})))
    empty_addition_days = {r['day'] for r in extra} - selected_days
    checks = dict(at_least_60_candidates=len(extra) >= 60,
                  at_least_30_days=len({r['day'] for r in extra}) >= 30,
                  every_block_10_candidates_5_days=all(b['candidates'] >= 10 and b['days'] >= 5 for b in blocks),
                  at_least_10_previously_empty_days=len(empty_addition_days) >= 10)
    return dict(rows=len(scored), source_sessions=len(sessions),
        cohort_counts=dict(Counter(r['cohort'] for r in scored)),
        selected_days=len(selected_days), additional_supply_days=len({r['day'] for r in extra}),
        previously_empty_supply_days=len(empty_addition_days), blocks=blocks, support_checks=checks,
        payoff_study_eligible=all(checks.values()), portfolio_experiment_eligible=False,
        pnl_consumed=False, independent_validation=False)


def register():
    ready_path = PARENT / 'native_engines/train/ready.json'
    ready = read(ready_path)
    files = Path(ready['files'])
    snapshot = files / ready['study']['olr_snapshots']
    receipt = snapshot.with_suffix('.receipt.json')
    assert sha(snapshot) == ready['manifest']['files'][ready['study']['olr_snapshots']]
    cache = Path(read(receipt)['acceleration_profile']['cache'])
    prior = REPO / 'data/backtests/analysis/olr_structural_20261012/native_controlled'
    assert read(prior / 'delivery_checks.json')['passed']
    assert sha(prior / 'results.json') == '23073baaaeddf521a3a443999d54c7f3ddd87962ba137a0aebef67d3e382c0e4'
    names = ['scripts/olr_opportunity_v7.py', 'scripts/olr_opportunity_worker_v7.py',
             'tests/strategy_olr/test_opportunity_v7.py', 'scripts/olr_structural_native_test_isolated_v6.py',
             'scripts/olr_snapshot_streaming.py', 'scripts/olr_snapshot_acceleration.py']
    bindings = [ready_path, Path(ready['study']['manifest']), snapshot, receipt, cache / 'index.json',
                PARENT / 'candidate_plan.json', prior / 'delivery_checks.json', prior / 'admission_feedback.json',
                ROOT / 'tests.xml', ROOT / 'tests.log']
    protocol = dict(version='olr-opportunity-inventory-v7', utc=utc(),
        rationale='Foundation return loss was dominated by changed quantities. It added entries but no entry days. Establish opportunity supply and later fixed-quantity net payoff separately before recalibrating selection or allocation.',
        hypothesis='The discontinuous scalar afternoon score map may exclude otherwise source-complete, daily-qualified recovery opportunities. This tests supply only, not whether excluded opportunities have alpha.',
        scope=dict(window='train', first_day=BLOCKS[0][0], last_day=BLOCKS[-1][1], blocks=BLOCKS,
            universe='Original fixed 103 names', max_rows=103*217, source_reconstruction=False,
            cohort='Original morning selection and every original non-score afternoon guard must pass. Only scalar score limits/bands may exclude the shadow cohort. No sector-participation relaxation, late-confirmation extension, F rescue or new feature threshold.'),
        source=dict(ready=str(ready_path), cache=str(cache), snapshot=str(snapshot),
            source_hash=ready['study']['source_generation_hash'],
            reuse='Authenticate original cache index, parent receipt, every payload before unpickling, every consumed source hash and every imported frozen native module. Recompute original selections and require exact candidates and rejection metadata.'),
        design=['Record all 103 symbol-days, including missing-source and upstream exclusions.',
                'Recompute native morning and afternoon decisions from authenticated cached native features; no account outcomes are read.',
                'Trim five-minute feature input at exchange-adjusted completed 14:30 cutoff and verify parity against the full cached day.',
                'Preserve all original threshold/rank/feature values; record score mapping exclusion separately from hard guards.',
                'Count date support and previously empty days in three fixed chronological blocks.',
                'If support passes, freeze a separate fixed-opportunity, fixed-notional causal payoff protocol before reading outcomes. More candidates alone cannot launch a portfolio experiment.'],
        advancement=dict(candidates_min=60, days_min=30, each_block_candidates_min=10,
                         each_block_days_min=5, previously_empty_days_min=10, no_retuning=True),
        future_payoff_requirements=['Native feasible entry/exit timing, prices, costs and capacity; unsupported/unfilled labels remain missing, never zero.',
            'Preserve original confirmation and exit horizons; record overlapping opportunities without claiming simultaneous funding.',
            'Use date-grouped chronological evaluation and net-of-stress outcomes; training dates must precede label maturity.',
            'An interpretable selection/size mapping needs genuine incremental expectancy and winner retention evidence; no symbol/sector blacklists or reused F family.',
            'Any account comparison still requires all original and matched-control return/DD/frequency/notional/stress/capped gates.'],
        files={n:sha(REPO/n) for n in names}, inputs={str(p):sha(p) for p in bindings},
        selected_id='c4f49be01ed09b8dc', promotion=False, live_activation=False,
        independent_validation=False, oos_consumed=False)
    write(ROOT / 'registration.json', protocol)
    write(OUT / 'registration.json', protocol)
    print(dict(registration=str(ROOT / 'registration.json'), sha256=sha(ROOT / 'registration.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['register'])
    register()

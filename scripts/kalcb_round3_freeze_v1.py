"""Immutable C16 control/score capsule. No policy evaluation or optimisation."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

REPO = Path(__file__).resolve().parents[1]
BASE = Path('E:/CodexResearch/k_stock_trader')
STAGE1 = BASE/'kalcb_round2_assessment_20261017/v1'
SOURCE = BASE/'kalcb_opportunity_lifecycle_20261016/funded_interaction_v1'
ROOT = BASE/'kalcb_round3_20261017/attempt_1'
SCENARIOS = ('baseline', 'participation', 'cost', 'combined')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def bound(path): return dict(path=str(Path(path).resolve()), sha256=digest(path))
def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f: json.dump(value, f, indent=2, allow_nan=False)
def signed(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    started=time.monotonic(); ROOT.mkdir(parents=True, exist_ok=False)
    write(ROOT/'freeze_started.json', dict(at_utc=datetime.now(timezone.utc).isoformat(), program=bound(__file__), max_workers=2))
    documents={}
    for p in [REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0042.md',
              REPO/'docs/KALCB-round2-assessment-2026-10-17.md',
              REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0041.md',
              REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0041-analysis-binding.md',
              REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0041-capital-binding.md',
              REPO/'scripts/kalcb_round2_score_checks_v4.py',
              Path('C:/Users/sehyu/Documents/Other/Projects/trading/docs/strategy-implementation-lessons.md')]:
        dest=ROOT/'freeze/documents'/p.name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p,dest)
        assert digest(p)==digest(dest); documents[p.name]=dict(original=bound(p), saved=bound(dest))
    score=dict(version='C16-unchanged', component_count=3,
        R='Net return percentage points; separate flat-start periods weighted .5 each, scenarios .25 each',
        D='Same weights on max(0, MTM drawdown percent minus target)',
        U='Evidence-qualified incremental useful issuer-days; tie-break only within .10 scalar points',
        scalar='R - 0.25 * D', target_drawdown_pct=dict(baseline=8, participation=10, cost=10, combined=10),
        hard_drawdown_pct=dict(baseline=12, participation=15, cost=15, combined=15),
        useful_material_lifetime_fraction=.01, useful_issuer_day_profit_fraction=.0001,
        useful_lifetime_identity='Collapse physical fragments/duplicates; retain all issuer-day losses',
        useful_evidence='Prospectively defined addition group; positive expected-value and paired funded added-minus-displaced lower bound',
        scales=read(STAGE1/'score_scaling/baseline_score.json'),
        display_scale_floor=dict(return_pp=1, drawdown_pp=1, useful_days=1),
        baseline_capital=100000000, qualification_capitals=[95000000,100000000,105000000],
        scenario_weights={s:.25 for s in SCENARIOS}, period_weights=dict(is_156=.5, reused_oos_69=.5),
        scales_file=bound(STAGE1/'score_scaling/cells.csv'), synthetic_checks=bound(STAGE1/'score_checks_v4.json'))
    qualification=dict(score_version='C16-unchanged', native_source_and_account_acceptance_required=True,
        failed_missing_stale_receipts='ineligible; never ordinary zero score or fallback control',
        flat_intraday_and_terminal=True, minimum_gain_pp=1,
        paired_lcb95_strictly_positive=True, gain_at_least_family_adjusted_paired_MDE=True,
        familywise_alpha=.05, planning_power=.8, maximum_selectable_alternatives=8,
        multiplicity='Bonferroni eight-alternative family; historical adaptive search not erased',
        max_IS_loss_pp=2, max_scenario_loss_pp=1, predicted_mechanism_signature_required=True,
        ablations_neighbourhood_delay_cost_capital_required=True,
        frequency_delivery=dict(min_net_useful_oos_days=3,min_net_useful_is_days=0,positive_added_minus_displaced_value=True),
        hard_risk_all_cells=True, finer_valid_breach_fails=True, finer_coverage_gaps_disclosed=True,
        diagnostic_removals_selectable=False, live_activation=False, untouched_holdout_available=False)
    evaluation=dict(calendar=dict(is_156=['2025-08-07','2026-03-31'],reused_oos_69=['2026-04-01','2026-07-10']),
        full_calendar_no_trade_days=True, blocks=[5,10,20], bootstrap_draws=4000, stream_seed='41017 + block_length + calendar_sessions',
        paired_common_draws_across_scenarios=True, periods_resampled_separately=True,
        capitals_and_scenarios_are_not_independent_samples=True, max_workers=2, nested_native_maximum=2,
        numerical_threads=1, maximum_native_account_hours=200, maximum_full_equivalent_accounts=512,
        per_worker_seconds=1200, per_worker_rss_bytes=2*1024**3,
        preparation_reporting_cost_included=True, frequency='native lifetimes and material profitable issuer-days; fragments excluded')
    oracle=dict(version='C16-specification-unchanged', objective='Terminal after-cost own-account equity',
        S=dict(fixed='Native sizing, management, exits and execution', hindsight='Admission/ordering in explicitly bound mechanically eligible opportunity universe'),
        E=dict(fixed='Native entry rule, sizing/risk and execution', hindsight='Executable exit submission clocks/quantities; future state evolves'),
        SE=dict(fixed='Mandate, native constraints, costs and execution chronology', hindsight='Admission/order and executable exit decisions'),
        all_fixed=['initial capital','2x funding','cash and reservations','position/concentration/heat limits','native risk sizing','integer units','liquidity','costs','execution chronology'],
        incumbent='Accepted baseline feasible for every oracle problem',
        classification='Exact/certified upper bound/approximate feasible result must remain distinct; a heuristic is a lower bound on optimum',
        future_information='Oracle-only context; never causal decision or time-valid training input',
        labels=dict(horizon_minutes=[30,60,120],maturity='Same session/deadline and period; no cross-partition maturity',missing='Explicit censoring, never zero'),
        relaxed=dict(universe='Actual selected baseline lifetimes',quantity='Original full quantity/entry VWAP',constraints_relaxed=['funding','occupancy','liquidity','partial-fill timing','endogenous sizing']),
        stopped_studies_excluded=['exp0031 per-opening omission-fork enumeration','exp0039/C15 new-event reconsideration'],
        native_measurement_binding_required=True)
    for name,value in [('score',score),('qualification',qualification),('evaluation',evaluation),('oracle_spec',oracle)]:write(ROOT/'freeze'/f'{name}.json',value)
    baseprofiles=read(SOURCE/'registration.json')['profiles']; accounts=[]; profiles={}
    journal_files=('result.json','execution_audit.json','fills.json','trades.json','orders.jsonl.gz','decisions.jsonl.gz','equity.json','daily.json','horizon_actions.json')
    for period in ('is','oos'):
        for scenario in SCENARIOS:
            for capital in (95000000,100000000,105000000):
                folder=SOURCE/'cases'/f'combined_{period}_{scenario}' if capital==100000000 else STAGE1/'noise_r2b/cases'/f'capital_{period}_{scenario}_{capital}'
                profile=baseprofiles[period+'_cap2'] if capital==100000000 else read(folder/'input_profile.json')
                profiles[period+'_'+str(capital)]=profile
                result=read(folder/'result.json'); assert result['audit_status']=='passed' and not result['terminal_inventory'] and not result['terminal_pending'] and not result['residual_sessions']
                files={}; target=ROOT/'freeze/accounts'/period/scenario/str(capital); target.mkdir(parents=True)
                for name in journal_files:
                    src=folder/name; dst=target/name; shutil.copy2(src,dst); assert digest(src)==digest(dst)
                    files[name]=dict(original=bound(src),saved=bound(dst))
                accounts.append(dict(period=period,scenario=scenario,initial_equity=capital,profile=profile,files=files))
    # Authenticate immutable input bytes; one pool of two, no nested executor.
    inputs={}
    for name,profile in profiles.items():
        for key in ('manifest','plan','policy'):assert digest(profile[key]['path'])==profile[key]['sha256']
        manifest=read(profile['manifest']['path']);code=Path(profile['manifest']['path']).parent/'files'
        for rel,sha in manifest['files'].items():
            path=code/rel
            if str(path) in inputs:assert inputs[str(path)]==sha
            inputs[str(path)]=sha
    def check(item):
        path,sha=item
        if digest(path)!=sha:raise ValueError('Changed baseline input '+path)
        return dict(path=path,sha256=sha)
    with ThreadPoolExecutor(max_workers=2) as pool: verified=list(pool.map(check,inputs.items()))
    optimized=REPO/'data/backtests/output/kalcb/round_2/optimized_config.json'
    shutil.copy2(optimized,ROOT/'freeze/optimized_config_round2.json')
    expected=read(optimized)['mutations']
    for period in ('is','oos'):
        plan=read(baseprofiles[period+'_cap2']['plan']['path']);assert read(plan['config'])==expected
    write(ROOT/'freeze/source_inputs.json',verified)
    capsule=dict(version='kalcb-round3-c16-freeze-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        documents=documents,profiles=profiles,accounts=accounts,source_inputs=bound(ROOT/'freeze/source_inputs.json'),
        optimized_config=bound(ROOT/'freeze/optimized_config_round2.json'),
        contracts={n:bound(ROOT/'freeze'/f'{n}.json') for n in ('score','qualification','evaluation','oracle_spec')},
        stage1_closeout=bound(STAGE1/'closeout.json'),stage1_review=bound(STAGE1/'final_review.json'),
        no_treatment_evaluated=True,source_input_files=len(verified),seconds=time.monotonic()-started,max_workers=2)
    write(ROOT/'freeze/manifest.json',{**capsule,'manifest_sha256':signed(capsule)})
    write(ROOT/'freeze_complete.json',dict(complete=True,manifest=bound(ROOT/'freeze/manifest.json'),seconds=time.monotonic()-started,
        accounts=24,source_input_files=len(verified),new_native_accounts=0,max_workers=2))
    print(json.dumps(read(ROOT/'freeze_complete.json')),flush=True)


if __name__=='__main__': main()

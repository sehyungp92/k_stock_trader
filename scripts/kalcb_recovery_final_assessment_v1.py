"""Close the finite recovery study from completed, immutable qualification evidence."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

REPO = Path(__file__).resolve().parents[1]
R = Path('C:/Users/sehyu/.codex/research/kalcb_full_oos_recovery_20261009/v3')
D = Path('E:/CodexResearch/k_stock_trader/kalcb_recovery_final_20261010/v2')
Q = Path('E:/CodexResearch/k_stock_trader/kalcb_recovery_qualification_20261010/v4')
F = Path('E:/CodexResearch/k_stock_trader/kalcb_forecast_qualification_20261010/v1')
S = Path('E:/CodexResearch/k_stock_trader/kalcb_atr_intervention_20261010/v2')
OUT = R/'final_assessment_20261010_v1'
NAMES = ('expanding', 'expanding_atr5', 'decay60', 'decay60_atr5')
SCENARIOS = ('baseline', 'participation', 'cost', 'combined')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bound(path):
    return {'path': str(Path(path).resolve()), 'sha256': digest(path)}


def verify(reference):
    assert digest(reference['path']) == reference['sha256'], reference['path']


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def main():
    assert not OUT.exists(), 'Preserve existing assessment'
    completion = read(D/'completion.json')
    assert completion['complete'] and len(completion['profiles']) == 8
    verify(completion['preparation'])
    for reference in read(D/'protocol.json')['bindings']:
        verify(reference)
    qualification = read(Q/'completion.json')
    expected = {(n, p, s) for n in NAMES for p in ('is', 'oos') for s in SCENARIOS}
    assert qualification['complete'] and len(qualification['results']) == 32
    assert {(x['name'], x['period'], x['scenario']) for x in qualification['results']} == expected
    assert all(x['metrics']['audit_status'] == 'passed' for x in qualification['results'])
    for row in qualification['results']:
        verify(row['result'])
    forecast = read(F/'forecast_reconstruction.json')
    assert forecast['passed'] and forecast['prediction_comparisons'] == 174096 and forecast['max_abs_error'] == 0
    assert read(R/'boundary_continuity.json')['passed']
    attribution = read(D/'attribution.json')['profiles']
    trials = read(D/'all_trials.json')
    assert len(trials['records']) == trials['trial_entries_including_aliases'] == 61
    comparisons = {}
    reports = []
    for item in completion['profiles']:
        assert item['complete']
        for kind in ('report', 'text', 'integrity'):
            verify(item[kind])
        folder = Path(item['report']['path']).parent
        integrity = read(item['integrity']['path'])
        assert integrity['passed'] and integrity['native_evidence_unchanged']
        report = read(item['report']['path'])
        profile = read(folder/'selected_research_profile.json')
        name, period = profile['primary_profile'], profile['recovery_report_context']['period']
        supplement = report['diagnostics_summary']['qualified_native_replay']
        assert supplement['recovery_context']['independent_validation'] is False
        assert supplement['date_range'] == profile['date_range']
        assert not supplement['production_enabled']
        comparisons[name+'/'+period] = {}
        for scenario in SCENARIOS:
            native = read(profile['result_paths'][scenario])
            assert supplement['scenarios'][scenario] == {k: native[k] for k in supplement['scenarios'][scenario]}
            assert attribution[name+'/'+period][scenario]['metrics'] == native
            comparisons[name+'/'+period][scenario] = {k: native[k] for k in ('return_pct', 'drawdown_pct', 'trades', 'active_days')}
        for metric, native_metric in [('net_return_pct', 'return_pct'), ('max_drawdown_pct', 'drawdown_pct')]:
            assert abs(report['final_metrics'][metric]*100 - comparisons[name+'/'+period]['baseline'][native_metric]) < 1e-8
        reports.append({'name': name, 'period': period, **item,
                        'profile': bound(folder/'selected_research_profile.json'),
                        'config': bound(folder/'strategy_config.json'),
                        'round_manifest': bound(folder.parent/'rounds_manifest.json') if (folder.parent/'rounds_manifest.json').exists() else None})
    canonical = read(REPO/'config/research/kalcb_intraday_advantage_20261007/selected_research.json')
    for period in ('is', 'oos'):
        comparisons['canonical/'+period] = {}
        for scenario in SCENARIOS:
            path = canonical['result_paths'][scenario] if period == 'is' else R/'control'/scenario/'result.json'
            native = read(path)
            comparisons['canonical/'+period][scenario] = {k: native[k] for k in ('return_pct', 'drawdown_pct', 'trades', 'active_days')}
    canonical_config = REPO/'config/research/kalcb_intraday_advantage_20261007/selected_config.json'
    assert digest(canonical_config) == '400f67719926e0e36f246cbeea60b930355ea262a47e04153d2d8099b8a758a1'
    deltas = {}
    for name in NAMES:
        deltas[name] = {period: {scenario: {metric: comparisons[name+'/'+period][scenario][metric] - comparisons['canonical/'+period][scenario][metric]
                                          for metric in comparisons[name+'/'+period][scenario]}
                                 for scenario in SCENARIOS} for period in ('is', 'oos')}
    support = [{k: row[k] for k in ('name', 'period', 'scenario')} | {
        'issuer_days': row['summary']['direct_feasible_issuer_days'],
        'sessions': row['summary']['direct_feasible_sessions'],
        'passed': row['summary']['preferred_support_passed']} for row in read(S/'completion.json')['results']]
    assert len(support) == 8 and all(x['passed'] for x in support)
    assessment = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'finite_study_complete': True, 'robust_positive_expectancy_demonstrated': False,
        'canonical_replaced': False, 'live_enabled': False,
        'verdict': 'Material measured OOS loss/drawdown reduction; no demonstrated joint optimum in expected return, trading frequency and IS preservation. Preserve qualified alternatives as research comparisons; do not promote a robust winner.',
        'periods': {'is': ['2025-08-07', '2026-03-31', 156], 'oos': ['2026-04-01', '2026-07-10', 69]},
        'scope': 'OOS is the original chronological split, now repeatedly reused development. Separate KRW100m accounts; continuous causal history and original model clock. All comparisons descriptive, selection-adjusted uncertainty unavailable.',
        'risk_preferences': {'baseline_dd_pct': 8, 'combined_dd_pct': 10, 'hard_cutoffs': False},
        'trial_entries_including_controls_and_aliases': 61,
        'metrics': comparisons, 'deltas_to_canonical': deltas, 'reports': reports,
        'direct_atr_support': support,
        'research_roles': {'expanding': 'Exact IS preservation and simplest continuous-clock reference; materially lower OOS frequency.',
                           'expanding_atr5': 'Preferred balanced research reference when preserving IS is primary; still negative OOS and lower frequency.',
                           'decay60': 'Highest baseline OOS return among four qualified profiles, barely positive and cost-fragile, with substantial IS sacrifice.',
                           'decay60_atr5': 'Preferred activity/stressed-loss research reference; near-canonical OOS trade count and much lower DD, substantial IS sacrifice, negative OOS.'},
        'canonical_config': bound(canonical_config),
        'evidence': [bound(p) for p in [Path(__file__), D/'completion.json', D/'attribution.json', D/'all_trials.json',
            D/'account_verification.json', D/'input_verification.json', Q/'completion.json', F/'forecast_reconstruction.json',
            S/'completion.json', R/'boundary_continuity.json', R/'history_boundary_audit.json', R/'alignment_outcome_review_v1.json',
            *[R/f'stage{i}_review.json' for i in (2, 3, 4, 5)], REPO/'docs/strategy-implementation-lessons.md']],
    }
    OUT.mkdir()
    write(OUT/'assessment.json', assessment)
    lines = [
        'KALCB recovery study: final assessment, 10 October 2026', '',
        'The finite ablation, perturbation and targeted recovery study is complete. All 32 finalist source/account audits and all eight existing-framework final diagnostic reports passed. The experiments materially reduced measured OOS losses and drawdown. They did not establish a robust positive OOS return, a joint return/frequency improvement that preserves IS, or a unique optimum. This is a completed research study with an unfulfilled robust-performance objective; it is not a claim of optimization success.', '',
        'Canonical round 1 remains the frozen reference. No replacement is promoted and live trading is not enabled. The 8% baseline / 10% combined drawdown preferences were never hard rejection cutoffs. Every valid improvement, neighborhood, backout, rejected trial and alias remains in the evidence, including results above those preferences.', '',
        'IS covers 7 August 2025 to 31 March 2026 (156 sessions). OOS covers the complete 1 April to 10 July 2026 period (69 sessions). OOS now means the original chronological split: it was repeatedly reused for development and is not independent validation. Each period starts a separate KRW100m account; market/model/calibration history remains continuous. No compounded IS+OOS headline is presented.', '',
        'The verified comparison follows. Return and maximum bar-level MTM drawdown are percentages; trades are completed native trades. Expanding means the original target with continuous expanding-history refits. Decay60 keeps all mature history but weights observations with a 60-session half-life. ATR5 denotes the registered ATR-dependent quantity adjustment, not a 5% portfolio drawdown cap.', '',
    ]
    for period in ('is', 'oos'):
        lines += [period.upper()+' baseline and combined stress',
                  'Profile | Baseline return | Baseline DD | Trades | Combined return | Combined DD | Trades']
        for name in ('canonical', *NAMES):
            b, c = [comparisons[name+'/'+period][s] for s in ('baseline', 'combined')]
            lines.append(f"{name} | {b['return_pct']:.4f}% | {b['drawdown_pct']:.4f}% | {b['trades']} | {c['return_pct']:.4f}% | {c['drawdown_pct']:.4f}% | {c['trades']}")
        lines.append('')
    lines += [
        'Continuous expanding history preserves the canonical IS account exactly, while OOS return improves by 11.5968 percentage points in baseline and 12.2927 points in combined stress. However, OOS trades fall from 290/300 to 211/220 and active sessions from 66 to 59. This establishes a useful historical loss reduction, with a meaningful activity tradeoff.', '',
        'Expanding ATR5 is the more balanced research starting point when preserving IS is important. OOS improves to -1.0389% / 11.4047% DD in baseline and -9.7968% / 13.8435% DD under combined stress. Its IS baseline return falls by 6.5184 points and combined IS by 7.7394 points. OOS trade counts of 219/225 remain well below the canonical 290/300. The IS sacrifice and reduced activity must remain visible.', '',
        'Decay60 ATR5 is the higher-activity research alternative. Baseline OOS improves by 14.7307 points and DD falls by 8.1753 points; combined return improves by 16.6858 points and DD falls by 13.3289 points. OOS trades are 282/293, close to the canonical 290/300, with 62 rather than 66 active sessions. IS return falls from 73.1303% to 50.9255% baseline and 47.8121% to 30.9287% combined. That deterioration is substantial. Compared with unscaled decay60, ATR5 improves IS return/DD and combined OOS return/DD/activity, but baseline OOS return declines from +0.6747% to -0.8296%. Neither profile dominates the other.', '',
        'Trading frequency per OOS session is approximately 4.20/4.35 for canonical baseline/combined, 3.06/3.19 for expanding, 3.17/3.26 for expanding ATR5, 3.93/4.07 for decay60 and 4.09/4.25 for decay60 ATR5. Meaningful trading continues, but greater frequency has not been shown to create greater expected profit. The ATR4 neighborhood produces 299/301 OOS trades and 9.4977% baseline DD, yet returns -3.0336%/-11.8099% and sacrifices IS; it is retained rather than discarded for missing a target.', '',
        'The history boundary is not the remaining explanation for weak OOS. Available intraday history starts 12 May 2025; older daily data starts 19 February 2024 and supports historical predictors. The first scored fit uses outcomes through 5 August, with 6 August embargoed, giving the 7 August IS start after 60 completed training sessions. Pre-August labels are used. The first OOS decision has 217 preceding intraday sessions and inherited daily/intraday feature history. The 305 direct feature comparisons passed. The continuous recipe inherits the 10 March fit on 1 April and refits on 7 April, rather than restarting its clock at the scoring boundary. More genuinely older intraday outcomes might improve estimation, but older daily prices cannot manufacture missing intraday execution labels.', '',
        'Forecast error and cost resilience remain the main measured economic weaknesses. In the common selected-management isolated-action diagnostic, original-target admitted core opening actions average +22.8295 bps in IS against a +13.6059 bps forecast; OOS averages -0.4564 bps against +23.3098 bps. Relabeling against selected management does not solve this: OOS actual is -1.8428 bps against +23.2755 bps forecast. The 18 selected-target checkpoint actions average -23.3318 bps against +21.4659 bps forecast. These are isolated fixed-KRW100m action statistics, not funded portfolio returns or independent samples. They show large conditional forecast errors; they do not justify selecting symbols or dates after the event.', '',
        'Cost attribution explains why the one slightly positive baseline is insufficient. Decay60 earns KRW17.2078m gross after slippage in OOS, then pays KRW16.5331m commissions/tax, leaving only KRW0.6747m. Its adverse slippage is another KRW4.5048m already embedded in that gross amount; subtracting it again would double count. Combined stress gives KRW4.2939m gross after slippage and KRW15.5436m commissions/tax, leaving -KRW11.2496m. Decay60 ATR5 earns KRW12.4001m gross after slippage but pays KRW13.2297m commissions/tax; combined stress earns KRW3.1197m and pays KRW12.7140m. These are realized path attributions, not a cost-free counterfactual portfolio. The four scenarios must be evaluated separately because participation changes the executed path.', '',
        'Exit attribution is diagnostic, not proof that deleting an exit fixes the strategy. Decay60 ATR5 baseline OOS has about -KRW37.44m from quick exits and -KRW18.70m from failed follow-through, partly offset by +KRW39.20m from end-of-day exits and other winners. Previous full-period backouts of quick exits and timing changes worsened results. A loss attached to an exit reason does not reveal the outcome of holding that losing position. Native ablations, rather than removing losses from a ledger, are the relevant evidence.', '',
        'The tested interventions go beyond the initial queue. The retained study contains 61 entries, including controls and aliases, spanning initial ablations/admission/refit tests, continuous-clock and sizing interactions, selected-management target reconstruction, expanding versus rolling120 versus decay60 learning, affine/residual calibration, 25/50/75% expanding/decay forecast blends, and ATR4/ATR5 neighborhoods. Selected-target value-only and all-target trials produced identical executed accounts and are explicitly aliases. Rolling120 deteriorated OOS, affine calibration damaged discrimination, and fixed blends were not a stable interpolation of their parents. Checkpoint residual adjustment produced too small and inconsistent an increment to prefer its extra mechanism. All outcomes remain available; this count does not mean 61 independent discoveries.', '',
        'There is direct support for the ATR quantity intervention, beyond altered scores or downstream compounding. Under the same decision state, timing, affordability, capacity and realized-volume limits, the baseline/combined OOS tests record 124/100 issuer-days across 53/50 sessions for expanding ATR5 and 172/141 issuer-days across 61/56 sessions for decay60 ATR5. All eight IS/OOS intervention cases pass their registered support preference and exactly reproduce native economic outputs. No hypothetical fills or profits are booked. This supports the mechanism having a real executable quantity effect; it does not establish robust profitability.', '',
        'Uncertainty and concentration prevent a robust-winner claim. For decay60 ATR5, descriptive five-day paired resampling gives an OOS relative-wealth improvement interval of [-8.7477%, +55.1521%] baseline and [-6.0807%, +63.6625%] combined, both including zero. Other 1/3/10-day block lengths and the other finalists also retain substantial uncertainty. These are relative-wealth intervals, not return-point intervals, and are unadjusted for adaptive selection. Removing the three largest incremental session contributions changes decay60 ATR5 baseline relative improvement to -1.6679%; combined remains only +1.7605%. Removing the largest winning issuer contribution leaves baseline PnL at about -KRW7.71m. These removals are concentration attribution only, not re-executed portfolios or proposed exclusion rules.', '',
        'All finalists have a positive May and negative April, June and partial July baseline returns. A historically strong broader market therefore does not establish that this particular after-entry intraday payoff and cost structure had positive conditional expectancy. The study does not provide a matched-factor decomposition against OLR and does not claim one. Reusing qualified market/history infrastructure and execution-integrity methods from the OLR work was appropriate; transferring its returns or selecting its winning dates would not prove KALCB alpha.', '',
        'Qualification binds actual source/config/policy/model/label/calibration lineage. All 32 four-scenario IS/OOS native replays pass source audit and independent fill, fee, inventory, cash and bar-level MTM reconciliation. All reference baseline/combined daily balances match. Saved-model verification covers 144 recipe/model checks over 120 unique models, reconstructs 174,096 forecasts with zero discrepancy, and matches 24 calibration folds using earlier out-of-fold observations. Late but valid cash outcomes remain recorded with their physical failure flags and enter training only after maturity. Invalid or censored labels are not converted to profitable/zero outcomes. The archived IS source contract retains outside-universe basis failures explicitly; its 103-symbol replay scope has none of those failures.', '',
        'The report job initially failed on a missing legacy single-preparation metadata field. The preserved v1 run was superseded by v2, which authenticates an explicit aggregate of all four actual feature preparations and the forecast receipt for report compatibility. It never changes native policy bytes or economic evidence. Twenty-one report compatibility/context/gate tests pass. The existing PhaseRunner/RoundManager generated eight complete research reports, each revalidating unchanged native evidence and matching return/DD/PF. This is source and research-account qualification, not certification that every possible issue or live execution risk has been eliminated. Paper order/fill/restart/latency qualification, genuinely untouched forward evidence and a funded fixed-capital replay remain outside the present claims.', '',
        'The appropriate decision is to retain expanding ATR5 and decay60 ATR5 as complementary qualified research references, with unscaled expanding and decay60 as required controls. Neither is promoted as an optimal or robust profitable strategy. Keeping the current canonical file unchanged does not endorse its OOS performance for deployment. The completed finite study supports loss reduction; it does not support promising that continued parameter search on the same 69 sessions will produce genuine expected-return recovery.', '',
        'For a subsequent research programme, the evidence points to conditional forecast reliability and cost-aware ranking as the next bottleneck, with causal earlier-fold calibration/shrinkage and executable net action outcomes evaluated before portfolio admission. Any extension should pre-register a small hypothesis set, retain the four controls, use prior-only internal folds for selection, and reserve genuinely uninspected future observations for confirmation. More OOS threshold sweeps, date/symbol exclusions, removing protective exits based on attribution, or relaxing costs/fill rules would not address the demonstrated problem. This is a recommendation for a distinct future programme, not unperformed work counted as completed here.', '',
        'All four native scenario results are included below; full precision is in assessment.json and the bound native files.', '',
        'Profile/period | Scenario | Return % | DD % | Trades | Active sessions',
    ]
    for name in ('canonical', *NAMES):
        for period in ('is', 'oos'):
            for scenario in SCENARIOS:
                m = comparisons[name+'/'+period][scenario]
                lines.append(f"{name}/{period} | {scenario} | {m['return_pct']:.6f} | {m['drawdown_pct']:.6f} | {m['trades']} | {m['active_days']}")
    lines += ['', 'Saved evidence and existing-script final diagnostic reports:', str(OUT/'assessment.json'),
              str(D/'all_trials.json'), str(D/'attribution.json'), str(Q/'completion.json'), str(F/'forecast_reconstruction.json')]
    for report in reports:
        lines.extend([report['name']+'/'+report['period']+': '+report['text']['path'], '  qualified profile: '+report['profile']['path']])
    with (OUT/'final_assessment.txt').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines)+'\n')
    write(OUT/'completion.json', {'complete': True, 'finite_study_complete': True,
        'robust_performance_objective_achieved': False, 'canonical_replaced': False, 'live_enabled': False,
        'report': bound(OUT/'final_assessment.txt'), 'assessment': bound(OUT/'assessment.json'),
        'diagnostics': bound(D/'completion.json'), 'program': bound(Path(__file__))})
    print(json.dumps({'complete': True, 'report': str(OUT/'final_assessment.txt'), 'reports_verified': len(reports), 'canonical_replaced': False}))


if __name__ == '__main__':
    main()

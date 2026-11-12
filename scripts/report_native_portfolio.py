"""Publish a reconciled implementation report from qualified frozen evidence."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.portfolio_native_budget_study import read, checked, SCENARIOS
from scripts.replay_selected_portfolio import verify_profile, binding
from scripts.analyze_native_portfolio import analyze, describe
from backtests.strategies.kalcb.executed_value import write_json


def metrics(row):
    return {k: row[k] for k in ('return_pct', 'drawdown_pct', 'trades', 'strategy_counts', 'audit_status')}


def report(profile_path, controls, robustness, output):
    profile = verify_profile(profile_path)
    if binding(controls / 'study.json') != profile['native_comparison_evidence']['reference_study']:
        raise ValueError('Report controls do not match the qualified profile')
    root = Path(profile['study']['path']).parent
    final = read(root / 'finalist.json')
    perturbation = read(robustness / 'registration.json')
    if (perturbation['parent_study_sha256'] != profile['study']['sha256']
            or perturbation['parent_finalist_sha256'] != profile['selection']['sha256']
            or perturbation['selected'] != final['winner']):
        raise ValueError('Robustness cases belong to a different selection')
    attribution = analyze(root, controls)
    naive = {s: describe(controls / 'cases' / ('naive_native_combination_' + s)) for s in SCENARIOS}
    risk = Path(read(root / 'registration.json')['parent_study']).parent
    base = read(controls / 'finalist.json')
    challenges = {}
    for label, path in (('planned_risk', risk), ('priority', root)):
        selected = read(path / 'finalist.json')
        names = {j['recipe'] for j in read(path / 'registration.json')['jobs']}
        challenges[label] = {'study': binding(path / 'study.json'),
            'selection': binding(path / 'finalist.json'), 'winner': selected['winner'],
            'results': {n: [metrics(r) for r in selected['results'][n]] for n in sorted(names)}}
    stress = {}
    for registration in sorted(robustness.glob('*/registration.json')):
        path = registration.parent
        for job in read(registration)['jobs']:
            name = job['recipe'] + '_' + job['scenario']
            case = path / 'cases' / name
            row = checked(case)
            stress[path.name + '/' + name] = {'initial_equity_krw': read(path / 'study.json')['initial_equity'],
                'metrics': metrics(row), 'completion': binding(case / 'completion.json'),
                'no_upward_compounding': job.get('no_upward_compounding', False)}
    if len(stress) != 14:
        raise ValueError('Expected all 14 registered robustness cases')
    incident = controls.parent / 'specification_hardlink_incident'
    recovery = read(incident / 'recovery.json')
    integrity = read(incident / 'full_manifest_reverification.json')
    if (not integrity['pass'] or recovery['manifest_or_registered_hashes_changed']
            or not all(row['verified_exact'] for row in recovery['repairs'])):
        raise ValueError('Frozen generation recovery has not been independently verified')
    tests_path = controls.parent / 'implementation_tests_post_packaging.json'
    tests = read(tests_path)
    if tests['exit_code'] != 0:
        raise ValueError('Final targeted test suite did not pass')
    comparisons = {}
    for name in ('control_o100_r1', 'kalcb_native', 'control_k075', 'control_k100', 'control_k200',
                 'naive_native_combination', 'k100_o100_r1', 'k050_o050_r1', base['winner']):
        comparisons[name] = [metrics(checked(controls / 'cases' / (name + '_' + s))) for s in SCENARIOS]
    comparisons['selected'] = [metrics(checked(root / 'qualified' / (final['winner'] + '_' + s), True)) for s in SCENARIOS]
    assessment = {'capital_ranges': {}, 'diagnostic_drawdown_gate_breaches': [],
                  'auction_minimum_drawdown_headroom_pp': {}}
    for scenario, selected in zip(SCENARIOS, comparisons['selected']):
        capital_rows = [item['metrics'] for name, item in stress.items()
                        if name.startswith('capital_') and name.endswith('_' + scenario)] + [selected]
        assessment['capital_ranges'][scenario] = {
            key: [min(row[key] for row in capital_rows), max(row[key] for row in capital_rows)]
            for key in ('return_pct', 'drawdown_pct', 'trades')}
        gate = final['selection'][scenario + '_max_dd_pct']
        auction = []
        for name, item in stress.items():
            if not name.endswith('_' + scenario):
                continue
            dd = item['metrics']['drawdown_pct']
            if dd > gate:
                assessment['diagnostic_drawdown_gate_breaches'].append(
                    {'case': name, 'drawdown_pct': dd, 'reference_gate_pct': gate})
            if name.startswith('auction/'):
                auction.append(gate - dd)
        assessment['auction_minimum_drawdown_headroom_pp'][scenario] = min(auction)
    payload = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'profile': binding(profile_path), 'initial_equity_krw': profile['initial_equity_krw'],
        'source_generation_hash': profile['source_generation_hash'],
        'settings': profile['settings'], 'scope': profile['scope'],
        'live_deployment_enabled': False, 'independent_performance_established': False,
        'comparisons': comparisons, 'challenges': challenges, 'robustness': stress,
        'robustness_assessment': assessment,
        'base_grid_results': {n: [metrics(r) for r in values] for n, values in base['results'].items()},
        'base_grid_feasible': base['feasible'], 'base_grid_pareto': base['pareto'],
        'attribution': attribution, 'primary_selection': binding(controls / 'finalist.json'),
        'packaging_validation': {'recovery': binding(incident / 'recovery.json'),
            'all_frozen_files_reverified': binding(incident / 'full_manifest_reverification.json')},
        'implementation_tests': {'record': binding(tests_path), 'log': binding(tests['log'])},
        'naive_combination_attribution': {s: {k: a[k] for k in ('olr_liquidation_delay', 'overnight_exposure', 'admission_diagnostics')} for s, a in naive.items()},
        'robustness_registration': binding(robustness / 'registration.json')}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.with_suffix('.json'), payload)
    lines = []
    def p(text=''): lines.append(text)
    def table(rows):
        p('Recipe | Baseline net return / max DD / entries | Combined stress net return / max DD / entries')
        for name, values in rows.items():
            p(name + ' | ' + ' | '.join(f"{r['return_pct']:.4f}% / {r['drawdown_pct']:.4f}% / {r['trades']}" for r in values))
        p()
    budget = profile['settings']['native_budget']
    p('Portfolio synergy: implemented native budgets at KRW25,000,000')
    p('Evidence date: 2026-10-09. Research selection on reused development data; no live deployment.')
    p()
    p(f"Selected recipe: {final['winner']}")
    p(f"KALCB sizing factor: {budget['kalcb_sizing_fraction']:.2f}; OLR sizing and sleeve ceiling: {budget['olr_fraction']:.2f} of the sizing base.")
    p(f"Aggregate KALCB planned risk-unit ceiling: {budget.get('max_kalcb_stop_risk_fraction', 0):.2%} of marked NAV (zero means disabled).")
    p(f"OLR native recycling: {profile['settings']['olr_capacity'].get('capacity_redistribute', False)}; execution precedence: {profile['settings']['policy']['strategy_priority']}.")
    p('The factors are sizing inputs, not fixed capital weights that must sum to one. Both strategies operate through one marked account. KALCB sizing 0.75, for example, means 0.75 times native sizing equity; it does not mean a 75% maximum holding.')
    p()
    p('Comparable outcomes')
    p('Net cumulative returns after the replay cost model; not annualized or expected future returns. Drawdown uses the marked bar-level account, not closed-trade PNL. Entries collapse partial exit legs. The common window is 2025-08-07 through 2026-03-31, 156 sessions including two no-entry liquidation-tail sessions.')
    table(comparisons)
    p('The 0.50x KALCB / 50% OLR screen is also important: in this sample it exceeds both full native OLR and matched KALCB1 on net return and entry count while having lower drawdown in both scenarios. It therefore demonstrates a lower-risk portfolio benefit without needing a higher drawdown than either standalone. It is a screened reference, not an additionally published fully qualified profile.')
    p('The selected controlled-aggressive point instead maximizes stressed return within the fixed drawdown limits. A portfolio need not have lower drawdown than the lowest-risk standalone to be preferable under that objective, but the additional return and frequency must justify the additional risk. Neither observed dominance nor this selection rule establishes future dominance.')
    p('Matched standalone controls have the same initial equity, observation feed, physical gross/name/sector ceilings and execution assumptions. At a given sleeve sizing setting, the matched control differs in whether the other native strategy trades. The grid also tests different standalone sizing settings explicitly. The earlier unconstrained KALCB native bridge remains a frequency benchmark, not a claim that its unconstrained economics are a matched risk comparison.')
    p('The raw fully sized combination and doubled KALCB are diagnostic return leaders only if their drawdown passes the registered limits. Risk-infeasible returns do not displace a feasible selected portfolio.')
    raw = comparisons['naive_native_combination']
    fixed = comparisons['k100_o100_r1']
    p('The full-size funding ablation separates the ledger correction from budget selection. At unchanged full native sizing and recycling, the corrected ledger changes baseline/combined returns by ' +
      '/'.join(f"{b['return_pct']-a['return_pct']:+.4f}pp" for a, b in zip(raw, fixed)) +
      ', and economic entries by ' + '/'.join(f"{b['trades']-a['trades']:+d}" for a, b in zip(raw, fixed)) +
      '. It still fails the stressed-drawdown gate. The large risk and frequency improvement therefore comes from selecting the shared budgets as well as correcting reservations; the funding fix alone does not maximize frequency.')
    p()
    p('What changed and why')
    p('1. Both native cores and their existing candidate schedules remain intact, including OLR joint capacity allocation and its guard against an incompatible external intent allocator. The shared layer changes sizing views and feasible funding. It does not change signal scores or exits, or rewrite an order after the native core computes its requested quantity.')
    p('2. Native requests retain their lifecycle. Pending orders reserve only feasible integer quantities against current marked holdings, sleeve capacity, gross/name/sector capacity and the optional KALCB risk-unit ceiling. A large unfilled request no longer mechanically consumes more funding than can actually be allocated.')
    p('3. OLR subtracts its own pending requests once inside its native allocator. The external account view subtracts funded foreign reservations. Sells do not release funds until they fill. Each buy is checked again using actual execution price and commission; cancellations and partial fills release only the appropriate remaining reservation.')
    p('4. The previous learned trade-quality filter remains off. The earlier ablation showed that it could suppress substantial profitable activity. Native ordering is the evidence-backed fallback when an incremental cross-strategy ranking rule lacks reliable support.')
    p('5. The canonical research profile starts at KRW25m. The legacy synthetic optimization fixture is also labeled and changed to KRW25m so it cannot silently suggest that the native selected portfolio still assumes KRW100m.')
    p()
    p('Why the previous allocation did not deliver the intended synergy')
    p('Portfolio behavior is path dependent: shared marks, pending reservations, different holding periods, liquidity-limited exits, integer shares and compounding change both future sizing and the available opportunities. Adding standalone equity curves cannot reproduce it. At KRW25m, liquidity no longer de-risks large native KALCB orders as much as at KRW100m; simply retaining doubled KALCB sizing raises drawdown and can reduce entry frequency.')
    p('The allocation correction therefore starts from preserved native strategies and a clean shared-capital contract. Previous rounds supply frozen strategy settings, ablation warnings, adverse execution assumptions and exact reference paths. Their optimal portfolio weights and historical return numbers are not portable to a changed strategy or capital base.')
    p()
    p('Selection protocol and finite search')
    p('Eight exact source bridges precede 52 portfolio screens and 24 matched standalone screens. The base portfolio grid is KALCB 0.50/0.75/1.00 times sizing, OLR 0.35/0.50/0.75/1.00 times sizing/budget, and recycling off/on. Two KALCB2/OLR0.20 migration controls add four scenario runs. Both baseline and combined stress are required.')
    p('Risk gates were fixed before results: maximum drawdown 11% baseline and 12.5% combined stress. A candidate must beat the best standalone feasible in BOTH scenarios, in each scenario, and retain at least 351/352 economic entries. Feasible candidates are Pareto filtered across both returns, both drawdowns and minimum scenario frequency. Rank by stressed net return; within one percentage point prefer greater minimum frequency, then lower maximum drawdown. This one-point band is a decision rule, not a confidence interval.')
    p(f"Measured standalone return floors: {final['return_floors'][0]:.6f}% and {final['return_floors'][1]:.6f}%. Base winner: {base['winner']}; feasible base recipes: {len(base['feasible'])}.")
    best_frequency = max(min(r['trades'] for r in base['results'][n]) for n in base['feasible'])
    best_stress_return = max(base['results'][n][1]['return_pct'] for n in base['feasible'])
    p(f"The feasible-grid maximum stressed return is {best_stress_return:.6f}%; its maximum minimum-scenario entry count is {best_frequency}. These are finite-grid observed results, not global or forward optima.")
    p('Higher KALCB sizing is explored only under the registered boundary trigger. An interior winner does not justify an open-ended extra search. The result is the preferred tested feasible profile, not proof of a global optimum or maximum expected future return.')
    p()
    p('Planned risk challenge')
    table(challenges['planned_risk']['results'])
    p('The cap includes marked-account KALCB holdings and funded pending requests using their original risk-per-share metadata. This is a position-sizing risk unit, not a guaranteed stop execution loss; the selected native strategies can have hard stops disabled. If the tested caps leave economic paths unchanged, no historical benefit is attributed to them. A dormant cap is a forward admission guard only.')
    p('Before any cap-challenge replay, an exact-economic-tie rule selects the smallest positive tested ceiling. That preference adds a tighter forward guard at no measured cost; it does not imply the sample identified an optimal risk threshold.')
    p('Priority challenge')
    table(challenges['priority']['results'])
    support = final.get('priority_support') or {}
    p(f"Potential constrained simultaneous cross-strategy sessions: {support.get('distinct_potential_conflict_sessions')}; minimum support to replace native priority: {support.get('minimum')}. Baseline and stressed repetitions of a session count once. Exact economic ties retain the existing priority.")
    p('This challenge changes only precedence among simultaneous native orders. It does not anticipate tomorrow\'s candidate, forcibly replace a live position, or label a known historical winner as stronger at its entry time.')
    stem = base['winner'].rsplit('_r', 1)[0]
    off, on = base['results'].get(stem + '_r0'), base['results'].get(stem + '_r1')
    if off and on:
        p('At the selected budget, recycling on versus off changes baseline/combined returns by ' +
          '/'.join(f"{b['return_pct']-a['return_pct']:+.4f}pp" for a, b in zip(off, on)) +
          ' and entry counts by ' + '/'.join(f"{b['trades']-a['trades']:+d}" for a, b in zip(off, on)) +
          '. The intervention count below is the relevant independent support; repeated runs and later compounding do not create more independent recycling examples.')
    p()
    p('Does it block weaker trades and admit stronger ones?')
    p('It removes an avoidable funding distortion and preserves validated native opportunity selection. It does not possess an oracle for future PNL. Realized winners among missing standalone trades cannot safely be used as prospective ranking labels: entries, prices, sizing, exits and later account state differ. The following shadow comparisons measure differences, not recoverable incremental alpha.')
    for scenario in SCENARIOS:
        a = attribution['scenarios'][scenario]
        p(scenario + ':')
        p(f"  Economic entries: {a['frequency_detail']['collapsed_strategy_entries']}; unique issuer-entry days: {a['frequency_detail']['unique_issuer_entry_days']}; unique symbol/fill timestamps: {a['frequency_detail']['unique_symbol_fill_timestamps']}.")
        p(f"  Reconciled net PNL by strategy (KRW): {a['pnl_by_strategy']}")
        p(f"  Admission diagnostics: {a['admission_diagnostics']}")
        p(f"  Observed funding exposure: {a['observed_funding_exposure']}")
        p(f"  Submission timing: {a['submission_timing']}")
        for s in a['matched_sleeve_comparisons']:
            p(f"  {s['strategy']}: matched entries {s['matched_entries']}; missing standalone {s['missing_reference']}; new shared {s['new_shared']}.")
            reasons = {}
            for d in s['missing_details']: reasons[d['reason']] = reasons.get(d['reason'], 0) + 1
            p(f"    Missing-entry reason counts: {reasons}")
            existing = [d for d in s['missing_details'] if d['existing_position_observed_without_new_submission']]
            p(f"    Already-held same-strategy positions without a new submission: {len(existing)}. Their absence as new entries must not be described as explicit funding rejections.")
        p(f"  Recycling support: {a['recycling_support']}; interventions: {a['recycling_interventions']}")
        p(f"  Coincident opposite-strategy same-symbol flows: {len(a['same_phase_opposite_strategy_flows'])}. No hypothetical crossing, fee saving or avoided tax is credited.")
    p('Future ranking work should estimate the incremental net value of admitting an opportunity under the actual shared account, including holding duration, opportunity cost, slippage and correlated exposure. Use only information available at the decision, matured chronological outcomes, and independent issuer/session support. Fit only after the native baseline is beaten after costs; retain native decisions when support is sparse. A profitable historical blocked trade alone does not establish that decision-time predictability.')
    p('An existing OLR position is a separate issue from pending-order priority: the frozen native core does not submit an unrelated new entry while that issuer is already held. A small older holding can therefore preclude a later, larger standalone entry. Changing this would require a separately tested position-renewal or multi-lot lifecycle design, including its extra turnover and exit obligations; it cannot be repaired safely by pretending the existing position has already exited.')
    p()
    p('Robustness: same selected recipe, no re-ranking')
    p('Test | Starting capital | Net return / max DD / entries | Full source qualification')
    for name, item in stress.items():
        r = item['metrics']
        p(f"{name} | {item['initial_equity_krw']:,} | {r['return_pct']:.4f}% / {r['drawdown_pct']:.4f}% / {r['trades']} | {r['audit_status']}")
    for scenario, ranges in assessment['capital_ranges'].items():
        rr, dd, count = (ranges[k] for k in ('return_pct', 'drawdown_pct', 'trades'))
        p(f"Across KRW20m/22.5m/25m/27.5m/30m, {scenario}: returns {rr[0]:.4f}% to {rr[1]:.4f}%, drawdown {dd[0]:.4f}% to {dd[1]:.4f}%, entries {count[0]} to {count[1]}.")
    p(f"Extra diagnostic cases exceeding the corresponding original drawdown gate: {assessment['diagnostic_drawdown_gate_breaches']}")
    if assessment['diagnostic_drawdown_gate_breaches']:
        p('These are material adverse results, not passing risk tests. The recipe remains the winner of the original registered selection, but it is not within the original drawdown limits under every additional stress. Higher net return in a nonfill scenario does not offset its drawdown breach: altered exit timing can improve PNL while increasing interim losses. The diagnostic results were not used to retune the recipe or relax a gate. This limits the case for broker deployment and any claim of robustness to unreliable auctions.')
    p(f"Smallest auction-stress drawdown headroom against those gates, in percentage points: {assessment['auction_minimum_drawdown_headroom_pp']}. Small positive headroom is not a reliable forward safety margin.")
    p('The capital sweep tests actual integer sizing and execution, not linear rescaling. The KRW25m point is the qualified selection above. No-upward-compounding caps the sizing base at initial equity while retaining current losses; it is not a fixed-notional PNL reconstruction.')
    p('Auction adverse price and nonfill tests alter OLR closing-auction execution. The selected OLR entry plan uses next-bar market entries, while many exits use the auction, so this is principally an exit and delayed-liquidation test. The independent full source audits cover the selected KRW25m baseline, combined, cost-only and participation-only runs; additional robustness screens retain their explicitly unverified source status while passing physical and funding checks.')
    for scenario, case in profile['qualified_cases'].items():
        r = case['metrics']
        p(f"Qualified {scenario}: {r['return_pct']:.6f}% net return, {r['drawdown_pct']:.6f}% max DD, {r['trades']} entries.")
    p()
    p('Residual risks and uncertainty')
    for scenario in SCENARIOS:
        a = attribution['scenarios'][scenario]
        p(f"{scenario}: worst session {a['worst_session_pct']:.4f}%; January/February share of final-exit net PNL {a['january_february_exit_profit_share']:.2%}; marked sleeve contribution statistics {a['daily_marked_sleeve_contributions']}.")
        p(f"  OLR liquidation delay: {a['olr_liquidation_delay']}")
        p(f"  Naive full-size combination liquidation delay for comparison: {naive[scenario]['olr_liquidation_delay']}")
        p(f"  Overnight exposure and simple gap shocks: {a['overnight_exposure']}")
        p(f"  Conditional paired bootstrap versus full native OLR: {a['bootstrap_vs_native_olr']}")
    p('Historical drawdown gates do not bound future losses. New buy admissions use ceilings of 2 times marked NAV for gross exposure, 1 time NAV per name, and 2 times NAV per sector under the common policy. These are permissive admission ceilings, not target exposures or continuous limits: existing exposure can move above them after mark changes without forced liquidation. OLR sleeve sizing separately limits overnight capital at admission.')
    p('The inherited KALCB buying-power model can produce negative intraday cash. Lower sizing factors do not eliminate that leverage. The replay costs do not contain a separate financing charge; actual account eligibility, settlement funding, collateral rules and any financing fees must be checked before broker deployment. Admission limits can be exceeded subsequently by mark changes without a new entry.')
    p('The cost model assumes 2bp commission per side and 20bp sell tax. Baseline slippage is 3bp KALCB/5bp OLR; cost stress adds 10bp. Capacity uses a fraction of the preceding completed contiguous five-minute bar, 1% baseline and 0.5% participation stress. The native OLR capacity forecast remains 1%. These are declared research assumptions, not current brokerage terms or verified order-book capacity. Actual opening-bar participation can exceed the preceding-bar fraction.')
    p('Every performance comparison reuses development observations, including the inherited strategy choices. Paired block bootstrap is conditional resampling and does not correct the search history. Repeated strategy selection can overfit: Bailey et al., The Probability of Backtest Overfitting, https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf . There is no valid independent forward-performance claim here and no invented out-of-sample result from an incompatible candidate schedule.')
    p()
    p('Verification and provenance')
    p('The final targeted suite passed all 117 tests, covering native funding, selection gates, physical execution policy, existing portfolio behavior, source replay, integrity repairs and publication. The exact command and test log are hash-bound in the companion JSON report.')
    p('All eight unchanged source bridges match trades, daily NAV and bar NAV exactly. Buy admissions are independently reconstructed from the order/fill/mark journal; copied account snapshots are not trusted as inputs. Share counts remain exact; only sub-won floating-point roundoff is tolerated in monetary comparisons. Every replay must finish without inventory, pending orders, deferred intents or KALCB overnight residuals.')
    p('Qualification reproduces the selected economic paths exactly and binds source, study, requests and artifact hashes. The publication entry point refuses missing full audit scenarios, altered allocation evidence, changed trade paths or failed return/risk/frequency gates. Prototype generations with an overly strict floating-point comparison were excluded; their failed audit status was not relabeled as passing.')
    p('A later packaging defect was caught before the cost-only qualification loaded trading inputs: creating child specifications wrote through shared hardlinks. The harness now creates private child specifications and verifies existing frozen specifications before accepting cached results. Eight affected specifications were restored byte-for-byte from their saved study records against the unchanged, pre-existing manifest hashes; failed work and original corrupted bytes were preserved. Every file in all nine active generations was subsequently rehashed and matched. The failed cost attempt was rerun; no failed audit was relabeled and no trading runtime, selection gate or economic output was changed by this repair. Two dedicated regression tests cover ancestor/sibling isolation and rejection of changed frozen specifications.')
    p(f"Frozen source generation: {profile['source_generation_hash']}")
    p(f"Qualified study: {profile['study']['path']}")
    p(f"Controls and source bridges: {controls}")
    p(f"Robustness registration: {robustness / 'registration.json'}")
    p(f"Published profile: {profile_path}")
    p('Verify the canonical selection: python scripts/replay_selected_portfolio.py verify')
    p('Replay a scenario into a NEW output directory: python scripts/replay_selected_portfolio.py replay --scenario combined --output <new-directory>')
    p('Implementation: deployment/olr_kalcb/native_budget.py; backtests/strategies/portfolio_synergy/native_budget.py; backtests/engine/integrity.py; scripts/portfolio_synergy_study.py. Research orchestration, attribution and charting are separate scripts with portfolio_native_budget, analyze_native_portfolio and plot_native_portfolio_frontier names.')
    p('Live broker parity remains separate work: correct shared buying power, the same observation feed and causal timestamps, partial-fill/cancel acknowledgments, overnight marks, auction fallback, cost and tax handling, and physical limits must be demonstrated. This implementation has no live-order or automatic promotion path.')
    output.with_suffix('.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(output.with_suffix('.txt'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--round-dir', type=Path, help='Write full final diagnostics for a registered native research round.')
    p.add_argument('--profile', type=Path)
    p.add_argument('--controls', type=Path)
    p.add_argument('--robustness', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.round_dir:
        if any((a.profile, a.controls, a.robustness, a.output)):
            p.error('--round-dir cannot be combined with the legacy report arguments')
        from backtests.strategies.portfolio_synergy.native_round_diagnostics import write_round_diagnostics
        result = write_round_diagnostics(a.round_dir)
        print(a.round_dir.resolve() / 'round_final_diagnostics.txt')
    else:
        if not all((a.profile, a.controls, a.robustness, a.output)):
            p.error('Supply --round-dir, or all of --profile, --controls, --robustness and --output')
        report(a.profile.resolve(), a.controls.resolve(), a.robustness.resolve(), a.output.resolve())

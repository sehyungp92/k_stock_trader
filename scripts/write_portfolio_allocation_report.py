"""Write an evidence-linked implementation report after the finite study."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def number(x):
    return f'{x:,.4f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.study.parent
    study = read(args.study)
    final = read(root / 'finalist.json')
    diagnostic = read(root / 'allocation_diagnostics.json')
    winner = final['winner']
    uncertainty=diagnostic.get('paired_development_uncertainty',{})
    lines = ['PORTFOLIO SYNERGY: IMPLEMENTATION AND EVALUATION', '2026-10-08', '',
             '1. Decision and scope', '']
    if winner:
        b, s = final['results'][winner]
        lines += [f'The selected research recipe is {winner}. On the common KRW100m account, it returned '
            f'{number(b["return_pct"])}% with {number(b["drawdown_pct"])}% marked maximum drawdown and {b["trades"]} completed trades. '
            f'Combined stress returned {number(s["return_pct"])}% with {number(s["drawdown_pct"])}% drawdown and {s["trades"]} trades.', '',
            'It is the selected member of the registered, finite set under the declared return, frequency and risk rules. '
            'It is not proof of a globally optimal portfolio or of future expected returns. '
            'The native strategies and allocator were evaluated on reused development history, not an independent holdout.', '']
    else:
        lines += ['No candidate passed all of the registered risk, standalone-return and frequency gates. '
            'Do not promote a portfolio simply because it is the best member of an inadequate set. '
            'The allocator implementation and all failed comparisons remain available for review.', '']
    lines += [f'The window is {study["configuration"]["date_range"]["start"]} through '
              f'{study["configuration"]["date_range"]["end"]}, 156 exchange sessions. '
              'Each comparison starts with KRW100m. Baseline and stress use matching dates, prices, candidate generations, '
              'native signal/exit rules and cost definitions. The original 50 registered cases were completed or reused from identical '
              'completed primary cases with provenance receipts. A subsequent 12-case controlled follow-up preserves native KALCB '
              'submission and repeats two paired source-generation bridges. These 62 completed case records include repeated '
              'controls; they are not 62 independent observations. Research profiles do not send live broker orders.', '',
              '2. Selection rule and complete comparison', '',
              'The fixed empirical gates are maximum marked DD <=11% in baseline and <=12.5% in combined stress. '
              'A portfolio must beat every feasible registered standalone sizing reference in both scenarios and retain '
              'at least native KALCB trade frequency in both. Physical leverage/funding limits remain unchanged. '
              'Risk-feasible non-dominated portfolios are ranked by stressed return; within one percentage point, '
              'higher minimum trade count is preferred, followed by lower maximum DD. The one-point band is a registered '
              'decision rule, not a statistically estimated confidence interval.', '',
              'Feasible standalone references: ' + ', '.join(final['reference_frontier']),
              f'Required return floors: baseline >{number(final["return_floors"][0])}%; combined >{number(final["return_floors"][1])}%.', '',
              'Recipe | Baseline net return / marked DD / trades | Combined net return / marked DD / trades | Gate result']
    for name, pair in final['results'].items():
        status = 'SELECTED' if name == winner else ', '.join(final['rejected'].get(name, [])) or (
            'feasible standalone' if name in final['reference_frontier'] else 'portfolio frontier' if name in final['frontier'] else 'reference or non-selected')
        cells = [f'{r["return_pct"]:.4f}% / {r["drawdown_pct"]:.4f}% / {r["trades"]}' for r in pair]
        lines.append(name + ' | ' + ' | '.join(cells) + ' | ' + status)
    lines += ['', 'A sizing multiplier changes the native sizing base and replays quantities, fills, exits and compounding. '
              'It does not multiply an already-completed return curve. KALCB 2x sizing does not authorize 4x account leverage: '
              'the actual account and strategy funding limits remain 2x. The standalone challenge matters because the old '
              'portfolio looked much less compelling against a stronger executable KALCB allocation than against unscaled KALCB.', '',
              '3. What changed and why', '',
              'Native signals are staged as intents before capital is reserved. A joint allocation pass uses marked physical '
              'NAV, filled exposure and remaining BUY commitments. Pending sales do not fund purchases. Only approved quantities '
              'are reserved, including cost buffers. Partial allocations are allowed, and strategy budgets are checked again '
              'at execution prices. Native state is reconciled through the existing fill, cancellation and expiry paths.', '',
              'The controlled native-preservation mode is a deliberate exception: KALCB retains its original submission order, '
              'working-order state, slot eligibility and execution-time funding admission. Only OLR uses the new pre-allocation '
              'path. KALCB still passes the shared physical funding, exposure and execution-time strategy budget checks. '
              'The allocation audit separately reconciles native KALCB submissions to actual broker journal entries. This '
              'mode cannot be combined with external value ranking. Preserving native submission does not mean preserving '
              'the identical standalone return path: shared NAV, OLR holdings and common risk caps can still change quantities.', '',
              'The reason for this narrower integration is measurable alpha displacement. In the earlier KALCB2/OLR20 '
              'proportional stressed portfolio, KALCB earned KRW53.7998m versus KRW71.5558m standalone, a KRW17.7560m '
              'shortfall, while OLR added only KRW8.9026m. Relative to that standalone, 11 newly admitted KALCB trades lost '
              'KRW7.4564m; matched quantity changes cost KRW4.3553m and matched price/exit-path changes cost KRW10.2240m. '
              'Missing 32 standalone KALCB trades removed a net KRW4.2796m loss, partially offsetting those effects. '
              'These are descriptive attribution terms, not independently tradable counterfactual profits. Clearing native '
              'pending-entry state, reordering proposals and reserving entire native requests before partial fills changed '
              'more than just competition with OLR. The direct joint/native pair isolates the submission integration change; '
              'the sector-cap variants separately test concentration constraints.', '',
              'The optional same-symbol aggregation preserves separate strategy ownership while applying common exposure limits. '
              'The optional entry and exit capacity forecasts use only completed, prior observations. Forecast capacity is shared '
              'across competing requests for the same symbol. Deferral is bounded and cannot retroactively fill an expired signal. '
              'These options were tested separately; implementation availability is not evidence that each belongs in the selected profile.', '',
              'The proportional OLR sizing experiments address a distinct issue: a lower OLR budget alone can be consumed by its '
              'first proposal, because that proposal is still sized against the entire account. Scaling the OLR sizing base with '
              'its budget preserves the native relative allocation across candidates. Small sleeves also reduce the amount that '
              'must exit through limited auction liquidity. Larger KALCB sizing then tests whether intraday risk can generate more '
              'return without carrying the previous near-full-account OLR exposure overnight.', '']
    if winner:
        lines += ['Selected settings, exactly as qualified:', json.dumps(final['recipe'], indent=2), '']
    lines += ['These are entry admission policies and empirical acceptance gates. They do not force continuous rebalancing '
              'after prices move and do not guarantee a future maximum drawdown. Existing positions keep their native management. '
              'Discretionary replacement of held positions was not enabled: it would require validated remaining-value estimates '
              'and switching-cost comparisons, which the old entry score does not provide.', '',
              '4. The old value model did not solve the trade-selection problem', '']
    for scenario in ('baseline', 'combined'):
        bad = diagnostic['results']['value_capacity_75_' + scenario]
        control = diagnostic['results']['deferred_capacity_75_' + scenario]
        lines.append(f'{scenario}: adding supported-value filtering changed return from {number(control["return_pct"])}% '
                     f'to {number(bad["return_pct"])}%, DD from {number(control["drawdown_pct"])}% to '
                     f'{number(bad["drawdown_pct"])}%, and trades from {control["trades"]} to {bad["trades"]}.')
    lines += ['', 'Causality and sample support were independently verified, but they did not make the forecast economically '
              'useful. The model uses previously selected trade cohorts, not complete counterfactual opportunity labels. '
              'Its nonpositive-value filter removed a large part of KALCB participation. A more complicated allocator cannot '
              'repair an unreliable payoff forecast merely by sorting its numbers accurately.', '',
              'Matched no-value-control diagnostics (retrospective, incomplete labels):',
              json.dumps(diagnostic['value_model_ablation_retention'], indent=2), '',
              'These reference trades have their own sizes and shared-liquidity path. Their outcomes are useful for detecting '
              'selection failure, but they are not jointly feasible profits that can be added back to the selected account. '
              'In particular, a positive average among rejected opportunities does not mean every rejected trade should be entered.', '']
    if winner:
        lines += ['5. Risk, economically meaningful frequency and retained opportunities', '']
        for scenario in ('baseline', 'combined'):
            r = diagnostic['results'][winner + '_' + scenario]
            old = diagnostic['results']['legacy_selected_' + scenario]
            strong = diagnostic['results']['kalcb_size_200_' + scenario]
            sleeves=r['diagnostics']['sleeves']
            lines += [scenario + ':',
                f'Return change versus previous portfolio: {r["return_pct"]-old["return_pct"]:+.4f} percentage points. '
                f'DD change: {r["drawdown_pct"]-old["drawdown_pct"]:+.4f} points. '
                f'Trade change: {r["trades"]-old["trades"]:+d}. Active sessions: {r["active_days"]} versus {old["active_days"]}.',
                f'Against KALCB2 standalone: return {r["return_pct"]-strong["return_pct"]:+.4f} points; '
                f'DD {r["drawdown_pct"]-strong["drawdown_pct"]:+.4f} points; trades {r["trades"]-strong["trades"]:+d}. '
                f'KALCB sleeve PnL is KRW{(sleeves["KALCB"]["net_pnl_krw"]-strong["diagnostics"]["sleeves"]["KALCB"]["net_pnl_krw"])/1e6:+.4f}m '
                f'relative to standalone; OLR contributes KRW{sleeves["OLR"]["net_pnl_krw"]/1e6:.4f}m. '
                'Thus net synergy is positive in this replay, but not every cost of sharing the account disappears.',
                f'Trades with entry notional >=1% of prior-close NAV: selected '
                f'{r["diagnostics"]["trades_at_least_1pct_prior_close_equity"]}, previous portfolio '
                f'{old["diagnostics"]["trades_at_least_1pct_prior_close_equity"]}, KALCB2 '
                f'{strong["diagnostics"]["trades_at_least_1pct_prior_close_equity"]}. '
                'The activity gain is substantial against standalone; it is modest against the old portfolio.',
                'Selected diagnostics:', json.dumps(r['diagnostics'], indent=2),
                'Allocation diagnostics:', json.dumps(diagnostic['forecast_binding'][winner + '_' + scenario], indent=2), '']
        lines += ['Trade counts are collapsed completed entry lifecycles, not partial fills or retry events. '
                  'Ticket-size and turnover statistics show whether extra activity is economically material. '
                  'The >=1% ticket count is a diagnostic, not a newly imposed trading filter. '
                  'End-of-day exposure is marked equity minus cash divided by equity; KALCB is required to be flat overnight. '
                  'OLR exits later than the next session identify final lifecycle exits, not the duration of every partial quantity.', '',
                  'Selected opportunity retention versus native standalone reference trades:',
                  json.dumps(diagnostic['selected_opportunity_retention'], indent=2), '',
                  'Retention versus KALCB2 and native OLR reference trades:',
                  json.dumps(diagnostic['selected_sized_reference_retention'], indent=2), '',
                  'Approved in these retention tables means an entry was submitted, not that it necessarily filled. '
                  'Native submissions can subsequently expire or be blocked at execution. Core-rejected candidates that '
                  'never generated an entry intent are outside these matched-intent tables.', '',
                  'The selected allocator cannot know future winners. Native rank is the fallback when values are unsupported. '
                  'Where cross-strategy simultaneous cohorts are absent, these results establish sizing and capital-availability '
                  'effects, not successful direct ranking of KALCB against OLR at the same instant. '
                  'Complete never-admitted shadow labels and independent calibration would be required before claiming that a '
                  'learned model reliably blocks the worst opportunities.', '',
                  '6. Uncertainty and interpretation', '',
                  f'The observed advantage over KALCB2 is {uncertainty["kalcb_size_200_baseline"]["observed_return_difference_pct_points"]:+.4f} '
                  f'percentage points in baseline and {uncertainty["kalcb_size_200_combined"]["observed_return_difference_pct_points"]:+.4f} '
                  'points under combined stress. The respective descriptive 95% paired-block intervals are '
                  f'{uncertainty["kalcb_size_200_baseline"]["descriptive_95pct_interval_pct_points"]} and '
                  f'{uncertainty["kalcb_size_200_combined"]["descriptive_95pct_interval_pct_points"]} points. Both include zero. Thus this finite search '
                  'found an economically better historical candidate, but did not establish a statistically reliable '
                  'future expected-return advantage. The intervals also omit the additional uncertainty from upstream '
                  'strategy selection and this adaptive series of portfolio experiments.', '',
                  json.dumps(diagnostic['paired_development_uncertainty'], indent=2), '',
                  'The paired five-session block resampling uses the same date indices for both books and reports compounded '
                  'return differences. It is descriptive sensitivity to observed dependence, not a correction for repeated '
                  'strategy/portfolio selection. The strategies themselves were selected using this history. Re-splitting '
                  'the same data does not establish independent out-of-sample performance. Return concentration, tail losses '
                  'and the retained sample size should be considered alongside the point estimates.', '']
        lines += ['7. Full native qualification', '']
        for scenario in ('baseline', 'combined', 'cost', 'participation'):
            path = root / 'qualified' / (winner + '_' + scenario)
            r = read(path / 'result.json')
            lines.append(f'{scenario}: audit={r["audit_status"]}; physical execution={r["execution_validity_pass"]}; '
                         f'return={r["return_pct"]:.4f}%; DD={r["drawdown_pct"]:.4f}%; trades={r["trades"]}; '
                         f'terminal positions={len(r["terminal_inventory"])}; pending orders={r["terminal_pending"]}; '
                         f'KALCB overnight residual sessions={r["intraday_residual_sessions"]}.')
        lines += ['', 'Baseline and combined qualification must exactly reproduce the selected screening economics. '
                  'The replay uses frozen source, market/candidate provenance, the independent execution ledger and a separate '
                  'allocation audit. A passing historical execution audit does not establish broker/live parity or observed '
                  'order-book queue liquidity. The capacity contract remains a declared participation model.', '']
    lines += ['8. Completion, limitations and use', '',
              'Implementation/report completion is separate from independent performance and operational readiness. '
              'No fixed minimum number of future sessions is a completion condition. The selected rule can be evaluated on '
              'the declared available window, with all eligible sessions and uncertainty reported. No automatic live promotion, '
              'broker order or recurring automation is created by this work.', '',
              'Baseline costs are 2bp commission each side, 20bp sell tax, 3bp KALCB and 5bp OLR slippage each side, '
              'and 1% bar participation. Combined stress adds 10bp slippage each side and halves participation. '
              'These tests do not model every financing cost, execution failure, price-limit event or future gap. '
              'All risk limits are interpreted within those declared assumptions.', '',
              'The selected profile is qualified at KRW100m. Equity-scaled caps do not make execution capacity scale '
              'linearly: applying the same settings at KRW10m or a larger account is a new size-specific qualification, '
              'not an already-established result. The 2x sector ceiling allows all gross exposure in one sector; it '
              'removes the earlier 1.2x sector bottleneck and is explicitly the more aggressive concentration choice. '
              'The 1x symbol ceiling, 2x account gross ceiling, native intraday exits and 20% OLR entry budget remain '
              'the controls. Prices can move exposures above entry ceilings after admission.', '',
              'The latest canonical raw OLR round began earlier and used a smaller account. Its 188% headline return is not '
              'a like-for-like benchmark for the KRW100m common-window portfolio. Canonical KALCB also used an 18bp sell tax '
              'rather than the 20bp shared comparison assumption. The tables above use matched capital, dates and scenarios.', '',
              'The architecture document describes the implementation and tests in detail:',
              'docs/research/portfolio_synergy_intent_allocator_20261008.txt', '',
              'Study: ' + str(args.study.resolve()),
              'Registration: ' + str(root / 'registration.json'),
              'Selection: ' + str(root / 'finalist.json'),
              'Diagnostics: ' + str(root / 'allocation_diagnostics.json'),
              'Source generation: ' + study['source_generation_hash'],
              'Workspace validation receipt: ' + str(root.parent / 'validation_receipt_v6.json'),
              'Research entry point: scripts/replay_selected_portfolio.py',
              'Reproduction: python -B scripts/portfolio_native_preservation_study.py select --study <runtime_v6/study.json>',
              'Verification: python -B scripts/replay_selected_portfolio.py verify --profile <selected_profile.json>',
              'Replay: python -B scripts/replay_selected_portfolio.py replay --profile <selected_profile.json> '
              '--scenario baseline --output <new-output-directory>', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'report': str(args.output.resolve()), 'winner': winner, 'lines': len(lines)}))


if __name__ == '__main__':
    main()

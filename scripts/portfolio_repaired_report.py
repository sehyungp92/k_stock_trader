"""Summarize the completed fixed OOS comparison without selecting another recipe."""
from collections import Counter
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.portfolio_source_repair import ROOT, PARENT
from scripts.portfolio_targeted_validation import read, bound
from scripts.portfolio_repaired_validation import verify
from backtests.strategies.kalcb.executed_value import digest, write_json


def build(root=ROOT):
    decision = read(root / "acceptance.json")
    if not decision["validation_window_now_consumed"] or len(decision["qualified_cases"]) != 3:
        raise ValueError("Await all fixed OOS comparisons")
    def test_counts(name):
        test = ET.parse(root / name).getroot()
        suites = [test] if test.tag == "testsuite" else list(test.iter("testsuite"))
        return {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    primary, sector, repeated = [test_counts(n) for n in ('software_tests.xml', 'sector_dependency_tests.xml', 'forward_dependency_tests.xml')]
    tests = {key: primary[key] + sector[key] for key in primary}
    if tests["failures"] or tests["errors"]: raise ValueError("Focused tests failed")
    if repeated['failures'] or repeated['errors']: raise ValueError('Forward dependency regression failed')
    latest = test_counts('software_tests_latest.xml')
    if latest != tests: raise ValueError('Final focused suite differs from the expected passing tests')
    final = decision["finalist"]; outcomes = decision["outcomes"]; cases = {}; marginal = {}; deltas = {}
    risk_failures = {}
    for name, scenarios in outcomes.items():
        risk_failures[name] = {}
        for scenario, metric in scenarios.items():
            checks = {'positive_net_return':metric['return_pct']>0,
                'absolute_drawdown_limit':metric['drawdown_pct'] <= (11 if scenario=='baseline' else 12.5),
                'overnight_max_limit':metric['overnight_max'] <= .8,
                'overnight_mean_limit':metric['overnight_mean'] <= .3,
                'worst_session_limit':metric['worst_session_pct'] >= -4.5}
            risk_failures[name][scenario] = [key for key,value in checks.items() if not value]
    for name, scenarios in decision["qualified_cases"].items():
        for scenario, receipt in scenarios.items():
            case = Path(receipt["output"]); actions = read(case / "corporate_action_audit.json")
            if not actions["pass"] or any(x["held_shares"] or x["pending_orders"] for x in actions["events"]):
                raise ValueError("Unaccounted corporate-action exposure")
            trades = read(case / "trades.json")
            result = read(case / 'result.json')
            cases[name, scenario] = {"strategy_trade_counts": dict(Counter(t["strategy_id"] for t in trades)),
                "strategy_net_pnl_krw":{s:sum(t['net_pnl'] for t in trades if t['strategy_id']==s) for s in ('KALCB','OLR')},
                'native_budget_blocked_orders':result['native_budget_blocked_orders'],
                'native_budget_resized_orders':result['native_budget_resized_orders'],
                "corporate_action_audit": bound(case / "corporate_action_audit.json"),
                "result": bound(case / "result.json")}
            if name == final:
                path = case / "marginal_quantity_evidence.json"; evidence = read(path)
                marginal[scenario] = {"changed_issuer_days": evidence["changed_issuer_days"],
                    "changed_sessions": evidence["changed_sessions"], 'reason_counts':dict(Counter(x['reason'] for x in evidence['decisions'])),
                    "scope": evidence["scope"], "artifact": bound(path)}
    for scenario in ("baseline", "combined"):
        a, b = outcomes[final][scenario], outcomes["control"][scenario]
        deltas[scenario] = {key: a[key] - b[key] for key in ("return_pct", "drawdown_pct", "trades", "worst_session_pct", "overnight_max", "overnight_mean")}
    from scripts.portfolio_targeted_result import paired_uncertainty
    uncertainty = {s: paired_uncertainty(Path(decision['qualified_cases'][final][s]['output']),
                                         Path(decision['qualified_cases']['control'][s]['output']))
                   for s in ('baseline', 'combined')}
    from scripts.analyze_portfolio_risk_return_tradeoff import load_case, matched_decomposition
    attribution = {s: matched_decomposition(load_case(decision['qualified_cases'][final][s]['output']),
                                             load_case(decision['qualified_cases']['control'][s]['output']))
                   for s in ('baseline', 'combined')}
    identity = {s:{n:digest(Path(decision['qualified_cases'][final][s]['output'])/n)==digest(Path(decision['qualified_cases']['control'][s]['output'])/n)
                   for n in ('trades.json','fills.json','daily.json','equity.json')} for s in ('baseline','combined')}
    worst_days = {}
    for scenario in ('baseline','combined'):
        control = load_case(decision['qualified_cases']['control'][scenario]['output'])
        previous = 25_000_000.; daily = []
        for row in control['daily']:
            change = row['equity']-previous
            kalcb = sum(t['net_pnl'] for t in control['trades'] if t['strategy_id']=='KALCB' and t['exit_fill_time'][:10]==row['day'])
            daily.append({'date':row['day'],'return_pct':100*change/previous,'account_pnl_krw':change,
                          'kalcb_pnl_krw':kalcb,'olr_marked_pnl_krw':change-kalcb})
            previous = row['equity']
        worst_days[scenario] = min(daily,key=lambda x:x['return_pct'])
    pointer = read(root / "protocol.json")["references"]["selected_pointer"]
    unchanged = digest(pointer["path"]) == pointer["sha256"]
    repair = read(root / "repaired_sessions/proof.json")
    data = {"acceptance": decision, "candidate_minus_current": deltas, "marginal_activity": marginal,
        "absolute_risk_failures": risk_failures,
        "descriptive_paired_uncertainty": uncertainty,
        "descriptive_trade_attribution": attribution,
        'candidate_control_economic_identity':identity, 'control_worst_day_attribution':worst_days,
        "cases": {n: {s: cases[n, s] for s in ("baseline", "combined")} for n in outcomes},
        "repairs": repair, "source_gate": bound(root / "source_basis_review.json"),
        "generation": read(root / "repair_generation.json"), "tests": tests, "forward_regression_recheck": repeated,
        "test_evidence": {n: bound(root / n) for n in ('software_tests.xml', 'sector_dependency_tests.xml', 'forward_dependency_tests.xml', 'software_tests_latest.xml')},
        "split": read(root / "split_audit_verified.json"), "canonical_pointer_unchanged": unchanged,
        "no_new_oos_search": True, "live_deployment_enabled": False}
    if (root / 'qualification_repair.json').exists():
        evidence_bridge = read(root / 'economic_evidence_repair_bridge.json')
        if not evidence_bridge['pass']: raise ValueError('Audit repair economic bridge failed')
        data['audit_evidence_repair_bridge'] = evidence_bridge
    stem = REPO / "docs/research/portfolio_source_repair_oos_validation_20261009"
    write_json(stem.with_suffix(".json"), data)
    lines = ["Portfolio source repair and frozen OOS validation", "9 October 2026 — starting account equity KRW25,000,000", "",
        "Decision: " + ("The sealed candidate passes the registered short-window OOS acceptance criteria." if decision["acceptance"]["pass"] else "The sealed candidate fails the registered OOS acceptance criteria and is not eligible for promotion."),
        "All six predeclared cases completed with full native source, execution and account audits. No candidate, model, sizing parameter, test date or acceptance threshold was selected using these results.", "",
        "1. Results", "", "K/O labels denote the registered native sizing/budget fractions. The candidate uses the current K0.75/O0.625 budget plus the supported marginal KALCB quantity reduction; the higher-risk reference raises OLR to 0.75.", "",
        "Portfolio / scenario                  Net return %       Maximum DD %       Economic entries       Final NAV KRW"]
    labels = {"control": "Current K0.75/O0.625", "incumbent": "Higher-risk K0.75/O0.75", final: "Candidate k_marginal"}
    for name in ("control", final, "incumbent"):
        for scenario in ("baseline", "combined"):
            m = outcomes[name][scenario]
            lines.append(f"{labels[name]+' / '+scenario:<39} {m['return_pct']:>11.6f} {m['drawdown_pct']:>18.6f} {m['trades']:>22} {25_000_000*(1+m['return_pct']/100):>21,.0f}")
    lines += ["", "Returns are for June 1–July 10, 2026, not annualized. There are 29 exchange sessions and 27 sessions eligible for new entries; July 9–10 are liquidation-only. Entry counts are economic trades after exit-leg collapse. Net returns include the frozen modeled transaction costs; financing charges remain excluded, as in the registered experiment.",
        "Baseline uses a 1% participation ceiling with KALCB/OLR slippage of 3/5 basis points. Combined stress halves participation to 0.5% and raises slippage to 13/15 basis points. Both use the frozen commissions and 20-basis-point sell tax. The same signals, date boundaries and allocation recipes are replayed under each execution scenario.", "",
        "Candidate minus current:"]
    for scenario, d in deltas.items():
        lines.append(f"- {scenario}: return {d['return_pct']:+.6f} percentage points, drawdown {d['drawdown_pct']:+.6f} percentage points, entries {d['trades']:+d}.")
    lines += ['', 'Absolute return/risk failures, including the current control:', json.dumps(risk_failures, indent=2),
        'Leaving the existing research pointer unchanged does not certify the current portfolio for deployment. A control that breaches these limits also fails the requested risk standard. A relative improvement in a candidate cannot override its absolute loss or risk failures.']
    lines += ["", "Full acceptance tests (unchanged):", json.dumps(decision["acceptance"], indent=2), "",
        "The candidate must be profitable in both scenarios, match or exceed current returns, retain at least 98% of current activity and increase drawdown by no more than 0.25 percentage points. Absolute drawdown caps are 11% baseline and 12.5% combined stress; maximum and mean overnight exposure caps are 80% and 30% of NAV; worst daily loss must not exceed 4.5%. The 515-entry development floor is not applied to a 29-session holdout.", "",
        "Risk detail (overnight fractions are percentages of NAV):"]
    for name in ("control", final, "incumbent"):
        for scenario in ("baseline", "combined"):
            m = outcomes[name][scenario]
            lines.append(f"- {labels[name]} / {scenario}: worst session {m['worst_session_pct']:.6f}%; maximum overnight {100*m['overnight_max']:.6f}%; mean overnight {100*m['overnight_mean']:.6f}%; KALCB/OLR entries {cases[name,scenario]['strategy_trade_counts']}; sleeve net PNL in KRW {cases[name,scenario]['strategy_net_pnl_krw']}.")
    lines += ["", "2. What was repaired", "",
        "The original archive interleaved original and adjusted OHLCV within 068270 on May 29 and used adjusted daily/minute observations on June 1–2. The repair creates an immutable generation instead of modifying the previous research or raw archive.", "",
        "- Incremental ingestion now sorts stably, rejects contradictory overlapping OHLCV, rejects incompatible declared bases and rejects contradictory provider-page duplicates. Raw page receipts retain retrieval time and explicitly leave the provider adjustment vintage unqualified. A real-data regression confirms that the 382-row original/adjusted conflict is rejected in either source order.",
        "- All 1,144 affected minutes are verified against preserved original/adjusted anchors. Every original share quantity has a unique integer inverse. Prices use the unique integer inverse or the documented KRX 100-KRW limit-tick / 50-KRW midpoint lattice in the affected price band. There is no nearest-tick guess or midpoint conversion-factor choice. One common factor must support every decoded field and all original anchors.",
        "- May 29 reconstruction exactly reproduces the retained original session. June 1–2 are reconstructed before aggregation. Five timeframes (1m, 5m, 15m, 30m, 1h) are produced from each accepted session. Delayed auction observations retain their true source time.",
        "- Each repaired session matches original daily OHLC and passes minute-level cumulative traded-value bounds. The daily/session residuals are exactly valued at the daily close: 117,150 shares on May 29, 11 on June 1 and 304 on June 2. These shares are not added to any bar or to execution capacity.",
        "- The actual replay generation changes one intraday file and 25 daily files (two basis-corrected daily rows plus 24 final-day volume revisions), and restores the omitted development sector table without changing its contents. All 103 symbols remain in the universe. All 3,090 original daily controls match and the complete retained cross-frequency audit has zero basis failures.",
        "- The affected investor-flow quantities match 18 direct provider comparisons. They remain original net shares; neither share nor monetary flow fields are blindly scaled by a chart-price factor.", "",
        "3. July 10 volume revisions and limits", "",
        "The 24 changed daily totals were checked separately. Original interval shares are preserved: 23 sessions match fresh minute OHLCV directly; 207940 has a later adjusted provider vintage, and its retained original observations pass the common-factor and traded-value checks. A redownload is therefore not automatically a valid replacement.",
        "Four opening cumulative-turnover observations lack an observed pre-session baseline. Their original interval OHLCV matches the provider directly; the audit records the feasible prior-turnover range rather than inventing a zero balance. Subsequent interval turnover bounds pass. The historical cause of the daily-total revisions remains unverified. The accepted scope is narrower: current original daily controls are recorded, July 10 daily data is excluded from all decisions by strict prior-day features, and residual daily volume never becomes intraday liquidity.", "",
        "4. Corporate actions", "",
        "Issuer notices establish Celltrion's 5% bonus entitlement, June 4 ex-date, June 5 record date, scheduled June 30 listing and cash settlement of fractions at the listing-day close. This is not an immediately tradable 1.05-for-1 stock split. The provider also identifies a June 29 interim-dividend event for 214370.",
        "The replay now checks actual holdings and pending orders before both action boundaries. All six runs confirm zero exposure at both events. No entitlement, position multiplier, dividend income or cash credit is therefore required for these particular comparisons. No stock or valid order was removed to produce that result. The guard would abort a replay with exposure; general bonus-share/dividend accounting for such a different run is not claimed to be implemented.",
        "Feature formulas remain those of the frozen strategies, applied to consistent original-price history. A new total-return or adjusted-indicator policy was not introduced using this OOS period.", "",
        "The first completed baseline simulation passed execution checks but its full qualification remained unverified because its inherited capacity declaration still named the May 12 basis and its forward OLR snapshots lacked the canonical feature-prefix exclusion annotations. The capacity declaration now references the verified July 10 original-price/share basis; execution limits and costs are unchanged. OLR annotations were independently recomputed from the frozen calendar and original bars for all 27 entry sessions. Candidate contents are exactly unchanged. The requalified baseline has identical daily/equity paths, trade economics and fill economics to the preserved pre-annotation attempt. These are audit-record repairs, not changes to signals, prices, execution rules or acceptance thresholds.", "",
        "5. Selection independence and reuse", "",
        "The original development selection remains sealed: k_marginal was chosen from the registered seven-recipe development screen and eight-scenario qualification battery. Development ends March 31; April 1–May 12 is already-consumed validation; June 1–July 10 was reserved for the fixed acceptance test. Six March 10 models, trained on labels ending before their fit, are reused without fitting on this window.",
        "The direct frozen-development comparison covers 1,609,324 intraday and 56,917 daily overlapping pre-OOS observations across all 103 symbols. It confirms complete coverage and identical OHLCV throughout the development interval, including 1,232,300 intraday and 16,068 daily rows. There are 34 later intraday boundary differences, all on May 12, outside both development and the actual May 29-onward intraday feature window. An additional 1,707,845-row intraday comparison against the preserved May 29 archive locates four May 12 differences and the damaged May 29 session. An initial daily-path probe was incomplete because the working archive no longer retained those filenames; the completed audit instead binds the actual frozen development inputs. The damaged May 29 and June 1–2 observations were rebuilt; model/training bytes and development outcomes were retained.",
        "Inference runs from frozen source files rather than the changing shared checkout. Two dependency omissions were found before any portfolio returns were produced. A missing calendar reference first stopped feature preparation. The first replay preflight then detected conflicting sector labels: freeze_market had omitted KALCB's sector_map.parquet, causing all 103 stocks to receive UNKNOWN during feature construction, while OLR retained its canonical YAML map. The complete runtime and the exact 103-stock development sector table are now retained. All model inputs, eligibility decisions and fixed-model predictions were rebuilt in a separate sector_complete_v2 generation; already scored candidates were not merely relabeled. The original failed artifacts are preserved. Sector coverage for the traded universe and cross-strategy agreement are now mandatory before replay, with categorical/eligibility checks added to the historical feature bridge. Auxiliary context stocks outside the traded universe keep their original historical classifications, including UNKNOWN where that was the original input. The June–July window is now consumed. Missing original strategy-search logs still prevent a categorical claim that no earlier external tuning ever used it.", "",
        "6. What this says about synergy", ""]
    for scenario, m in marginal.items():
        lines.append(f"- {scenario}: the supported marginal-quantity rule changes {m['changed_issuer_days']} issuer-days across {m['changed_sessions']} sessions.")
        lines.append('  Decision reasons: ' + json.dumps(m['reason_counts'],sort_keys=True))
        lines.append('  Exact candidate/control artifact identity: ' + json.dumps(identity[scenario],sort_keys=True))
        lines.append(f"  Shared native-budget blocked/resized orders: current {cases['control',scenario]['native_budget_blocked_orders']}/{cases['control',scenario]['native_budget_resized_orders']}; candidate {cases[final,scenario]['native_budget_blocked_orders']}/{cases[final,scenario]['native_budget_resized_orders']}.")
    if all(all(x.values()) for x in identity.values()):
        lines.append('The candidate is economically identical to the current portfolio in both scenarios. It provides zero observed return, drawdown or frequency improvement. Its failure is not evidence that an activated marginal policy worsened these trades: the rule never changed an issuer-day. No shared native-budget blocking/resizing occurred in either account, so better conflict ranking has no observed budget conflict to repair in this particular window. Ordinary execution-capacity constraints remain separate.')
    lines += ['', 'Worst-session co-loss accounting in the current shared account:']
    for scenario, row in worst_days.items():
        lines.append(f"- {scenario}, {row['date']}: account return {row['return_pct']:.6f}%, account PNL {row['account_pnl_krw']:,.0f} KRW; KALCB {row['kalcb_pnl_krw']:,.0f} KRW and OLR marked PNL {row['olr_marked_pnl_krw']:,.0f} KRW. KALCB is flat at session end, so the residual account change reconciles to OLR's marked sleeve.")
    lines.append('Both sleeves contribute losses on the worst session. This is evidence of shared realized downside in the tested account. It does not isolate standalone-strategy returns or prove a structural causal explanation for forecast failure.')
    lines += ['', 'Observed trade accounting, candidate minus current (KRW):']
    for scenario, comparison in attribution.items():
        for sleeve in ('KALCB', 'OLR'):
            a = comparison[sleeve]
            lines.append(f"- {scenario} / {sleeve}: PNL change {a['pnl_change']:+,.0f}; matched-trade quantity component {a['quantity_component']:+,.0f}; per-share PNL component {a['per_share_pnl_component']:+,.0f}; entry-set component {a['entry_set_component']:+,.0f}; {a['matched']} matched, {len(a['new'])} added and {len(a['missing'])} absent entries.")
    lines.append('The components reconcile exactly to realized PNL differences. Quantity includes integer sizing, wealth feedback and caps; per-share PNL includes fills, exits and costs. This is an accounting identity, not an isolated causal estimate or a decision-time label for whether a blocked trade was better.')
    lines += ['', 'Descriptive paired uncertainty (existing fixed 5/10-session moving-block procedure; 5,000 resamples per block length):']
    for scenario, detail in uncertainty.items():
        for block, values in detail['blocks_in_sessions'].items():
            q = values['return_difference_quantiles_pp']
            lines.append(f"- {scenario}, {block}-session blocks: candidate-minus-current 2.5th/50th/97.5th percentile return differences {q['p025']:.6f}/{q['p50']:.6f}/{q['p975']:.6f}pp; fraction positive {values['fraction_positive']:.4f}.")
    lines.append('These resamples describe dependence on the observed daily sequence. They are not a fresh policy replay, a probability of future outperformance or an additional selection gate. No acceptance criterion changes in response to them.')
    lines += ["The rule uses decision-time forecast support to reduce an unprofitable incremental KALCB quantity tier; it does not create new signals or use realized OOS PNL to decide which orders to block. Its quantity forecasts retain their original reference-capital calibration. Physical account equity and all executions here use KRW25m.",
        "This fixed test measures whether the development finalist survives new data and execution stress. It is not a search for the highest OOS-returning recipe, a proof of maximum portfolio synergy, proof that every rejected order was inferior, or a standalone-strategy dominance test. A short common market path cannot establish those stronger claims.", "",
        "7. Delivery and next use", "",
        f"Focused software verification after the final changes: {tests['tests']} tests, {tests['failures']} failures, {tests['errors']} errors. All six OOS cases passed their full native source/execution/account audits. Canonical research pointer unchanged: {unchanged}. Live deployment remains disabled.",
        "Use the new frozen validation study and its repaired manifest to reproduce this comparison. Preserve the earlier source incident, generation and development evidence. Any later strategy changes require development data and a fresh forward test; do not tune them against this now-consumed June–July result.", "",
        "Primary source documentation:",
        "https://www.celltrion.com/ko-kr/investment/notice/announce/4775",
        "https://www.celltrion.com/en-us/company/notice/4785",
        "https://global.krx.co.kr/contents/GLB/06/0602/0602010201/GLB0602010201T3.jsp",
        "https://regulation.krx.co.kr/contents/RGL/03/03010208/RGL03010208.jsp", "",
        "Reproducible evidence:", str(root / "acceptance.json"), str(root / "source_basis_review.json"),
        str(root / "validation/study.json"), str(root / "validation/inputs/manifest.json"), str(root / "repaired_sessions/proof.json")]
    stem.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot(outcomes, final, labels, stem.with_suffix(".png"))
    write_json(root / "delivery.json", {"complete": True, "repair_qualified": True, "oos_cases": 6,
        "acceptance_pass": decision["acceptance"]["pass"], "canonical_pointer_unchanged": unchanged,
        "tests": tests, "artifacts": [bound(stem.with_suffix(s)) for s in (".txt", ".json", ".png")],
        "decision": bound(root / "acceptance.json"), "source_review": bound(root / "source_basis_review.json")})
    print(json.dumps({"report": str(stem.with_suffix('.txt')), "acceptance_pass": decision["acceptance"]["pass"], "deltas": deltas}), flush=True)


def plot(outcomes, final, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    names = ["control", final, "incumbent"]; colors = ["#166a77", "#de8c30"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    for ax, key, title in zip(axes, ("return_pct", "drawdown_pct", "trades"), ("Modeled net return (%)", "Maximum drawdown (%)", "Economic entries")):
        x = np.arange(3)
        for offset, scenario, color in zip((-.19, .19), ("baseline", "combined"), colors):
            values = [outcomes[n][scenario][key] for n in names]
            bars = ax.bar(x + offset, values, width=.36, label=scenario, color=color)
            ax.bar_label(bars, labels=[f"{v:.1f}" if key != "trades" else str(v) for v in values], padding=3, fontsize=9)
        ax.set_xticks(x, ["Current", "Candidate", "Higher-risk\nreference"])
        ax.set_title(title); ax.spines[["top", "right"]].set_visible(False); ax.axhline(0, color="#555", linewidth=.6)
        ax.margins(y=.18)
    axes[0].legend(frameon=False)
    fig.suptitle("Fixed OOS validation · KRW25m · June 1–July 10, 2026", fontsize=15)
    fig.savefig(path, dpi=170); plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'oos')
    build(parser.parse_args().root)

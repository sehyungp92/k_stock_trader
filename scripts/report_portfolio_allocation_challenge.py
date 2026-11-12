"""Evidence report for the bounded allocation challenge; no selection or trading."""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import sys
import sqlite3
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.portfolio_allocation_challenge import read,checked,risk_metrics,BATTERY
from scripts.analyze_native_portfolio import describe,paired_bootstrap,rows
from scripts.review_native_portfolio_synergy import ordinal_conflicts,path_statistics,identity_decomposition
from scripts.replay_selected_portfolio import binding,digest,bound
from backtests.strategies.kalcb.executed_value import write_json

_VERIFIED_DATABASES=set()


def concentration_detail(case):
    study=read(read(case/'request.json')['arguments']['study']);db=Path(study['code'])/study['database']
    stat=db.stat();identity=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,study['database_hash'])
    if identity not in _VERIFIED_DATABASES:
        if digest(db)!=study['database_hash']:raise ValueError('Changed concentration price source')
        _VERIFIED_DATABASES.add(identity)
    admissions=list(rows(case/'admissions.jsonl.gz'));sectors=admissions[0]['sectors'] if admissions else {}
    connection=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);days=[]
    try:
        for d in read(case/'daily.json'):
            issuer=defaultdict(float);sector=defaultdict(float)
            for key,qty in d['positions'].items():
                sid,symbol=ast.literal_eval(key)
                value=connection.execute('SELECT c FROM bars WHERE symbol=? AND day=? AND c>0 ORDER BY ts DESC LIMIT 1',(symbol,d['day'])).fetchone()
                if value is None:raise ValueError('Missing closing mark for held position')
                issuer[symbol]+=qty*value[0];sector[sectors.get(symbol,'UNKNOWN')]+=qty*value[0]
            if abs(sum(issuer.values())+d['cash']-d['equity'])>.01:
                raise ValueError('Closing concentration does not reconcile to saved NAV: '+d['day'])
            days.append({'day':d['day'],'issuer_nav_fraction':{s:v/d['equity'] for s,v in issuer.items()},
                'sector_nav_fraction':{s:v/d['equity'] for s,v in sector.items()}})
    finally:connection.close()
    return {'max_overnight_issuer_nav_fraction':max((v for d in days for v in d['issuer_nav_fraction'].values()),default=0),
        'max_overnight_sector_nav_fraction':max((v for d in days for v in d['sector_nav_fraction'].values()),default=0),
        'max_unknown_sector_overnight_nav_fraction':max((d['sector_nav_fraction'].get('UNKNOWN',0) for d in days),default=0),
        'max_observed_prebuy_issuer_nav_fraction':max((r['exposure']['symbol']/r['equity'] for r in admissions if r['equity']>0),default=0),
        'max_observed_prebuy_sector_nav_fraction':max((r['exposure']['sector']/r['equity'] for r in admissions if r['equity']>0),default=0),
        'sessions':days,'database_sha256':study['database_hash'],
        'scope':'Exact saved session-end holdings marked at frozen last closing prices, reconciled to NAV within KRW0.01; pre-buy figures sample the order issuer/sector, not exhaustive intraday maxima. UNKNOWN denotes missing sector classification.'}


def admission_detail(case):
    events=list(rows(case/'native_budget_admissions.jsonl.gz'))
    by_order=defaultdict(list)
    for r in events:by_order[r['order']['order_id']].append(r)
    fills=read(case/'fills.json');totals=defaultdict(int)
    for f in fills:
        if f['side']=='BUY':totals[f['order_id']]+=f['qty']
    changed=[]
    for oid,attempts in by_order.items():
        r=attempts[0];requested=r['order']['qty'];filled=totals[oid]
        if filled<requested or any(x['approved_qty']<x['offered_qty'] for x in attempts):
            changed.append({'order_id':oid,'strategy':r['order']['strategy_id'],'symbol':r['order']['symbol'],
                'submitted_at':r['order']['submitted_at'],'candidate_rank':r['order']['metadata'].get('candidate_rank'),
                'first_admission_requested':requested,'lifetime_filled':filled,
                'zero_admission_attempts':sum(x['approved_qty']==0 for x in attempts),
                'funding_constraints':sorted({b for x in attempts for b in x['reservation']['binding_constraints']})})
    return {'constrained_or_incomplete_orders':changed,'count':len(changed),
        'scope':'Every resized, zero-admitted or incomplete native request; repeated callbacks collapsed. Later fills can make lifetime filled exceed first offered liquidity. These are diagnostics, never training labels.'}


def inventory_detail(case):
    daily=read(case/'daily.json');trades=read(case/'trades.json')
    ix={d['day']:i for i,d in enumerate(daily)}
    late=[t for t in trades if t['strategy_id']=='OLR' and ix[t['exit_fill_time'][:10]]-ix[t['entry_fill_time'][:10]]>1]
    fills=read(case/'fills.json');held=defaultdict(int);entry_day={};cursor=0;exposures=[]
    # Count aged issuers separately from exact total marked overnight exposure;
    # issuer/sector marked values are reconstructed by concentration_detail.
    for d in daily:
        while cursor<len(fills) and fills[cursor]['timestamp'][:10]<=d['day']:
            f=fills[cursor];key=(f['strategy_id'],f['symbol'])
            if f['side']=='BUY' and held[key]==0:entry_day[key]=f['timestamp'][:10]
            held[key]+=(1 if f['side']=='BUY' else -1)*f['qty'];cursor+=1
        aged=[k for k,q in held.items() if q>0 and k[0]=='OLR' and ix[d['day']]-ix[entry_day[k]]>=2]
        exposures.append({'day':d['day'],'aged_olr_issuers':len(aged),'all_overnight_nav_fraction':max(0.,1-d['cash']/d['equity'])})
    return {'late_final_exits':len(late),'late_exit_net_pnl':sum(t['net_pnl'] for t in late),
        'late_trades':late,'sessions_with_inventory_at_least_two_sessions_old':sum(x['aged_olr_issuers']>0 for x in exposures),
        'session_inventory':exposures,
        'interpretation':'Positive eventual delayed-exit PNL cannot excuse an interim drawdown breach. No retrospective sale, next-day signal forecast, or recovered reference winner is credited.'}


def transfer_detail(case,description):
    audit=read(case/'allocation_audit.json');transfers=audit['wealth_transfers']
    net=sum(t['kalcb_transfer'] for t in transfers)
    expected={'KALCB':12_500_000+description['pnl_by_strategy'].get('KALCB',0)+net,
        'OLR':12_500_000+description['pnl_by_strategy'].get('OLR',0)-net}
    errors={s:audit['terminal_sleeve_cash'][s]-expected[s] for s in expected}
    if any(abs(x)>.01 for x in errors.values()):raise ValueError('Terminal sleeve transfer reconciliation failed')
    if any(abs(t['kalcb_transfer'])>max(0.,t['donor_cash'])+1e-7 for t in transfers):
        raise ValueError('Transfer exceeded available donor cash')
    return {'transfer_observations':len(transfers),
        'net_transfer_to_kalcb_krw':net,
        'gross_absolute_transfers_krw':sum(abs(t['kalcb_transfer']) for t in transfers),
        'cash_constrained_rebalances':sum(abs(t['requested_transfer']-t['kalcb_transfer'])>.01 for t in transfers),
        'terminal_sleeve_cash':audit['terminal_sleeve_cash'],'terminal_reconciliation_errors_krw':errors}


def report(root,output):
    selection=read(root/'selection.json');name=selection['winner'];details={}
    canonical=read(REPO/'config/research/portfolio_synergy_selected.json')
    canonical_profile=bound(canonical['profile'])
    promoted=bool(name and canonical_profile.get('challenge_root')==str(root)
        and canonical_profile['recipe']==selection['candidates'][name]['baseline']['recipe'])
    protocol=read(root/'protocol.json');reference=Path(protocol.get('reference_challenge_root',str(root)))
    initial=read(reference/'selection.json')
    tests=read(root/'implementation_tests.json')
    test_log=Path(tests['log']);test_count=int(re.search(r'(\d+) passed',test_log.read_text(encoding='utf-8'))[1])
    outside=read(root/'canonical_verification_outside_cwd.json')
    if tests['exit_code'] or outside['exit_code']:raise ValueError('Failed final validation record')
    validation={'targeted_tests_passed':test_count,'test_record':binding(root/'implementation_tests.json'),
        'test_log':binding(test_log),'canonical_verification_outside_cwd':binding(root/'canonical_verification_outside_cwd.json'),
        'cash_transfer_fix_bridge':binding(reference/'partition_cash_funded/correction_bridge.json'),
        'incumbent_bridge':binding(reference/'incumbent_bridge.json'),
        'registered_search_validation':binding(root/'registered_search_validation.json')}
    all_candidates={**initial['candidates'],**selection['candidates']}
    for n,battery in all_candidates.items():
        for scenario in ('baseline','combined'):
            case=Path(battery[scenario]['case'])
            details[n+'_'+scenario]={'description':describe(case),'ordinal_conflicts':ordinal_conflicts(case),
                'allocation':admission_detail(case),'path_statistics':path_statistics(case)}
    partitions={}
    partition_ref=read(reference/'partition_study.json');partition_root=Path(partition_ref['path']).parent
    if digest(partition_ref['path'])!=partition_ref['sha256']:raise ValueError('Changed partition study')
    for mode in ('shared','partition','daily_rebalanced'):
        partitions[mode]={s:describe(partition_root/'cases'/(mode+'_'+s)) for s in ('baseline','combined')}
        for s,v in partitions[mode].items():v['wealth_transfer_reconciliation']=transfer_detail(partition_root/'cases'/(mode+'_'+s),v)
    standalone={}
    controls=Path.home()/'.codex/research/portfolio_native_budget_20261009/generation_3/cases'
    for control in ('control_k100','control_k200','control_o100_r1'):
        standalone[control]={}
        for s in ('baseline','combined'):
            case=controls/(control+'_'+s);m=checked(case)
            standalone[control][s]={**{k:m[k] for k in ('return_pct','drawdown_pct','trades')},
                'completion':binding(case/'completion.json')}
    inventory={};uncertainty={};identity_changes={}
    if name:
        for label,m in selection['candidates'][name].items():
            inventory[label]=inventory_detail(Path(m['case']))
            inventory[label]['concentration']=concentration_detail(Path(m['case']))
        for scenario in ('baseline','combined'):
            case=Path(selection['candidates'][name][scenario]['case'])
            old=Path(initial['candidates']['k075_o075'][scenario]['case'])
            uncertainty[scenario]=paired_bootstrap(case,old)
            identity_changes[scenario]={sid:identity_decomposition(case,old,sid) for sid in ('KALCB','OLR')}
            for v in identity_changes[scenario].values():
                v['interpretation']='Selected portfolio minus incumbent portfolio accounting identity. Same entry identities can have different quantities, fills, exits and subsequent compounding. Hindsight PNL does not establish expected trade quality.'
    value={'created_utc':datetime.now(timezone.utc).isoformat(),'selection':selection,'initial_challenge':initial,'diagnostics':details,
        'partition_comparison':partitions,'partition_study':partition_ref,'matched_standalones':standalone,
        'selected_inventory':inventory,'paired_uncertainty_vs_incumbent':uncertainty,
        'identity_changes_vs_incumbent':identity_changes,
        'protocol':binding(root/'protocol.json'),'qualification':binding(root/'qualification.json') if name else None,
        'generator':binding(Path(__file__)),'canonical_pointer_at_report':binding(REPO/'config/research/portfolio_synergy_selected.json'),
        'canonical_profile_at_report':canonical['profile'],'selected_is_canonical_default':promoted}
    value['validation']=validation
    write_json(output.with_suffix('.json'),value)
    comparison=[]
    if name:
        comparison=['','Selected versus preserved incumbent:','Scenario | selected return / DD / entries | incumbent return / DD / entries']
        for s in BATTERY:
            a=selection['candidates'][name][s];b=initial['candidates']['k075_o075'][s]
            comparison.append(f'{s} | {a["return_pct"]:.4f}% / {a["drawdown_pct"]:.4f}% / {a["trades"]} | {b["return_pct"]:.4f}% / {b["drawdown_pct"]:.4f}% / {b["trades"]}')
        a=selection['candidates'][name];b=initial['candidates']['k075_o075']
        comparison.append(f'Across all eight scenarios: selected maximum DD {max(m["drawdown_pct"] for m in a.values()):.4f}% versus {max(m["drawdown_pct"] for m in b.values()):.4f}%; minimum entries {min(m["trades"] for m in a.values())} versus {min(m["trades"] for m in b.values())}; maximum mean overnight capital {max(m["overnight_mean"] for m in a.values())*100:.3f}% versus {max(m["overnight_mean"] for m in b.values())*100:.3f}%.')
        comparison.append(f'Maximum overnight exposure across the full battery: selected {max(m["overnight_max"] for m in a.values())*100:.3f}% NAV versus incumbent {max(m["overnight_max"] for m in b.values())*100:.3f}%. Baseline/combined return changes: {a["baseline"]["return_pct"]-b["baseline"]["return_pct"]:+.4f}pp / {a["combined"]["return_pct"]-b["combined"]["return_pct"]:+.4f}pp. Any negative changes are an explicit return sacrifice, not a return improvement.')
        comparison.append(f'A static uniform 20% gap against the maximum observed overnight book implies approximately {max(m["overnight_max"] for m in a.values())*20:.3f}% NAV loss for the selected mix versus {max(m["overnight_max"] for m in b.values())*20:.3f}% for the incumbent, before liquidation costs or other losses. This is an exposure calculation, not a replay, forecast or loss bound.')
        comparison.append(f'Selection rule: selected minimum adverse return {selection["adverse_return_floor"][name]:.4f}%; highest eligible minimum adverse return {max(selection["adverse_return_floor"].values()):.4f}%. The one-percentage-point tie band and frequency/drawdown tie breaks apply after all risk gates.')
        comparison.append(f'Observed headroom to the drawdown gates: baseline {11-a["baseline"]["drawdown_pct"]:.4f}pp; worst adverse case {12.5-max(m["drawdown_pct"] for s,m in a.items() if s!="baseline"):.4f}pp. Small headroom is disclosed rather than treated as a forward safety margin.')
    lines=['PORTFOLIO ALLOCATION AND ADVERSE-EXECUTION CHALLENGE — 2026-10-09',
        'Starting capital: KRW25,000,000. Frozen observation window: 2025-08-07 through 2026-03-31.',
        'All reported returns are net cumulative development backtest returns, not annualized or forecasts.',
        'The common cost model charges 2bp commission per side, 20bp sell tax, and baseline slippage of 3bp for KALCB / 5bp for OLR. Cost stress raises slippage to 13bp / 15bp. Participation stress halves the prior-completed-five-minute-volume cap from 1% to 0.5%; combined stress applies both changes.',
        'The eight-scenario battery contains baseline, cost, participation and combined, plus an additional adverse 15bp OLR auction-price perturbation on baseline/combined and a 50% OLR auction-nonfill perturbation on baseline/combined. Extra auction-price and nonfill perturbations are separate tests; they are not jointly stacked in this battery. All scenarios reuse the same 156-session account window, including two no-entry liquidation-tail sessions.',
        '', 'DECISION',f'Selected finite-battery candidate: {name}. Feasible candidates: {selection["feasible"]}.',
        'Publication status: selected candidate is the canonical research default; live deployment remains disabled.' if promoted else 'Publication status: this selected candidate is not the canonical research default at report generation.',
        'The previous profile remains preserved as an immutable benchmark. A new default may be published only after every selected battery case passes full native source/account qualification and exactly reproduces its screening trades, daily NAV and bar NAV.',
        *comparison,
        '', 'WHAT CHANGED',
        'Native allocation now supports symbol order, native ordinal priority, and integer proportional grants among orders from the same strategy and submission timestamp. Earlier timestamps and existing strategy precedence remain intact. Native strategy signal generation, exit rules, order quantities at submission, and holding lifecycles are preserved.',
        'The proportional rule progressively raises completion fractions, credits already filled shares, freezes only requests using a saturated resource, and redistributes feasible residual resources. Integer remainders use native rank. Holdings and pending requests consume the same physical, sleeve, issuer, sector and planned-risk resources. Actual execution price and fees are checked again. An unfilled exit supplies no cash.',
        'Reservation and physical execution ordering use the same configured key. The independent audit reconstructs cash, holdings, remaining requests and already filled quantities from journal events. Two fixed hash orderings (seeds 11 and 29) are sensitivity diagnostics and cannot win selection.',
        '', 'PROSPECTIVE RISK AND SELECTION RULE',
        'Baseline drawdown <=11%; every other battery scenario <=12.5%; maximum overnight capital <=80% NAV; mean overnight capital <=30%; worst marked session loss <=4.5%; each scenario >=515 economic entries. These are empirical screening gates, not future loss limits. The 80% ceiling retains limited headroom above the incumbent 75% overnight allocation; it is an explicit aggressive research stance, not a demonstrated tolerance for every possible market gap.',
        'Among eligible candidates, maximize the minimum adverse-scenario return. Within one percentage point, prefer minimum scenario frequency and then smaller worst drawdown. Rank/proportional promotion also requires a material return gain without lower minimum frequency or materially worse drawdown at the same budget. Random seeds are excluded.',
        'The numerical risk/activity limits are research assumptions used to operationalize the requested risk stance; they are not uniquely implied by that preference. Maximizing worst-adverse-scenario return is a selection heuristic, not an estimate or proof of maximum expected future returns.',
        'The linked challenge protocol is the selection authority. Generation registrations also retain legacy selection metadata from the shared freeze helper; those inherited parent-search thresholds are not used by this challenge. Publication reconstructs the actual protocol gates and selection from all completed cases.',
        'The original grid contains K/O sizing pairs .75/.75, .50/1.00 and .50/.75. All fail at least one unchanged overnight/risk gate. Consequently its planned K-only refinement (.625/.75 and .6875/.75) did not trigger and was not run.',
        'A separate adaptive follow-up was registered after observing those failures, before observing its own results. It tests K=.75/.875 against O=.50/.625. For each O fraction, one K=.8125 midpoint is allowed only if K=.75 passes every gate and K=.875 fails at least one. All original risk and activity limits stay unchanged. This targets the actual binding constraint—overnight OLR capital—while retaining bounded intraday aggression. It is adaptive development research, not a claim that the entire enlarged search was registered before the first challenge.',
        'These are native sizing/budget factors, not portfolio weights required to sum to one. KALCB can use modeled intraday margin. The OLR factor controls its native account/budget view.',
        '', 'ALL REGISTERED RESULTS',
        'Candidate | scenario | net return % | max DD % | entries | max overnight % | mean overnight % | worst session % | gate failures']
    for n,battery in all_candidates.items():
        for s,m in battery.items():
            failures=','.join(k for k,v in m['gate']['tests'].items() if not v) or 'pass'
            lines.append(f'{n} | {s} | {m["return_pct"]:.4f} | {m["drawdown_pct"]:.4f} | {m["trades"]} | {m["overnight_max"]*100:.3f} | {m["overnight_mean"]*100:.3f} | {m["worst_session_pct"]:.3f} | {failures}')
    lines+=['','MATCHED STANDALONE REFERENCES','Control | scenario | return % | DD % | entries']
    for control,battery in standalone.items():
        for s,m in battery.items():lines.append(f'{control} | {s} | {m["return_pct"]:.4f} | {m["drawdown_pct"]:.4f} | {m["trades"]}')
    lines+=['Each standalone has its own KRW25m account under the same frozen market window and costs. These are comparative strategy paths, not components whose returns can be added into one KRW25m portfolio. Only baseline/combined are claimed here; standalone qualification on the extra auction scenarios was not performed.']
    lines+=['','ALLOCATION QUALITY AND FREQUENCY','Candidate/scenario | ordinal-inversion sessions | resized/incomplete request count']
    for n,v in details.items():lines.append(f'{n} | {len(v["ordinal_conflicts"]["distinct_sessions"])} | {v["allocation"]["count"]}')
    if name:
        selected_days=set().union(*(set(details[name+'_'+s]['ordinal_conflicts']['distinct_sessions']) for s in ('baseline','combined')))
        incumbent_days=set().union(*(set(details['k075_o075_'+s]['ordinal_conflicts']['distinct_sessions']) for s in ('baseline','combined')))
        lines.append(f'The selected baseline/combined paths retain {len(selected_days)} distinct ordinal-conflict sessions versus {len(incumbent_days)} for the incumbent: {sorted(selected_days)}. Repeated sessions across scenarios count once. Thus the observed conflict count falls, but preferential admission of stronger expected-value trades remains unestablished.')
        for s in ('baseline','combined'):
            a=details[name+'_'+s]['description']['result']['strategy_counts']
            b=details['k075_o075_'+s]['description']['result']['strategy_counts']
            lines.append(f'{s}, selected/incumbent entry counts: KALCB {a["KALCB"]}/{b["KALCB"]}; OLR {a["OLR"]}/{b["OLR"]}. Minimum battery activity and baseline activity are different measures; neither should be substituted for the other.')
    lines+=['','Selected minus incumbent accounting decomposition (KRW):',
        'Scenario / sleeve | matched entries | total PNL change | change on matched identities | new minus missing identities']
    for s,sleeves in identity_changes.items():
        for sid,v in sleeves.items():
            lines.append(f'{s} / {sid} | {v["matched_entries"]} | {v["total_sleeve_pnl_change_krw"]:,.0f} | {v["matched_entry_pnl_change_krw"]:,.0f} | {v["new_minus_missing_pnl_krw"]:,.0f}')
        lines.append(f'{s}, combined entry-set PNL component: KRW{sum(v["new_minus_missing_pnl_krw"] for v in sleeves.values()):+,.0f}; PNL change on retained identities: KRW{sum(v["matched_entry_pnl_change_krw"] for v in sleeves.values()):+,.0f}.')
    lines+=['Each row reconciles within KRW0.01. This distinguishes changed quantities/executions on retained opportunities from entry-set changes; it does not infer the expected profitability of rejected signals from their realized outcomes.']
    lines+=['Native-rank and proportional alternatives failed their registered promotion criteria at the incumbent .75/.75 budget. This does not establish their inferiority at every other budget. The follow-up sizing search retains symbol ordering; it does not establish preferential admission of stronger expected-value trades. No additional allocation-by-budget factorial was run.',
        'A lower ordinal rank is not a calibrated expected-profit estimate. Removal of rank inversions establishes mechanical priority, not preferential rejection of future losing trades. Every tested rule was replayed through the complete account, allowing changes in future sizing, inventory and signal eligibility. Shadow outcomes were not used to train or reorder orders.',
        '', 'FUNDED CAPITAL AND WEALTH TRANSFER',
        'Each partitioned account begins with KALCB KRW12.5m and OLR KRW12.5m, totaling KRW25m. Separate sleeve cash ledgers accumulate the actual commissions, sell taxes, realized flows and marked holdings. Both books share one physical broker and same-phase participation budget, so simultaneous trades cannot consume the same market volume twice.',
        'The partition case retains profits/losses in the originating sleeve. The daily-rebalanced case moves wealth toward 50/50 before the first opening mark, using only previously observed prices and positive actual donor cash. The target can remain partly unmet while capital is invested. Every transfer is zero-sum and independently reconstructed; no unrealized gain or prospective liquidation proceeds fund a transfer. Neither case knows next-day signals. The shared .50/.50 comparator has the same initial sizing and holding budgets (KALCB KRW25m, OLR KRW12.5m), but later sizes both sleeves from total portfolio NAV. Its KALCB holding-budget factor is 1.0; the partitioned factor is 2.0 times that sleeve\'s half of account wealth.',
        'Mode | scenario | return % | DD % | entries']
    for mode,battery in partitions.items():
        for s,v in battery.items():
            m=v['result'];lines.append(f'{mode} | {s} | {m["return_pct"]:.4f} | {m["drawdown_pct"]:.4f} | {m["trades"]}')
    for s in ('baseline','combined'):
        base=partitions['partition'][s]['result']
        for mode in ('shared','daily_rebalanced'):
            m=partitions[mode][s]['result']
            lines.append(f'{s}, {mode} minus fixed partition: return {m["return_pct"]-base["return_pct"]:+.4f}pp; DD {m["drawdown_pct"]-base["drawdown_pct"]:+.4f}pp; entries {m["trades"]-base["trades"]:+d}.')
        t=partitions['daily_rebalanced'][s]['wealth_transfer_reconciliation']
        lines.append(f'{s}, daily rebalancing: net KRW{t["net_transfer_to_kalcb_krw"]:,.0f} transferred to KALCB; {t["cash_constrained_rebalances"]} target transfers limited by donor cash. Both terminal sleeve balances reconcile to initial funding plus own PNL and actual transfers within KRW0.01.')
    lines+=['These are full-account funding counterfactuals. They separate wealth sharing from simply adding the PNL of two accounts each funded with KRW25m. They are diagnostic benchmarks, not newly eligible strategy candidates. Screen audits establish physical/account validity; only the selected eight-scenario profile receives the full source-qualification battery in this round.',
        'Benchmark implementation corrections were recorded before interpretation: the initial unused freeze had mismatched KALCB holding budgets; partition_v2 matched those budgets but allowed an uncapped marked-wealth transfer that could overdraw an invested donor sleeve. Its outputs are preserved as superseded diagnostics. The reported partition_cash_funded generation caps transfers at positive donor cash and reruns all six benchmark cases. The portfolio-selection generations and recipes were unaffected.',
        '', 'PERSISTENT OLR INVENTORY']
    for s,v in inventory.items():
        c=v['concentration']
        lines.append(f'{s}: {v["late_final_exits"]} entries exit later than the next session; eventual net PNL KRW{v["late_exit_net_pnl"]:,.0f}; {v["sessions_with_inventory_at_least_two_sessions_old"]} session ends hold OLR inventory at least two sessions old. Maximum overnight issuer/sector exposure: {c["max_overnight_issuer_nav_fraction"]*100:.2f}%/{c["max_overnight_sector_nav_fraction"]*100:.2f}% NAV; maximum UNKNOWN-sector exposure {c["max_unknown_sector_overnight_nav_fraction"]*100:.2f}%.')
    lines+=['The adverse nonfill scenario explicitly carries unliquidated positions into later sessions; the account replay reflects their funding use and existing-position guard. This addresses the interaction honestly without treating tomorrow\'s actual KALCB signals as known today. A profitable eventual exit can still deepen intervening drawdown.',
        'Causal persistent-inventory policy experiments remain outstanding. This round implements exposure/holding-state diagnostics and delayed-liquidation stress, not the proposed inventory-policy experiment itself. No OLR renewal/multi-lot or portfolio-driven forced-exit override was tested or promoted. Those changes alter lifecycle, fees, holding risk and future eligibility; recovering a known historical winner is insufficient evidence.',
        '', 'CHRONOLOGY, UNCERTAINTY AND EXPECTED RETURNS',
        'The JSON records continuous-account monthly marked returns, the pre-2026 return, tail sessions, PNL concentration, and session-block paired uncertainty. Position continuity is preserved throughout the replay. No new expected-value classifier was fitted, so no future trade outcomes enter the new allocation rule. The frozen strategies and budget search still use an already studied development period; slicing it again does not create untouched validation.',
        'Full chronological refit-and-select validation and validation on genuinely unused compatible data remain outstanding. Descriptive calendar slices and resampling do not complete those recommendations. Broker funding/execution parity is also unestablished. This round does not invent an out-of-sample period or an estimated probability of backtest overfitting. Repeated search can produce false discoveries: Bailey et al., The Probability of Backtest Overfitting (2015), https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf .',
        'Next bounded evidence step, not launched here: freeze the selected and incumbent profiles, obtain an unused compatible evaluation segment, and preregister one causal inventory-policy comparator using only observable holding age, current liquidity and actual funding. Fit/select only on preceding chronological segments and evaluate all accounts on the same subsequent segment. Keep that work separate from this completed sizing round.']
    for s,v in uncertainty.items():lines.append(f'{s}: paired five-session bootstrap 95% return-difference interval versus incumbent = {v["paired_5day_bootstrap_return_difference_pp_95pct"]} percentage points; conditional reused-data resampling, not selection-adjusted confidence in future outperformance.')
    lines+=['','INTERPRETATION AND LIMITS',
        'A risk-feasible replacement can intentionally give up headline return to remedy an adverse-execution failure. The selected result is the registered choice within a finite tested set, not proof of a global maximum or of maximum expected future returns. A true expected-value allocator would need calibrated decision-time incremental net value, capital duration and liquidation-risk estimates with stronger support than the five previously observed ordinal-conflict sessions.',
        'Modeled intraday borrowing remains subject to actual broker availability; separate financing charges are absent. Native planned stop-risk units are not guaranteed losses. The existing gross/name/sector admission ceilings do not force liquidation after price changes, and extreme overnight gaps can exceed observed drawdowns. These limitations constrain interpretation of the aggressive stance.',
        'Live deployment remains disabled. No live strategy route, broker order or source strategy core was changed.',
        '', 'COMPLETED VALIDATION',
        f'{test_count} targeted software tests passed. All eight selected scenarios passed full native source/account qualification and exactly matched their screening trades, daily NAV and bar NAV. The canonical profile was verified from outside the repository working directory.',
        'This round completed 80 registered allocation/budget screens, six corrected funded-account benchmarks and eight selected full qualifications. Six superseded benchmark runs are preserved separately: 100 completed new replays in total. These are repeated experiments on the same historical observations, not 100 independent performance samples.',
        'The incumbent baseline/combined economic paths remain byte-identical across the allocator source bridge. The cash-transfer correction also leaves all four shared/fixed-partition trade and equity paths byte-identical. Frozen input bytes and evidence hashes were checked during publication. Validation records and logs are bound in the companion JSON.',
        '', 'REPRODUCIBILITY',f'Experiment root: {root}',f'Protocol: {root/"protocol.json"}',f'Selection: {root/"selection.json"}',
        f'Qualification: {root/"qualification.json"}',f'Detailed results and diagnostics: {output.with_suffix(".json")}',
        'Run commands: python scripts/portfolio_allocation_challenge.py prepare/run/partition/refine/select/qualify --root <appropriate root>; registered generations and outputs are immutable; existing completed replays are verified before reuse.',
        'Canonical replay: python scripts/replay_selected_portfolio.py verify; replay --scenario <declared scenario> --output <new directory>. Auction-price and nonfill cases are available through the new profile as well as the four primary scenarios.']
    output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plot(value,output.with_suffix('.png'))
    print(json.dumps({'report':str(output),'selected':name,'diagnostic_cases':len(details)}))


def plot(report,output):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(14,6),gridspec_kw={'width_ratios':[1,1.55]},layout='constrained')
    candidates={**report['initial_challenge']['candidates'],**report['selection']['candidates']};winner=report['selection']['winner']
    table_rows=[];colors=[]
    for name,b in candidates.items():
        if set(b)!=set(BATTERY):continue
        dd=max(m['drawdown_pct'] for m in b.values());ret=min(m['return_pct'] for s,m in b.items() if s!='baseline')
        freq=min(m['trades'] for m in b.values());overnight=max(m['overnight_max'] for m in b.values())*100
        color='#087f5b' if name==winner else '#c92a2a' if not all(m['gate']['pass'] for m in b.values()) else '#4263eb'
        number=len(table_rows)+1
        base=b['baseline'];budget=read(base['study'])['candidates'][base['recipe']]['native_budget']
        label=f'{budget["kalcb_sizing_fraction"]:g} / {budget["olr_fraction"]:g}'
        axs[0].scatter(dd,ret,s=130 if name==winner else 65,c=color,marker='*' if name==winner else 'o')
        axs[0].annotate(str(number),(dd,ret),xytext=(6,5),textcoords='offset points',fontsize=10)
        table_rows.append([number,label,f'{freq}',f'{overnight:.1f}%',f'{max(m["overnight_mean"] for m in b.values())*100:.1f}%',
            'selected' if name==winner else 'pass' if all(m['gate']['pass'] for m in b.values()) else 'fail'])
        colors.append(color)
    axs[0].axvline(12.5,ls='--',c='grey',lw=1);axs[0].set(xlabel='Worst battery drawdown (%)',ylabel='Minimum adverse-scenario net return (%)',title='Registered return / adverse-risk frontier')
    axs[0].margins(.14);axs[0].grid(alpha=.2);axs[0].spines[['top','right']].set_visible(False)
    axs[1].axis('off');axs[1].set_title('Full eight-scenario candidates: activity and overnight exposure',pad=16)
    table=axs[1].table(cellText=table_rows,colLabels=['ID','KALCB / OLR factor','Min.\nentries','Max.\novernight','Max. mean\novernight','All gates'],
        colWidths=[.05,.34,.12,.16,.17,.16],cellLoc='center',bbox=[0,.15,1,.76])
    table.auto_set_font_size(False);table.set_fontsize(9)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#dddddd')
        if row==0:cell.set_facecolor('#edf2f7');cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('#f8f9fa' if row%2 else 'white')
            if col in (0,5):cell.set_text_props(color=colors[row-1],weight='bold')
    axs[1].text(0,.08,'Gates: minimum 515 entries; maximum overnight 80%; maximum mean overnight 30%.\nAlso required: baseline DD ≤11%, other DD ≤12.5%, worst session loss ≤4.5%.',transform=axs[1].transAxes,fontsize=9,va='top')
    fig.suptitle('KRW25m | Reused development data | Green = selected; blue = feasible; red = failed gate',fontsize=12)
    fig.savefig(output,dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report(a.root.resolve(),a.output.resolve())

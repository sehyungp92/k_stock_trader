"""Produce the final, evidence-bound optimization report and review bundle."""
from __future__ import annotations
import difflib,html,json,shutil,sys
from html.parser import HTMLParser
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import olr_boundary_restart_results_v1 as r
b=r.b;s=b.s;n=b.n
OUT=s.OUT


def number(v,d=2):return f'{v:,.{d}f}'
def label(value):
    return {'literal_carry_original':'Original rules + IS history carry',
        'literal_carry_guard_1p0':'IS history carry + 1.0R guard',
        'literal_carry_full_panel_guard_1p0':'History carry + corrected sector panel + 1.0R guard',
        'broad_declared_universe_full_panel':'Broad auction history + corrected sector panel',
        'broad_declared_universe_full_panel_guard_1p0':'Broad history + corrected sector panel + 1.0R guard',
        'literal_carry_original_scoring_guard_0.8':'IS history carry + 0.8R guard',
        'literal_carry_original_scoring_guard_1.2':'IS history carry + 1.2R guard'}.get(value,str(value).replace('_',' '))
def window(value):return 'IS' if value=='train' else 'OOS'
def table(headers,rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def p(value):return '<p>'+value+'</p>'
def h(value):return '<h2>'+value+'</h2>'
def money(v):return f'₩{v/1e6:+.3f}m'


def export_bundle(selected):
    destination=OUT/'review_bundle';destination.mkdir(exist_ok=True)
    files={}
    for w in ('train','oos'):
        before=n.read(s.PRIOR/'registrations'/w/'c272fe0390df855bd_100000000/replay_study.json')
        old=Path(before['code']);new=Path(n.read(b.ENGINES/w/'ready.json')['files'])
        patch=[]
        for name in ('strategy_olr/config.py','strategy_olr/research.py','scripts/portfolio_synergy_study.py'):
            a=(old/name).read_text(encoding='utf-8').splitlines(keepends=True);z=(new/name).read_text(encoding='utf-8').splitlines(keepends=True)
            patch.extend(difflib.unified_diff(a,z,fromfile='a/'+name,tofile='b/'+name))
            files[w+'/'+name]={'before':n.sha(old/name),'after':n.sha(new/name)}
        (destination/(w+'_frozen_source_changes.patch')).write_text(''.join(patch),encoding='utf-8')
    for name in ('selected_research_policy.json','decision_summary.json','native_run_inventory.json','literal_restart_contract_checks.json','cross_window_olr_implementation_parity.json',
                 'panel_oos_exact_economic_parity.json','literal_carry_original_is_exact_parity.json','literal_carry_original_stress_is_exact_parity.json','literal_carry_guard_1p0_is_exact_parity.json','boundary_restart_qualification.json',
                 'boundary_restart_paired_results.json','boundary_restart_stress_results.json','boundary_restart_no_upward_compounding.json',
                 'boundary_restart_diagnostics.json','boundary_restart_exit_and_quantity_attribution.json'):
        if (OUT/name).exists():shutil.copy2(OUT/name,destination/name)
    for name in ('registration.json','qualification_protocol.json','feature_adoption_amendment.json','guard_neighborhood_registration.json','stress_finalist_registration.json',
                 'boundary_validation_train.json','boundary_validation_oos.json','preparation_complete.json','qualification_completion.json'):
        if (b.ROOT/name).exists():shutil.copy2(b.ROOT/name,destination/name)
    shutil.copy2(b.x.BUILDER,destination/b.x.BUILDER.name)
    shutil.copy2(REPO/'data/backtests/analysis/olr_exit_diagnosis_20261010/analyze.py',destination/'observed_fill_accounting_helper.py')
    for name in ('observation_continuity.json','continuous_history_verification.json','completion.json'):
        shutil.copy2(OUT/'history_check'/name,destination/('warm_signal_'+name))
    n.write(destination/'bundle_manifest.json',{'created_utc':s.utc(),'status':'research_only','selected':selected,
        'native_root':str(b.ROOT),'source_engines':str(b.ENGINES),'source_changes':files,
        'instruction':'Review these patches against their exact before hashes. Do not blindly apply them to a divergent working tree. Reproduce through the bound native study; copying exit parameters alone omits the declared scoring mode and market-history handoff.',
        'file_hashes':{q.name:n.sha(q) for q in destination.iterdir() if q.is_file() and q.name!='bundle_manifest.json'}})


def chart(catalog,chosen,prefix='optimization_results'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    policies=[('Original',None,'#64748b'),('Continuous control',next(v for v in catalog.values() if v['label']=='literal_carry_original'),'#d97706')]
    plotted=chosen or next(c['id'] for c in catalog.values() if c['label']=='literal_carry_guard_1p0')
    candidate_label='Selected policy' if chosen else '1.0R nominal candidate'
    policies.append((candidate_label,catalog[plotted],'#047857'))
    fig,axes=plt.subplots(2,2,figsize=(13,8.4),layout='constrained')
    for wi,w in enumerate(('train','oos')):
        ax=axes[0,wi]
        for label,c,color in policies:
            folder=b.output(c,w) if c else s.PARENT/'cases'/s.forensic.BASE/w/'100000000_baseline'
            rows=n.read(folder/'daily.json')
            ax.plot(range(len(rows)+1),[0,*[100*(d['equity']/1e8-1) for d in rows]],label=label,color=color,lw=2)
        ax.set(title=('IS: 7 Aug 2025 – 31 Mar 2026' if w=='train' else 'OOS: 1 Apr – 10 Jul 2026'),xlabel='Exchange session',ylabel='Net account return (%)')
        ax.axhline(0,color='#cbd5e1',lw=.8);ax.grid(alpha=.15);ax.legend(fontsize=8,loc='best')
    labels=[];returns=[];dds=[]
    for label,c,color in policies:
        folder=b.output(c,'oos') if c else s.PARENT/'cases'/s.forensic.BASE/'oos/100000000_baseline'
        a=n.read(folder/'assessment.json');labels.append(label);returns.append(a['return_pct']);dds.append(a['drawdown_pct'])
    x=__import__('numpy').arange(len(labels));axes[1,0].bar(x-.18,returns,.36,label='Net return',color='#047857');axes[1,0].bar(x+.18,dds,.36,label='Maximum intraday drawdown',color='#b91c1c')
    axes[1,0].set(xticks=x,xticklabels=labels,ylabel='Percent',title='OOS payoff and drawdown');axes[1,0].legend(fontsize=8);axes[1,0].axhline(0,color='#64748b',lw=.7)
    if plotted:
        data=n.read(OUT/'boundary_restart_diagnostics.json');base=data['original']['oos']['calendar'];new=data['candidates'][plotted]['windows']['baseline']['oos']['calendar']
        months=list(base);x=np.arange(len(months));axes[1,1].bar(x-.18,[base[m]['return_pct'] for m in months],.36,label='Original',color='#64748b');axes[1,1].bar(x+.18,[new[m]['return_pct'] for m in months],.36,label=candidate_label,color='#047857')
        axes[1,1].set(xticks=x,xticklabels=[m[5:] for m in months],xlabel='Month in 2026 (July through the 10th)',ylabel='Monthly NAV return (%)',title='Where the change occurs');axes[1,1].legend(fontsize=8);axes[1,1].axhline(0,color='#64748b',lw=.7)
    fig.suptitle('OLR round 1 — continuous-history optimization',fontsize=17,fontweight='bold')
    fig.savefig(OUT/(prefix+'.png'),dpi=170,facecolor='white');fig.savefig(OUT/(prefix+'.svg'));plt.close(fig)


def main():
    assert (b.ROOT/'qualification_completion.json').exists(),'Qualification must finish before final reporting'
    rows=r.pairs();decision=n.read(OUT/'boundary_restart_qualification.json');chosen=decision['selected']['candidate_id'] if decision.get('selected') else None
    catalog={c['id']:c for c in n.read(b.ROOT/'candidate_plan.json')['candidates']}
    diagnostics=n.read(OUT/'boundary_restart_diagnostics.json');accounting=n.read(OUT/'boundary_restart_exit_and_quantity_attribution.json')
    stress=n.read(OUT/'boundary_restart_stress_results.json') if (OUT/'boundary_restart_stress_results.json').exists() else []
    capped=n.read(OUT/'boundary_restart_no_upward_compounding.json') if (OUT/'boundary_restart_no_upward_compounding.json').exists() else []
    selected=next((v for v in rows if v['candidate_id']==chosen),None)
    analysis_row=selected or max((v for v in rows if v['eligible']),default=None,
        key=lambda v:(v['both_returns_improved'],min(v[w]['return_gain_pp'] for w in ('train','oos')),v['oos']['return_pct']))
    analysis_id=analysis_row['candidate_id'] if analysis_row else None
    inventory=[]
    for root in (s.ROOT,b.x.ROOT,b.ROOT):
        for path in sorted((root/'cases').glob('**/assessment.json')):
            a=n.read(path);inventory.append({'assessment':str(path),'sha256':n.sha(path),'qualified':a.get('qualified',False),
                'label':a.get('label'),'window':a.get('window'),'scenario':a.get('scenario'),'capped':a.get('capped',False)})
    n.write(OUT/'native_run_inventory.json',{'runs':inventory,'completed_assessments':len(inventory),
        'qualified':sum(v['qualified'] for v in inventory),'excluded':sum(not v['qualified'] for v in inventory),
        'interpretation':'Distinct completed native replay assessments in the three executed follow-up roots; includes controls, scenarios and repeated configurations. Not independent statistical trials.'})
    chart(catalog,chosen)
    parts=['<h1>OLR round 1: continuous-history repair and optimization</h1>',p('Evidence as of '+s.utc()+'. IS: 7 August 2025–31 March 2026 (156 sessions). Historical OOS: 1 April–10 July 2026 (69 sessions). Each standalone OLR account starts flat with ₩100 million. Returns include the native commissions, tax, slippage and volume constraints. OOS is now development data because it has informed the changes.')]
    if selected:
        parts+=[p(f'<strong>Selected research successor: {html.escape(label(selected["label"]))}.</strong> IS return is {number(selected["train"]["return_pct"])}% ({selected["train"]["return_gain_pp"]:+.2f} percentage points versus round 1); OOS return is {number(selected["oos"]["return_pct"])}% ({selected["oos"]["return_gain_pp"]:+.2f} points). OOS maximum intraday drawdown is {number(selected["oos"]["drawdown_pct"])}%. It passed the registered nominal and combined execution-stress gates. The result supports a research successor, not a claim of global optimality or independent future profitability.')]
        if not selected['mutations'].get(b.x.PANEL_KEY):
            parts+=[p('<strong>Scoring compatibility is deliberate.</strong> The selected policy retains the original scoring formula and the independently repaired native-history handoff. The corrected current-afternoon sector-volume formula is implemented and causally validated, but its activation failed the IS preservation gate. It remains an explicit disabled research treatment, not an unreported fix in the selected policy. The conditional fallback and its trigger were recorded in feature_adoption_amendment.json at 17:13 UTC after observing that panel failure and before the neighborhood search. This was an adaptive amendment, not a wholly precommitted feature-selection decision; the return, drawdown and stress gates were unchanged.')]
    else:parts+=[p('<strong>No candidate passed the complete preservation and stress gates.</strong> The report retains the improvements that were observed and explains the remaining trade-offs; it does not label an unqualified variant a successful replacement.')]
    parts+=[p('The August start is a scoring-calendar restriction inherited from the shared KALCB study. It is not the start of the retained intraday history. A real liquidity-state reset existed at the OOS boundary; fixing that alone moved OOS from −3.3277% to −3.5440%, while leaving IS exactly unchanged. It therefore does not explain the large performance collapse. The dominant economic problem remains concentrated adverse trades and an exit rule that needed favorable excursion before protecting them.'),'<img src="optimization_results.png" alt="Verified account-return paths, OOS drawdown and calendar comparison">']
    parts+=[h('1. What was actually fixed'),p('Three distinct histories must be kept separate. First, the ordinary lagged signal features already used older intraday bars. Second, the current-afternoon sector panel was rebuilt from only the current session, making its sector relative-volume input degenerate. Third, the native liquidity forecaster started with empty state in each independent replay. Checking only the first layer gave an incomplete answer to the warm-up question. This follow-up corrects that limitation in the earlier review.')]
    parts+=[table(['Layer','Evidence','Implemented treatment'],[
        ['Raw / lagged signal history','IS and OOS share exactly 1,977,054 pre-April OHLCV rows over 217 sessions','Retain all available earlier sessions; score only the requested dates'],
        ['IS scoring start','61 retained sessions before 7 Aug; shared 60-training-session + one-embargo schedule','Keep the official scoring window; separate history start from scoring start'],
        ['Current-afternoon sector history','Old selected sector relative volume was 1.0; causal correction changes scores on 100 IS / 35 OOS dates','Use the source-bound complete prior-session panel at the same decision cutoff'],
        ['Literal native state','Restore actual IS terminal capacity state, with matching candidate / scenario / account mode','OOS starts on 1 Apr with IS history, independently reset cash and a verified flat parent'],
        ['Broader market observation alternative','Declared 103-stock universe; 6,262 prior IS auction rows / 22,288 prior OOS rows','Separately test full-universe auction warm-up and continued observation']])]
    parts+=[p('For the literal baseline, 1,156 retained market observations across 103 symbols are restored. The 1.0R variant has 1,151 because its native routing and holding paths differ. These are the observations actually retained by each original policy, not every bar in the database. All are verified against the OOS retained source. Matching the parent configuration, transaction-cost scenario, participation scenario, compounding mode and initial equity is mandatory. The parent must have a passed native audit and no positions or pending orders. No account equity, open position, order or future outcome is copied.'),p('The broader policy is a separate economic change: it seeds 61 pre-IS sessions and 217 pre-OOS sessions, including all 156 IS sessions, and observes the same declared universe afterward. It retains the native last 20 auction observations per symbol and the native 60-calendar-day lookback. Independent checks show that uninterrupted streaming, a restart before August followed by streaming, and a scoring-boundary seed produce identical state and forecasts. It is not equivalent to the sparse legacy feed.'),p('The first OOS trade, 012450 on 3 April, now sees its IS auction observation from 10 March. Under literal legacy routing that is still only one eligible sample, below the three-observation minimum. Continuity therefore does not guarantee broad coverage. Prior-to-April samples also expire before the June tail losses under the 60-day forecast window; a June change cannot be attributed directly to those old samples. Cash-path effects can persist.')]
    parts+=[p('Literal continuity preserves a material modeling limitation: the legacy liquidity forecaster sees whichever symbols the shared replay feed delivers, including KALCB-related observations. It is not invariant to that feed universe. Restoring its state fixes the restart defect without removing this coupling. The selected literal-policy result therefore depends on reproducing that observation contract; feeding every live-market auction would instead implement the separately tested broad policy. The broader policy’s performance cost cannot be concealed by presenting it as an equivalent warm-start implementation.'),p('These are independently scored, flat-start accounts, not one uninterrupted funded portfolio. Each window retains its original terminal-liquidation convention, including the last two sessions without new entries. Session-local bar lists reset daily by design; position/order state is flat at the checkpoint, and the value-learning mode is off. The repaired cross-window state is the relevant persistent liquidity history. Signal-history reconstruction includes earlier dates independently of those account conventions.')]
    parts+=[p('For exact comparability, the literal policy retains the original empty liquidity state at the IS account’s August start; lagged signal features still use the 61 earlier sessions. It then carries the actual resulting IS liquidity state into OOS. The broad alternative changes both pre-IS liquidity initialization and ongoing observation coverage, so its performance delta is not attributed solely to August warm-up. Pre-August samples themselves are outside the native 60-day forecast window by April; the separately tracked routing and allocation paths can still differ.')]
    continuity=next(v for v in rows if v['label']=='literal_carry_original')
    sparse=sum(count for size,count in continuity['oos']['authentication']['entry_auction_sample_counts'].items() if int(size)<3)
    parts+=[p(f'Literal carry reduces the number of original-rule OOS trades with fewer than three eligible auction samples from 28 of 47 to {sparse} of 47. Broad observation supplies 20 samples for every one of its 186 IS and 47 OOS control trades. Under the legacy rule, an insufficient forecast removes that forecast-based request bound; the broker’s actual volume-participation limit still constrains every fill. These are different layers of liquidity protection, and neither should be confused with permission to invent unconstrained fills.')]
    parts+=[h('2. Native paired results'),p('Every row below is a completed native replay with artifact hashes checked, a passed source/execution audit, flat terminal state, and independent reconciliation of trade P&L, final NAV and every-bar maximum drawdown. Percentage-point gains compare against the original saved round 1, not a more favorable replacement benchmark.')]
    base=diagnostics['original'];tab=[['Original saved round 1',number(base['train']['result']['return_pct']),number(base['oos']['result']['return_pct']),number(base['train']['result']['drawdown_pct']),number(base['oos']['result']['drawdown_pct']),'176 / 47','Reference']]
    for v in rows:tab.append([label(v['label']),number(v['train']['return_pct']),number(v['oos']['return_pct']),number(v['train']['drawdown_pct']),number(v['oos']['drawdown_pct']),f'{v["train"]["trades"]} / {v["oos"]["trades"]}','Eligible nominal' if v['eligible'] else 'Control / failed gate'])
    parts+=[table(['Policy','IS return %','OOS return %','IS DD %','OOS DD %','Trades IS / OOS','Status'],tab)]
    bylabel={v['label']:v for v in rows};ablations=[]
    for left,right,meaning in (
        ('literal_carry_original','literal_carry_guard_1p0','Add 1.0R guard, same original scoring / actual-history carry'),
        ('literal_carry_guard_1p0','literal_carry_full_panel_guard_1p0','Activate corrected sector panel, same guard / carry'),
        ('literal_carry_full_panel_guard_1p0','broad_declared_universe_full_panel_guard_1p0','Broaden liquidity initialization and ongoing observation, same corrected panel / guard'),
        ('broad_declared_universe_full_panel','broad_declared_universe_full_panel_guard_1p0','Add guard under broad-history / corrected-panel policy')):
        if left in bylabel and right in bylabel:
            a,z=bylabel[left],bylabel[right];ablations.append([meaning,number(z['train']['return_pct']-a['train']['return_pct']),number(z['oos']['return_pct']-a['oos']['return_pct'])])
    parts+=[table(['Matched native ablation','IS return change (pp)','OOS return change (pp)'],ablations),p('Ablations are comparisons of complete native portfolio paths. Effects need not add across different controls because fills, capital reuse, candidate membership and subsequent position sizes interact.')]
    parts+=[p('The working definition of “without material IS deterioration” was fixed before this follow-up: at most a 3-percentage-point IS return reduction, positive OOS return with at least a 2-point improvement, and at most a 2-point drawdown increase in either period. Three points is an operational research tolerance, not a user-specified universal threshold or a statistical non-inferiority margin. Selection first prefers improvement in both returns, then the larger of the worst-window gains, then OOS return. Actual differences are shown so stricter preferences remain assessable.'),p('The corrected current-afternoon panel changes candidate membership or rank on only four IS dates and no OOS dates: 13 November, 8 January, 14 January and 20 February. This is a feature-validity repair rather than evidence that every new score is predictive. It uses information available at the original cutoff; a synthetic future session and a bar at/after 14:30 cannot change an earlier decision feature.')]
    parts+=[p('These failures reject the tested combinations, not the informational value of properly measured volume. The original scoring coefficients were fitted to the original feature definitions, and the original liquidity policy used a selected, sparse observation stream. Correcting their inputs without recalibrating those relationships is a model change. A future adoption study should estimate affected weights or sizing relationships within pre-April chronological training/validation folds and freeze the result before a new holdout; optimizing them directly against these April–July outcomes would compound the selection problem. This follow-up deliberately does not expand into that additional parameter search.')]
    parts+=[h('3. Why the exit change helps, and what it costs'),p('The tested guard uses completed bars. After at least one held bar, it requests an exit when close-to-entry return divided by the position’s native risk unit is at or below the chosen negative threshold, provided maximum favorable excursion remains below 4R. The original MFE-fade exit remains active. Execution occurs through the native subsequent eligible fill process with volume limits; this is not a promise of an intrabar fill at exactly −1R. Overnight gaps and partial fills can produce a larger realized loss. The risk unit comes from the existing native position metadata and its fallback floors; it is not a constant percentage stop.')]
    if analysis_row:
        comparison=accounting['comparisons']['original_to_'+analysis_row['label']]
        attr=[]
        for w in ('train','oos'):
            v=comparison[w];a=v['accounting_components'];unit=a['exit_price_component']+a['entry_price_component']+a['cost_component']
            attr.append([window(w),money(v['account_pnl_delta']),money(unit),money(a['quantity_component']),money(a['new_minus_missing']),f'{v["matched"]} / {v["new"]} / {v["missing"]}'])
        parts+=[table(['Window','Total Δ P&L','Observed per-unit price / cost','Quantity component','New less missing trades','Matched / new / missing'],attr)]
        matched_label=('literal_carry_full_panel' if analysis_row['mutations'].get(b.x.PANEL_KEY) else 'literal_carry_original') if analysis_row['history_policy']=='literal_carry' else 'broad_declared_universe_full_panel'
        matched_comparison=accounting['comparisons'].get(matched_label+'_to_'+analysis_row['label'],{}).get('oos')
        if matched_comparison:
            cohorts=matched_comparison['cohorts'];wins=cohorts['baseline_winners'];losses=cohorts['baseline_losers']
            parts+=[p(f'Against the same-history, same-scoring unguarded OOS control, {wins["exit_changed_count"]} of {wins["matched_count"]} matched baseline winners and {losses["exit_changed_count"]} of {losses["matched_count"]} matched baseline losers change their final exit timestamp or reason. Among the affected losers, {losses["exit_changed_improved_per_unit"]} improve and {losses["exit_changed_worsened_per_unit"]} worsen on the observed per-unit accounting. This is a targeted reduction of adverse outcomes, not a rule that improves every losing trade.')]
        v=comparison['oos'];tops=v['two_best_contributions']
        parts+=[p('For the leading candidate OOS replay, the two largest matched contributions total '+money(sum(q['pnl_delta'] for q in tops))+'. Removing those contributions arithmetically leaves '+money(v['net_delta_excluding_two_best_arithmetic_only'])+' of the total change. This concentration means the observed improvement is not broad evidence of uniformly better exits across market regimes.'),table(['Symbol / entry decision','Original P&L','Candidate P&L','Δ P&L'],[[q['symbol']+' / '+q['entry_decision_time'][:10],money(q['baseline_pnl']),money(q['candidate_pnl']),money(q['pnl_delta'])] for q in tops])]
        worst=sorted(v['rows'],key=lambda q:q['pnl_delta'])[:3]
        parts+=[p('The adverse examples matter equally. The largest matched OOS deteriorations are:'),table(['Symbol / entry decision','Original → candidate quantity','Original → candidate trade return %','Δ P&L'],[[q['symbol']+' / '+q['entry_decision_time'][:10],f'{q["baseline_qty"]} → {q["candidate_qty"]}',f'{q["baseline_return_pct"]:.2f} → {q["candidate_return_pct"]:.2f}',money(q['pnl_delta'])] for q in worst])]
    parts+=[p('The decomposition is an exact identity on observed fills: baseline quantity times the change in per-share outcome, plus quantity change times the candidate per-share outcome, plus entry/missing-trade effects. It is not an executable fixed-quantity causal simulation. Earlier exits free cash and can change later allocation, even when upward compounding is disabled. Identical final exit timestamps also do not establish identical partial-fill schedules.'),p('This matters particularly in IS: the original-history 1.0R guard loses about ₩3.666m on its observed per-unit price/cost component, gains about ₩3.537m from quantity changes, and loses about ₩0.383m from new versus missing matches, for a net reduction of about ₩0.511m. One 23 January 010130 position increases from 11 to 35 shares and contributes approximately ₩5.167m of the offset. Small aggregate IS deterioration is therefore not evidence that the exit rule is costless.')]
    parts+=[h('4. Execution stress and capital-path checks'),p('The combined stress reduces actual bar participation from 1% to 0.5% and raises OLR slippage from 5 to 15 basis points. Commission remains 2 basis points on each side and sell tax 20 basis points. These are scenario assumptions, not a statement of the universally applicable current tax schedule. Full native IS and OOS runs use matching scenario checkpoints. The same return and drawdown gates apply against the original stressed benchmark.')]
    parts+=[table(['Policy','Window','Return %','DD %','Trades','Role'],[['Original saved round 1' if a['comparison']=='original' else label(a['label']),window(a['window']),number(a['return_pct']),number(a['drawdown_pct']),a['trades'],a['comparison'].replace('_',' ')] for a in stress])]
    if decision.get('finalists'):parts+=[p('Stress gate verdicts: '+ '; '.join(html.escape(label(v['label']))+(': pass' if v.get('stress_eligible') else ': fail') for v in decision['finalists'])+'.')]
    prior=diagnostics['existing_1p2_stress_reference']
    parts+=[p(f'The 1.2R alternative remains relevant because it has less observed IS per-unit exit damage and lower nominal IS drawdown than 1.0R; it is not dominated on every risk measure. Its earlier, authenticated original-history combined-stress result was {prior["train"]["result"]["return_pct"]:.4f}% IS and {prior["oos"]["result"]["return_pct"]:.4f}% OOS. The IS reduction versus the original stressed 38.7214% is 3.1041 points, slightly beyond the registered 3-point limit. This is retained prior-source evidence, not relabeled as a new restart-corrected OOS result. The close margin should be understood as an operational gate boundary, not a sharp statistical distinction.')]
    parts+=[table(['No-upward-compounding policy','Window','Return %','DD %','Trades'],[[label(a['label']),window(a['window']),number(a['return_pct']),number(a['drawdown_pct']),a['trades']] for a in capped]),p('The no-upward-compounding runs preserve cash availability, native fills and capital reuse while limiting the upward expansion of the account budget. They are a diagnostic of dependence on capital growth, not an isolated exit experiment and not an additional searched objective. A material loss of the IS preservation property here is reported explicitly rather than hidden by the nominal gate.')]
    capital_diagnostic=None
    if chosen and capped:
        candidate_pair={a['window']:a for a in capped if a['candidate_id']==chosen}
        control_pair={a['window']:a for a in capped if a['candidate_id']!=chosen}
        assert set(candidate_pair)==set(control_pair)=={'train','oos'}
        capital_diagnostic={w:{'candidate_return_pct':candidate_pair[w]['return_pct'],'control_return_pct':control_pair[w]['return_pct'],
            'return_gain_pp':candidate_pair[w]['return_pct']-control_pair[w]['return_pct'],
            'drawdown_change_pp':candidate_pair[w]['drawdown_pct']-control_pair[w]['drawdown_pct']} for w in ('train','oos')}
        gain=capital_diagnostic['train']['return_gain_pp'];oosgain=capital_diagnostic['oos']['return_gain_pp']
        verdict=('The IS return difference remains within the same 3-point working tolerance. The observed nominal preservation therefore does not require upward budget compounding, although cash reuse and quantity-path dependence remain.' if gain>=-3 else
            'This exceeds the 3-point IS tolerance used for the nominal and execution-stress comparisons. Consequently the nominal IS preservation is conditional on the original capital-growth treatment; it is not robust to this capital diagnostic. The registered nominal/stress selection is retained as a research result, but this limitation materially weakens the case for adopting it under a different sizing policy.')
        parts+=[p(f'With upward budget growth capped, the guard changes IS return from {control_pair["train"]["return_pct"]:.4f}% to {candidate_pair["train"]["return_pct"]:.4f}% ({gain:+.4f} points), and OOS from {control_pair["oos"]["return_pct"]:.4f}% to {candidate_pair["oos"]["return_pct"]:.4f}% ({oosgain:+.4f} points). '+verdict)]
    n.write(OUT/'decision_summary.json',{'created_utc':s.utc(),'selected':chosen,'status':'research_only_not_live_promoted',
        'nominal_and_execution_stress_gates_passed':bool(chosen),'no_upward_compounding_diagnostic':capital_diagnostic,
        'independent_validation':False,'scoring_policy':'Original scoring retained; corrected sector panel disabled because its activation failed IS preservation.',
        'observation_policy':'Literal candidate-specific IS terminal market state carried to OOS; sparse legacy feed contract retained.',
        'limitations':['The inspected April-July period is development data.','OOS gain is concentrated in two adverse June trades.','IS preservation includes substantial offsetting quantity-path effects.','The selected policy has not established portability to a broad all-market observation feed.']})
    parts+=[h('5. Payoff, calendar and statistical limits')]
    if analysis_row:
        sel=diagnostics['candidates'][analysis_id]['windows']['baseline'];pay=[]
        for name,series in (('Original',base),('Leading candidate',sel)):
            for w in ('train','oos'):
                a=series[w]['payoff'];pay.append([name,window(w),number(100*a['win_rate']),number(a['profit_factor']),number(a['dollar_payoff_ratio']),money(a['mean_win_pnl']),money(-a['mean_loss_magnitude']),money(a['pnl_size_return_covariance'])])
        parts+=[table(['Policy','Window','Win rate %','Profit factor','Win / loss payoff','Average winner','Average loser','Size–return covariance P&L'],pay)]
        months=[]
        for w in ('train','oos'):
            for month,a in base[w]['calendar'].items():
                z=sel[w]['calendar'][month];months.append([window(w),month,number(a['return_pct']),number(z['return_pct']),money(z['pnl']-a['pnl']),a['sessions']])
        parts+=[table(['Window','Month','Original NAV return %','Candidate NAV return %','Δ marked P&L','Sessions'],months)]
        uncertainty=[]
        for w,refs in diagnostics['comparisons'].get(analysis_id,{}).items():
            for ref,a in refs.items():
                v=a['paired_block_bootstrap']['5'];ci=v['ci95_pct_or_gain_pp'];uncertainty.append([window(w),ref.replace('_',' '),f'[{ci[0]:.2f}, {ci[1]:.2f}]',number(100*v['resampled_fraction_positive'])])
        parts+=[table(['Window','Comparator','5-session block 95% interval for gain (pp)','Resampled gains above zero %'],uncertainty)]
    parts+=[p('The bootstrap uses 10,000 paired circular resamples of daily account returns, with fixed seed 20261010; 1-, 3-, 5- and 10-session blocks are retained in the diagnostics. It conditions on the observed series. It neither creates unseen regimes nor corrects for strategy selection, repeated trial inspection, tail-event selection or the original round’s search. A confidence interval spanning zero cannot support a claim of statistically established superiority. Forty-seven OOS trades are a small, dependent sample, and the periods differ in length.'),p('The fundamental discrepancy remains a payoff and sizing issue, not a falling hit rate: original hit rate rises from 48.30% to 51.06%, while mean losing P&L increases about 4.89-fold. Original size–return covariance shifts from approximately +₩26.07m in IS to −₩19.39m in OOS. Candidate supply also thins from 356 candidates across 154 entry snapshots to 74 across 67. Rank concentration and a 65% position ceiling amplify an adverse handful of trades. None of these descriptive identities proves that simply flattening rank weights will improve both periods; the prior native variants and this bounded follow-up determine the practical choice.')]
    matched=n.read(s.forensic.OUT/'matched_length_is_windows.json')['windows']
    parts+=[p(f'The earlier duration-matched diagnostic contains {len(matched)} overlapping 69-session IS NAV windows, with returns from {min(v["return_pct"] for v in matched):.2f}% to {max(v["return_pct"] for v in matched):.2f}%. These are slices of one realized account path, not independently reset strategy runs. They nevertheless show that the original OOS deterioration is not explained merely by comparing 69 OOS sessions with 156 IS sessions. Even a positive optimized OOS result does not establish recovery of the original IS risk/return regime.')]
    parts+=[h('6. Search discipline, discarded evidence and reproducibility'),p('The initial follow-up completed three pairs before the native-state defect was discovered: the 1.0R guard, a −10bp entry-confirmation threshold, and their 0.8R interaction. The confirmation alternatives damaged IS materially and were not carried into the final search. The replacement design separated literal continuity, corrected sector features and broad declared-universe observation. Only the predeclared 0.8R / 1.2R guard neighborhood was eligible for adaptive local refinement. At most two nominal finalists received combined stress; the selected policy and its continuous unguarded reference received the capital diagnostic. Reused and duplicate configurations are not treated as independent evidence.'),p('A predecessor broad-history IS run using every archived symbol failed native source qualification because extra out-of-universe files were included. Its raw 46.4076% return is excluded from all qualified rankings. The source was corrected to the declared 103-stock universe; the audits were not relaxed. The failed output and its disposition remain available. A separate pre-market manifest-location error was also preserved and corrected before any trading outcome from that attempt.'),p('The final qualified results bind exact frozen source generations, input manifests, candidate mutations, snapshot-builder hashes, scenario and warm-start receipts. The checkpoint branch was exercised directly with 12 malformed or incompatible input cases, all rejected, plus a valid retained-source checkpoint. These cover future observations, fabricated volumes, undeclared symbols, duplicate samples, phase errors, unqualified/non-flat parents, and configuration, scenario, compounding, capital or source-generation mismatches. The earlier history and causal panel checks verify prior-session coverage, default-behavior parity, and invariance to future/post-cutoff observations. The focused repository checks also passed; the source-bound validation receipts are retained with the report.')]
    parts+=[p(f'The executed follow-up roots contain {len(inventory)} completed native replay assessments: {sum(v["qualified"] for v in inventory)} qualified and {sum(not v["qualified"] for v in inventory)} excluded. The inventory includes control reruns and execution/capital scenarios; it is not a count of independent hypotheses. Passing a native execution audit establishes compliance with the declared simulation model, not a guarantee of identical live fills.')]
    parts+=[h('7. What to use next'),p('Use the selected research specification together with its frozen source generation, explicit panel/scoring mode, snapshot builder and history handoff. A bare exit-configuration change does not implement this result. The review bundle contains source diffs against exact authenticated parent hashes and the full selected mutation set. The canonical/live strategy is not automatically replaced by a research winner.'),p('The next acceptance evidence must come from an untouched, source-qualified later period or a precommitted forward shadow run. Freeze the chosen rule, universe, observation policy, fees and sizing treatment before observing that evidence. Track loss severity, early-exit rebound costs, win/loss notional asymmetry, capacity-history coverage, partial exits, realized slippage and rank-one concentration. Compare the frozen successor against the original and its continuous-history control on the same opportunities. Do not keep tuning April–July while describing it as an independent OOS test.'),p('The immediate implementation requirement is satisfied by continuity from IS into the first OOS session. Extending the scoring date backward is a different study and is not needed to provide warm-up history. No claim is made that moving the IS start or maximizing this already-inspected OOS period would solve generalization.')]
    parts+=[h('Evidence files'),'<ul>'+''.join('<li><a href="'+name+'">'+label+'</a></li>' for name,label in [
        ('selected_research_policy.json','Selected policy and bound source / history contract'),
        ('decision_summary.json','Selection verdict and capital-diagnostic qualification'),
        ('boundary_restart_paired_results.json','Authenticated paired comparisons'),
        ('boundary_restart_stress_results.json','Matched combined execution stress'),
        ('boundary_restart_no_upward_compounding.json','Native no-upward-compounding diagnostics'),
        ('boundary_restart_exit_and_quantity_attribution.json','Observed-fill exit and quantity accounting'),
        ('boundary_restart_diagnostics.json','Calendar, payoff and bootstrap diagnostics'),
        ('literal_restart_contract_checks.json','Checkpoint acceptance and rejection checks'),
        ('review_bundle/bundle_manifest.json','Review bundle and file hashes')] if (OUT/name).exists())+'</ul>',p('Methodological context: <a href="https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf">Bailey and López de Prado, The Deflated Sharpe Ratio</a>, and <a href="https://escholarship.org/uc/item/4w1110bb">The Probability of Backtest Overfitting</a>. These sources explain why selecting among historical alternatives requires a higher evidentiary standard; their statistics have not been misapplied as a significance score for this small return sample.')]
    export_bundle(chosen)
    body='\n'.join(parts)
    if not chosen:
        body=body.replace('Use the selected research specification together with its frozen source generation, explicit panel/scoring mode, snapshot builder and history handoff. A bare exit-configuration change does not implement this result. The review bundle contains source diffs against exact authenticated parent hashes and the full selected mutation set. The canonical/live strategy is not automatically replaced by a research winner.',
            'No replacement specification passed all qualification gates. Retain the verified history handoff and the complete experimental evidence; the nominal winner remains a research candidate with an unresolved stress trade-off. The review bundle provides exact source diffs and qualification decisions. The canonical/live strategy has not been replaced.')
    css='body{font:16px/1.65 system-ui,sans-serif;color:#172033;background:#f8fafc;margin:0}main{max-width:1160px;margin:auto;padding:48px 32px;background:white}h1{font-size:34px;line-height:1.2}h2{margin-top:42px;font-size:24px}p{max-width:1080px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px 12px;border-bottom:1px solid #e2e8f0;text-align:left;vertical-align:top}th{background:#edf2f7}tr:nth-child(even){background:#f8fafc}.table{overflow-x:auto;margin:24px 0}img{width:100%;height:auto}a{color:#0369a1}strong{color:#065f46}@media print{main{padding:0}h2{break-after:avoid}.table{overflow:visible}table{font-size:10px}}'
    (OUT/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>OLR continuous-history optimization</title><style>'+css+'</style><main>'+body+'</main></html>',encoding='utf-8')
    class Plain(HTMLParser):
        def __init__(self):super().__init__();self.text=[]
        def handle_starttag(self,tag,attrs):
            if tag in ('p','h1','h2','tr','li'):self.text.append('\n\n')
        def handle_endtag(self,tag):
            if tag in ('td','th'):self.text.append(' | ')
        def handle_data(self,data):self.text.append(data)
    plain=Plain();plain.feed(body);(OUT/'report.txt').write_text(''.join(plain.text).strip()+'\n',encoding='utf-8')
    n.write(OUT/'report_manifest.json',{'created_utc':s.utc(),'selected':chosen,'generator_sha256':n.sha(__file__),
        'qualification_complete':n.sha(b.ROOT/'qualification_completion.json'),
        'files':{name:n.sha(OUT/name) for name in ('report.html','report.txt','optimization_results.png','optimization_results.svg','decision_summary.json','review_bundle/bundle_manifest.json','native_run_inventory.json','boundary_restart_diagnostics.json','boundary_restart_exit_and_quantity_attribution.json')}})
    print({'report':str(OUT/'report.html'),'selected':chosen})


if __name__=='__main__':main()

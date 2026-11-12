"""Reconstruct the registered decision and publish only validated research evidence.

No candidate is selected, refitted, or rescued using validation outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.portfolio_targeted_validation import ROOT,read,bound,oos_checks,development_checks,choose_finalist
from scripts.portfolio_native_budget_study import checked,verified_spec
from scripts.portfolio_allocation_challenge import risk_metrics,BATTERY
from backtests.strategies.kalcb.executed_value import digest,write_json

VERSION='portfolio-targeted-validation-profile-v1'


def verify_binding(item):
    if digest(item['path'])!=item['sha256']:raise ValueError('Changed bound evidence: '+item['path'])
    return read(item['path'])


def reconstruct_development(root):
    """Verify the complete registered battery without opening holdout returns."""
    root=Path(root).resolve();seal=read(root/'development_selection.json')
    protocol=verify_binding(seal['protocol']);screen=verify_binding(seal['screen'])
    verify_binding(seal['bridge'])
    risk=protocol['development_risk'];qualified={};candidate_metrics={}
    for name,battery in seal['candidate_batteries'].items():
        if name not in screen['qualifiers'] or set(battery)!=set(BATTERY):
            raise ValueError('Unregistered or incomplete development battery')
        actual={}
        for label,old in battery.items():
            verify_binding(old['completion']);case=Path(old['completion']['path']).parent;checked(case,True)
            m=risk_metrics(case);m['risk_checks']=development_checks(m,label,risk)
            if m!=old:raise ValueError('Sealed development metrics changed')
            actual[label]=m
        candidate_metrics[name]=actual
        if all(all(m['risk_checks'].values()) for m in actual.values()):qualified[name]=actual
    if set(candidate_metrics)!=set(screen['qualifiers']):raise ValueError('Missing screened challenger')
    if choose_finalist(qualified)!=seal['finalist']:raise ValueError('Development finalist does not reconstruct')
    for label,item in seal['qualified_control_reuse'].items():
        verify_binding(item['case']);verify_binding(item['qualified'])
        a=Path(item['case']['path']).parent
        b=Path(item['qualified']['path']).parent
        checked(b,True)
        if any(digest(a/f)!=digest(b/f) for f in ('trades.json','daily.json','equity.json')):
            raise ValueError('Changed historical control bridge')
        if risk_metrics(b)!=seal['control_batteries']['control'][label]:raise ValueError('Changed control risk metrics')
    for old in seal['control_batteries']['incumbent'].values():
        verify_binding(old['completion']);case=Path(old['completion']['path']).parent
        checked(case)
        if risk_metrics(case)!=old:raise ValueError('Changed historical reference metrics')
    return seal,protocol


def source_blocked_decision(root,seal):
    """An invalid source is an untested holdout, never a failed return test."""
    basis=read(root/'provider_basis_audit/audit.json')
    review=read(root/'source_basis_review.json')
    if basis['source_gate_pass'] is not False or review['source_gate_pass'] is not False:
        raise ValueError('A source-blocked decision requires a completed failed source audit')
    if verify_binding(review['provider_audit'])!=basis:raise ValueError('Changed provider audit')
    for item in basis['original_and_adjusted_controls'].values():verify_binding(item)
    for item in review['dependencies'].values():
        if digest(item['path'])!=item['sha256']:raise ValueError('Changed source dependency')
    if (root/'validation').exists():raise ValueError('Do not describe an existing validation replay as unopened')
    pointer=read(root/'protocol.json')['references']['selected_pointer']
    if digest(pointer['path'])!=pointer['sha256']:raise ValueError('Canonical research pointer changed')
    return {'version':'portfolio-targeted-acceptance-v1','protocol':bound(root/'protocol.json'),
        'development_selection':bound(root/'development_selection.json'),
        'source_basis_review':bound(root/'source_basis_review.json'),
        'finalist':seal['finalist'],'acceptance':{'pass':None,'status':'not_evaluated_source_gate_failed','tests':{}},
        'promotion_eligible':False,'outcomes':{},'qualified_cases':{},
        'source_and_execution_pass':False,'independent_performance_established':False,
        'validation_window_now_consumed':False,'oos_used_for_acceptance_only':True,
        'independence_basis':'No portfolio OOS replay in this task. Retained records reserve June–July; incomplete original search logs prevent a categorical historical-independence claim.',
        'oos_reselection_performed':False,'live_deployment_enabled':False}


def assess(root):
    """Deterministic recomputation, including development and holdout gates."""
    root=Path(root).resolve();seal,protocol=reconstruct_development(root)
    if not (root/'validation').exists():return source_blocked_decision(root,seal)
    risk=protocol['development_risk']
    registration=read(root/'validation/registration.json')
    if verify_binding(registration['development_selection'])!=seal:raise ValueError('Unsealed validation choice')
    verify_binding(registration['protocol'])
    study=read(root/'validation/study.json');verified_spec(study)
    names=list(dict.fromkeys(('control','incumbent',seal['finalist'])))
    expected={(n,s) for n in names for s in ('baseline','combined')}
    if {(j['recipe'],j['scenario']) for j in registration['jobs']}!=expected or len(registration['jobs'])!=len(expected):
        raise ValueError('Validation expanded beyond the registered comparisons')
    if study['initial_equity']!=25_000_000 or study['candidates'][seal['finalist']]!=seal['recipe']:
        raise ValueError('Changed validation capital or finalist recipe')
    controls=read(root/'development/study.json')['candidates']
    if any(study['candidates'][n]!=controls[n] for n in names):raise ValueError('Validation recipe drift')
    if study['configuration']['date_range']!={'start':'2026-06-01','end':'2026-07-10'}:
        raise ValueError('Changed reserved validation window')
    outcomes={};cases={}
    for name,scenario in sorted(expected):
        case=root/'validation/qualified'/(name+'_'+scenario)
        result=checked(case,True)
        if result['sessions']!=29:raise ValueError('Incomplete reserved validation sessions')
        trades=read(case/'trades.json')
        if any(t['entry_fill_time'][:10]>='2026-07-09' for t in trades):
            raise ValueError('Entry in the registered liquidation-only tail')
        outcomes.setdefault(name,{})[scenario]=risk_metrics(case)
        cases.setdefault(name,{})[scenario]={
            'output':str(case),'recipe':name,'scenario':scenario,
            'evidence':{f:bound(case/f) for f in ('completion.json','request.json','result.json','execution_audit.json','allocation_audit.json')},
            'metrics':{k:v for k,v in outcomes[name][scenario].items() if k!='completion'}}
    acceptance=oos_checks(outcomes[seal['finalist']],outcomes['control'],risk)
    return {'version':'portfolio-targeted-acceptance-v1','protocol':bound(root/'protocol.json'),
        'development_selection':bound(root/'development_selection.json'),
        'validation_registration':bound(root/'validation/registration.json'),
        'finalist':seal['finalist'],'acceptance':acceptance,
        'promotion_eligible':acceptance['pass'] and seal['finalist']!='control',
        'outcomes':outcomes,'qualified_cases':cases,
        'source_and_execution_pass':True,'independent_performance_established':False,
        'validation_window_now_consumed':True,'oos_used_for_acceptance_only':True,
        'independence_basis':'Retained records reserve June–July; user expects no prior use. Some original search logs are missing.',
        'oos_reselection_performed':False,'live_deployment_enabled':False}


def make_profile(root):
    root=Path(root).resolve();decision=assess(root)
    if not decision['promotion_eligible']:raise ValueError('Registered validation does not permit research promotion')
    if read(root/'acceptance.json')!=decision:raise ValueError('Acceptance record does not reconstruct')
    study=read(root/'validation/study.json');seal=read(root/'development_selection.json')
    from scripts.portfolio_allocation_profile import verify_frozen_bytes
    cache={}
    for child in ('development','battery','validation'):
        verify_frozen_bytes(read(root/child/'study.json'),cache)
    return {'version':VERSION,'status':'research_selected_full_native_audits_and_reserved_validation_passed',
        'scope':'Short reserved validation, with incomplete historical independence records',
        'live_deployment_enabled':False,'independent_performance_established':False,
        'recipe':decision['finalist'],'settings':seal['recipe'],'initial_equity_krw':25_000_000,
        'evaluation_window':study['configuration']['date_range'],
        'source_generation_hash':study['source_generation_hash'],'study':bound(root/'validation/study.json'),
        'challenge_root':str(root),'protocol':bound(root/'protocol.json'),
        'selection':bound(root/'development_selection.json'),'acceptance':bound(root/'acceptance.json'),
        'verification_program':bound(Path(__file__)),
        'qualified_cases':decision['qualified_cases'][decision['finalist']],
        'development_battery':seal['candidate_batteries'][decision['finalist']],
        'risk_interpretation':'Historical full-battery gates and fixed holdout acceptance, not a guarantee or global optimum.',
        'operational_scope':'Frozen native research replay only; no live execution route or broker parity certification.'}


def verify_targeted_profile(profile):
    for key in ('protocol','selection','acceptance','study'):verify_binding(profile[key])
    if digest(profile['verification_program']['path'])!=profile['verification_program']['sha256']:
        raise ValueError('Changed profile verification program')
    if profile!=make_profile(profile['challenge_root']):raise ValueError('Profile differs from reconstructed qualification')
    return profile


def publish(root):
    decision=assess(root);path=root/'acceptance.json'
    if path.exists():
        if read(path)!=decision:raise ValueError('Prior acceptance changed')
    else:write_json(path,decision)
    if not decision['promotion_eligible']:
        if decision['acceptance'].get('status')=='not_evaluated_source_gate_failed':
            seal=read(root/'development_selection.json')
            provisional={'version':'portfolio-development-finalist-v1',
                'status':'development_qualified_awaiting_source_qualified_oos',
                'recipe':seal['finalist'],'settings':seal['recipe'],'initial_equity_krw':25_000_000,
                'development_selection':bound(root/'development_selection.json'),
                'acceptance':bound(path),'live_deployment_enabled':False,
                'canonical_selected_profile_changed':False}
            destination=root/'development_finalist.json'
            if destination.exists() and read(destination)!=provisional:raise ValueError('Preserve existing development finalist')
            if not destination.exists():write_json(destination,provisional)
            repository_copy=REPO/'config/research/portfolio_targeted_validation_20261009/development_finalist.json'
            repository_copy.parent.mkdir(parents=True,exist_ok=True)
            if repository_copy.exists() and read(repository_copy)!=provisional:raise ValueError('Preserve repository development finalist')
            if not repository_copy.exists():write_json(repository_copy,provisional)
        print(json.dumps({'promotion':False,'finalist':decision['finalist'],'acceptance':decision['acceptance']}),flush=True)
        return
    profile=make_profile(root)
    destination=REPO/'config/research/portfolio_targeted_validation_20261009/selected_profile.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and read(destination)!=profile:raise ValueError('Preserve existing research profile')
    if not destination.exists():write_json(destination,profile)
    verify_targeted_profile(read(destination))
    pointer=Path(read(root/'protocol.json')['references']['selected_pointer']['path'])
    replacement={'version':'portfolio-research-pointer-v1','profile':bound(destination),'live_deployment_enabled':False}
    if read(pointer)!=replacement:
        original=read(root/'protocol.json')['references']['selected_pointer']
        if digest(pointer)!=original['sha256']:raise ValueError('Selected pointer changed during research; preserve it')
        write_json(root/'previous_selected_pointer.json',read(pointer))
        write_json(pointer,replacement)
    print(json.dumps({'promotion':True,'profile':str(destination),'live_deployment_enabled':False}),flush=True)


def paired_uncertainty(candidate,control):
    """Descriptive paired moving-block resampling, never a selection input."""
    import numpy as np
    a=read(candidate/'daily.json');b=read(control/'daily.json')
    if [d['day'] for d in a]!=[d['day'] for d in b]:raise ValueError('Unpaired sessions')
    def returns(rows):
        nav=np.array([25_000_000]+[r['equity'] for r in rows]);return nav[1:]/nav[:-1]-1
    x,y=returns(a),returns(b);output={}
    for block in (5,10):
        rng=np.random.default_rng(20261009+block);samples=[]
        for _ in range(5000):
            starts=rng.integers(0,len(x)-block+1,size=int(np.ceil(len(x)/block)))
            ix=np.concatenate([np.arange(s,s+block) for s in starts])[:len(x)]
            samples.append(float(100*(np.prod(1+x[ix])-np.prod(1+y[ix]))))
        output[str(block)]={'resamples':5000,'return_difference_quantiles_pp':dict(zip(('p025','p50','p975'),np.quantile(samples,[.025,.5,.975]).tolist())),
                            'fraction_positive':float(np.mean(np.array(samples)>0))}
    return {'blocks_in_sessions':output,'scope':'Conditional resampling of paired realized daily returns; not a fresh policy replay, multiple-testing correction, or probability of future outperformance.'}


def mechanism_evidence(seal):
    import gzip
    batteries=seal['candidate_batteries'];comparisons={};release_events={}
    if {'joint','k_marginal'}<=set(batteries):
        for scenario in BATTERY:
            a=Path(batteries['joint'][scenario]['completion']['path']).parent
            b=Path(batteries['k_marginal'][scenario]['completion']['path']).parent
            comparisons[scenario]={f:digest(a/f)==digest(b/f) for f in ('trades.json','daily.json','equity.json','fills.json')}
            with gzip.open(a/'native_decisions.jsonl.gz','rt',encoding='utf-8') as stream:
                release_events[scenario]=sum('stale_inventory_release' in line for line in stream)
    return {'joint_vs_k_marginal_economic_identity':comparisons,
            'joint_native_decision_lines_with_stale_release':release_events,
            'scope':'Exact observed paths and native decision audit; inactivity does not establish effectiveness in unobserved regimes.'}


def report(root):
    from scripts.analyze_portfolio_risk_return_tradeoff import load_case,matched_decomposition
    from scripts.portfolio_opportunity_audit import export_case
    from xml.etree import ElementTree
    decision=assess(root);seal=read(root/'development_selection.json');screen=read(root/'screen.json')
    finalist=decision['finalist'];comparisons={};costs={};capital_audits={}
    windows=('development','validation') if decision['outcomes'] else ('development',)
    for window in windows:
        folder='cases' if window=='development' else 'qualified'
        for scenario in ('baseline','combined'):
            a=root/window/folder/(finalist+'_'+scenario);b=root/window/folder/('control_'+scenario)
            comparisons[window+'_'+scenario]={'attribution':matched_decomposition(load_case(str(a)),load_case(str(b))),
                'uncertainty':paired_uncertainty(a,b)}
            for n,p in ((finalist,a),('control',b)):
                key=window+'_'+n+'_'+scenario
                destination=root/'opportunity_audit'/key
                if (destination/'manifest.json').exists():
                    receipt=read(destination/'manifest.json');verify_binding(receipt['case'])
                    if receipt['case']!=bound(p/'completion.json'):raise ValueError('Changed capital audit source')
                    for item in receipt['tables'].values():
                        if digest(item['path'])!=item['sha256']:raise ValueError('Changed capital audit table')
                else:export_case(p,destination,reserved_validation=window=='validation')
                capital_audits[key]=bound(destination/'manifest.json')
                fills=read(p/'fills.json');result=read(p/'result.json')
                costs[window+'_'+n+'_'+scenario]={
                    'explicit_fees_and_tax_krw':sum(f['metadata']['fee'] for f in fills),
                    'buy_and_sell_turnover_krw':sum(f['qty']*f['price'] for f in fills),
                    'strategy_counts':result['strategy_counts'],'active_days':result['active_days'],
                    'trades_per_21_sessions':result['trades_per_21_sessions'],
                    'native_budget_blocked_orders':result['native_budget_blocked_orders'],
                    'native_budget_resized_orders':result['native_budget_resized_orders']}
    evidence={'decision':decision,'development_selection':seal,'screen':screen,
        'mechanism_evidence':mechanism_evidence(seal),
        'inventory_boundary_diagnostic':read(root/'inventory_boundary_diagnostic.json'),
        'comparisons':comparisons,'activity_and_costs':costs,'capital_opportunity_audits':capital_audits,
        'source_evidence':{n:bound(root/n) for n in ('source_audit.json','split_audit_verified.json','source_residuals/resolution.json',
            'frozen_inference_bridge.json','olr_inference_bridge.json','forward_feature_bridge/comparison.json','forward_prediction_completion.json')},
        'canonical_pointer_at_report':bound(REPO/'config/research/portfolio_synergy_selected.json')}
    evidence['source_evidence']['inventory_boundary_diagnostic.json']=bound(root/'inventory_boundary_diagnostic.json')
    evidence['source_evidence']['development_frozen_bytes_verified.json']=bound(root/'development_frozen_bytes_verified.json')
    testcases={};invocations={}
    for name in ('software_tests.xml','source_gate_tests.xml'):
        tree=ElementTree.parse(root/name)
        invocations[name]=bound(root/name)
        for case in tree.findall('.//testcase'):testcases[(case.attrib['classname'],case.attrib['name'])]=case
    totals={'tests':len(testcases),**{label:sum(case.find(tag) is not None for case in testcases.values())
            for label,tag in (('failures','failure'),('errors','error'),('skipped','skipped'))}}
    evidence['software_verification']={'junit_invocations':invocations,'focused_tests':totals,
        'broader_suite_issue':'Earlier broader run: test_research_generation.py::test_sourced_exceptional_session_windows expects no 2026-10-03 coverage; retained calendar now has coverage. Not changed or counted as passing.'}
    if not decision['outcomes']:
        evidence['source_basis_review']=read(root/'source_basis_review.json')
        evidence['source_evidence']['source_basis_review.json']=bound(root/'source_basis_review.json')
    output=REPO/'docs/research/portfolio_targeted_validation_20261009.json';write_json(output,evidence)
    write_report_text(root,evidence,output.with_suffix('.txt'))
    plot_report(root,evidence,output.with_suffix('.png'))
    print(json.dumps({'report':str(output.with_suffix('.txt')),'finalist':finalist,'acceptance':decision['acceptance']['pass']}),flush=True)


def plot_report(root,e,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import date
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    finalist=e['decision']['finalist'];selection=e['development_selection']
    palettes={'control':'#596875','incumbent':'#C37924',finalist:'#147D92'}
    labels={'control':'Current control','incumbent':'Higher-return reference',finalist:'Sealed finalist: '+finalist}
    batteries={**selection['control_batteries'],**selection['candidate_batteries']}
    names=list(dict.fromkeys(('control','incumbent',finalist)))
    scenario_labels=['Base','Cost','Liquidity','Combined','Auction\nprice','Price +\ncombined','Auction\nnonfill','Nonfill +\ncombined']
    x=np.arange(8)
    for n in names:
        b=batteries[n]
        axes[0,0].plot(x,[b[s]['return_pct'] for s in BATTERY],marker='o',color=palettes[n],label=labels[n])
        axes[0,1].plot(x,[b[s]['drawdown_pct'] for s in BATTERY],marker='o',color=palettes[n],label=labels[n])
    axes[0,0].set(title='Development returns across the fixed stress battery',ylabel='Net return (%)')
    axes[0,1].plot(x,[11]+[12.5]*7,ls='--',color='#A93737',label='Registered drawdown limit')
    axes[0,1].set(title='Development drawdown and risk limits',ylabel='Maximum marked drawdown (%)')
    for ax in axes[0]:
        ax.set_xticks(x,scenario_labels,fontsize=8);ax.grid(axis='y',alpha=.2)
        ax.legend(fontsize=8,loc='best')
    if not e['decision']['outcomes']:
        for n in names:
            axes[1,0].plot(x,[batteries[n][s]['trades'] for s in BATTERY],marker='o',color=palettes[n],label=labels[n])
        axes[1,0].axhline(515,ls='--',color='#A93737',label='Registered activity floor')
        axes[1,0].set_xticks(x,scenario_labels,fontsize=8)
        axes[1,0].set(title='Development activity across the same scenarios',ylabel='Economic entries')
        axes[1,0].grid(axis='y',alpha=.2);axes[1,0].legend(fontsize=8)
        axes[1,1].axis('off')
        axes[1,1].text(.03,.96,'Reserved OOS: not evaluated',transform=axes[1,1].transAxes,va='top',fontsize=15,weight='bold',color='#A93737')
        axes[1,1].text(.03,.83,
            'June 1–July 10, 2026 | 29 exchange sessions\n\n'
            'PASS  Chronology, frozen models, target-free features\n'
            'FAIL    Consistent price/share basis at the boundary\n\n'
            '068270 daily close, May 29 → June 1:\n'
            'Archive:    192,900 → 182,722  (mixed basis)\n'
            'Original:   192,900 → 191,700  (provider control)\n\n'
            'Development finalist saved; canonical selection retained.\n'
            'Repair and requalify inputs before the fixed OOS test.\n'
            'No OOS winner search, no live orders.',
            transform=axes[1,1].transAxes,va='top',fontsize=11,linespacing=1.65)
        fig.suptitle('Registered portfolio challenge | KRW25m | Development finalist only\n'
                     '156 development sessions. OOS source gate failed before performance replay.',fontsize=14)
        fig.savefig(path,dpi=170);plt.close(fig);return
    width=.23;x=np.array([0,1])
    for i,n in enumerate(names):
        rows=e['decision']['outcomes'][n]
        values=[rows[s]['return_pct'] for s in ('baseline','combined')]
        bars=axes[1,0].bar(x+(i-(len(names)-1)/2)*width,values,width,color=palettes[n],label=labels[n])
        for rect,scenario in zip(bars,('baseline','combined')):
            m=rows[scenario];h=rect.get_height()
            axes[1,0].annotate(f"{h:.2f}%\n{m['trades']} entries\nDD {m['drawdown_pct']:.2f}%",
                (rect.get_x()+width/2,h),xytext=(0,4 if h>=0 else -4),textcoords='offset points',
                ha='center',va='bottom' if h>=0 else 'top',fontsize=8)
        daily=read(root/'validation/qualified'/(n+'_combined')/'daily.json')
        axes[1,1].plot([date.fromisoformat(r['day']) for r in daily],
                       [100*(r['equity']/25_000_000-1) for r in daily],color=palettes[n],label=labels[n])
    axes[1,0].set_xticks(x,['OOS baseline','OOS combined stress'])
    axes[1,0].set(title='Reserved validation: return, entries and drawdown',ylabel='Net return (%)')
    axes[1,0].axhline(0,color='#555555',lw=.8);axes[1,0].margins(y=.3);axes[1,0].grid(axis='y',alpha=.2)
    axes[1,1].set(title='Reserved validation: combined-stress daily NAV',ylabel='Cumulative net return (%)')
    axes[1,1].tick_params(axis='x',rotation=25);axes[1,1].grid(axis='y',alpha=.2);axes[1,1].legend(fontsize=8)
    accepted=e['decision']['acceptance']['pass']
    fig.suptitle(f"Registered portfolio challenge | KRW25m | OOS acceptance: {'PASS' if accepted else 'FAIL'}\n"
                 '156 development sessions; 29 reserved validation sessions. Research only; historical independence records incomplete.',fontsize=14)
    fig.savefig(path,dpi=170);plt.close(fig)


def write_report_text(root,e,path):
    d=e['decision'];s=e['development_selection'];finalist=d['finalist']
    blocked=not d['outcomes']
    lines=['TARGETED PORTFOLIO VALIDATION — 9 OCTOBER 2026','',
        'Account: KRW 25,000,000. Research replay only. No live orders.',
        f"Development-selected finalist: {finalist}. Reserved validation: {'NOT EVALUATED — SOURCE GATE FAILED' if blocked else d['acceptance']['pass']}. Research promotion eligible: {d['promotion_eligible']}.",
        'This is a bounded, registered comparison, not proof of maximum possible synergy or future expected returns. The full machine-readable evidence accompanies this file.','',
        '1. What was implemented',
        'The main intervention compares an existing opening KALCB action with its next smaller native quantity tier (1 to .5, or .5 to .25). It keeps the signal and its native ranking. The smaller action must have positive forecasts in baseline and combined stress, failure probability below .001, and both quantity cohorts must contain at least 20 mature issuer-days across eight sessions. The extra size is removed only if its predicted increment is nonpositive in at least one scenario. Unsupported estimates leave the native action unchanged; timed additions are unchanged.',
        'Forecast lineage and raw-score support are matched to the original admission. The missing session counts in the old calibration are reconstructed from its exact frozen inputs and checked against the saved offsets and counts. No current-fold or holdout outcomes are used to change sizing. These forecasts were trained at their original 100m reference account: they are a proxy for quantity preference, not a calibrated KRW25m shared-account marginal value model.',
        'A separate OLR experiment attempts to sell half of weak residual inventory only after its intended next-session holding horizon has elapsed, after three complete opening bars, and only below completed VWAP. Native managed exits retain priority. A pending exit cannot be replaced; an actual sale must fill before buying power is released. This is a bounded risk heuristic, not an estimated continuation-value model.',
        'The joint recipe combines the two interventions. Existing native-rank and proportional allocation are mechanical comparators. No parameter grid was expanded after seeing results.','',
        '2. IS/OOS separation and what the validation establishes',
        'Development: 2025-08-07 through 2026-03-31, 156 sessions. Existing model preparation begins earlier, in May 2025; calibration and model fitting are chronological within development. The last model fit, dated 2026-03-10, is held fixed for the new window.',
        '2026-04-01 through 2026-05-12 is consumed validation and is not described as fresh OOS. Reserved validation is 2026-06-01 through 2026-07-10: 29 exchange sessions, of which the last two are liquidation-only, leaving 27 entry sessions. The account starts at 25m in both periods; these are separate paths, not one concatenated return.',
        'The selected KALCB/OLR records and retained boundary audits reserve June–July. The user expects this window not to have been used. Original OLR search logs are incomplete, so the available evidence cannot prove absolute historical independence. Accordingly independent_performance_established remains false even if this validation passes.',
        'Seven recipes were screened on development baseline/combined only. Screen-qualified challengers received the eight-scenario battery. The finalist was sealed before opening any portfolio validation results. Only that finalist and two fixed controls are permitted in OOS; neither a runner-up swap nor OOS threshold tuning is allowed.',
        'The original models reproduce all opening and timed selections in the final 16 saved development sessions. Target-free feature construction reproduces 4,005 saved feature inputs; OLR reconstruction reproduces two saved development sessions. Forward preparation uses prior daily data and completed intraday prefixes. No June–July outcome labels are fitted.',
        'The preliminary 103-symbol review found agreement between 2,987 OOS archived daily and intraday closes. Nine sparse symbol-sessions were examined against fresh read-only provider receipts. Six conserve daily OHLCV exactly; three satisfy the existing exact daily-close turnover residual proof. Missing bars are not invented and residual daily volume is not assigned to bars. Agreement within an archive does not establish a consistent price/share basis across its boundaries. The subsequent warm-up and provider-basis audit controls the source gate.','',
        '3. Registered risk and activity rules',
        'Physical gross cap 2 times marked NAV; name cap 1 times NAV; sector cap 2 times NAV. KALCB native sizing fraction .75, OLR .625; KALCB planned-stop risk cap 2% NAV. These are sizing controls, not realized-loss guarantees.',
        'Development: baseline drawdown <=11%; every adverse drawdown <=12.5%; maximum overnight exposure <=80%; mean overnight exposure <=30%; worst daily loss <=4.5%; at least 515 economic entries in every scenario. All gates must pass. Among feasible screen qualifiers, maximize the minimum adverse return; candidates within 1pp are ordered by minimum activity and then lower worst drawdown.',
        'OOS: positive net return in both baseline and combined; no lower return than the matched current control; at least 98% of its entries in each scenario; drawdown no more than .25pp above the control; the same absolute drawdown/exposure/worst-day limits. The 515-entry development floor is deliberately not applied to a 29-session window.',
        'Baseline participation uses at most 1% of prior completed interval volume, with KALCB/OLR slippage 3/5bp. Cost stress uses 13/15bp; participation stress halves the cap to .5%; combined applies both. Additional auction tests use 15bp adverse pricing or 50% nonfill. Explicit fees are 2bp per side and sell tax 20bp. Financing charges are not modeled.','',
        '4. Development screen',
        'Recipe | baseline return / DD / entries | combined return / DD / entries | full battery eligible']
    fmt=lambda m:f"{m['return_pct']:.4f}% / {m['drawdown_pct']:.4f}% / {m['trades']}"
    for n,b in e['screen']['results'].items():lines.append(f"{n} | {fmt(b['baseline'])} | {fmt(b['combined'])} | {n in e['screen']['qualifiers']}")
    lines += ['',
        'KALCB sizing changed actual native requests on 24 issuer-days across 19 sessions in both screen scenarios. OLR stale release did not activate in these two scenarios; the joint screen result therefore matches KALCB sizing alone. The auction-nonfill battery is needed before concluding the OLR addition is redundant.',
        'Native ranking improved screen returns, but even its conservative upper bound of contested cohorts spans only seven baseline and six combined sessions. This is below the registered eight-session floor and is not an exact causal activation count. Proportional allocation increased activity but reduced both returns. Neither passes the registered challenge.','',
        '5. Full development stress battery',
        'Recipe | scenario | return / drawdown / entries | mean overnight / maximum overnight | failed gates']
    for n,b in {**s['control_batteries'],**s['candidate_batteries']}.items():
        for label,m in b.items():
            checks=development_checks(m,label,read(root/'protocol.json')['development_risk'])
            lines.append(f"{n} | {label} | {fmt(m)} | {100*m['overnight_mean']:.3f}% / {100*m['overnight_max']:.3f}% | {', '.join(k for k,v in checks.items() if not v) or 'none'}")
    if finalist in s['candidate_batteries']:
        b=s['candidate_batteries'][finalist];control=s['control_batteries']['control'];inc=s['control_batteries']['incumbent']
        gains=[b[x]['return_pct']-control[x]['return_pct'] for x in BATTERY]
        lines += ['',f"The finalist improves return in every fixed development scenario by {min(gains):.4f} to {max(gains):.4f}pp. "
            f"Minimum scenario activity is {min(m['trades'] for m in b.values())} versus {min(m['trades'] for m in control.values())} entries. "
            f"Worst drawdown is {max(m['drawdown_pct'] for m in b.values()):.4f}%; largest mean overnight exposure {100*max(m['overnight_mean'] for m in b.values()):.4f}%; "
            f"largest single-night exposure {100*max(m['overnight_max'] for m in b.values()):.4f}%; worst daily return {min(m['worst_session_pct'] for m in b.values()):.4f}%.",
            'Activity is not uniformly higher: '+', '.join(f"{x} {b[x]['trades']-control[x]['trades']:+d}" for x in BATTERY)+'.',
            f"Against the higher-return .75/.75 reference, baseline return remains {inc['baseline']['return_pct']-b['baseline']['return_pct']:.4f}pp lower. "
            'The improvement recovers some return lost by the earlier risk reduction; it does not erase the return/risk tradeoff or establish a global optimum.']
    identity=e['mechanism_evidence']['joint_vs_k_marginal_economic_identity']
    if identity and all(all(row.values()) for row in identity.values()):
        lines += ['All eight joint cases have byte-identical trades, daily equity, intraday equity and fills to KALCB sizing alone. '
            f"Native decision lines mentioning stale release: {sum(e['mechanism_evidence']['joint_native_decision_lines_with_stale_release'].values())}. "
            'The OLR addition has no demonstrated incremental effect in this battery; the exact tie resolves to k_marginal under the existing deterministic selection rule.']
    inventory=e['inventory_boundary_diagnostic']['cases']['k_marginal_nonfill_combined']['counts']
    lines += [f"OLR mechanism diagnosis under auction-nonfill combined stress: {inventory['late_horizon_trades']} trades reached the second post-entry session or later; "
        f"{inventory['fully_exited_before_stale_eligibility']} fully exited before the new rule's earliest activation. "
        f"{inventory['native_fallback_submitted_before_eligibility']} already had a native market fallback submitted before that time. "
        'The new partial-exit rule cannot improve a fully committed market liquidation by replacing it with another order subject to the same participation/availability constraints. '
        'This targets a nonexistent discretionary inventory state in the observed sample. Any attempt to reduce the residual overnight risk must act before the failed auction, or forecast and size for its liquidity risk; releasing inventory after the gap does not undo the gap loss. '
        'A broad early-exit rule could remove profitable exposure, so it is not enabled or retuned after this result.']
    lines += ['',
        'The current control reuses all eight fully qualified cases after byte-identical trade/daily/equity bridges. The .75/.75 higher-return incumbent is a historical execution/account-checked reference; it is not eligible to bypass exposure or stress-drawdown gates. Qualifying challenger baseline/combined paths must match their screens byte for byte under full audit.','',
        '6. Reserved validation status',
        'Recipe | baseline return / DD / entries | combined return / DD / entries']
    for n,b in d['outcomes'].items():lines.append(f"{n} | {fmt(b['baseline'])} | {fmt(b['combined'])}")
    lines.append('Registered acceptance: '+json.dumps(d['acceptance'],sort_keys=True))
    if blocked:
        review=e['source_basis_review']
        lines += [
            'No OOS strategy return, drawdown or entry count was evaluated. This is a source qualification failure, not a failed performance test. The pre-existing canonical selection remains unchanged; the development finalist is saved separately. Source and feature inspection occurred, but this task did not consume the window for performance-based selection. Historical independence remains limited by missing older logs.',
            'The all-universe audit queried original and adjusted daily OHLCV for 103 symbols from May 29 through July 10. It found '+str(review['non_original_sessions'])+' archived symbol-sessions differing from current original-price responses: two adjusted-price/share-basis sessions for 068270 (June 1–2), and 24 July 10 volume-only revisions in other symbols. The volume revisions are separate from the decisive mixed-price issue; no strategy PNL effect is inferred from them. Their field-level classification and immutable raw receipts are in source_basis_review.json and provider_basis_audit/audit.json.',
            'The decisive discrepancy is 068270: archived May 29 daily close 192,900 is original-price basis; archived June 1 close 182,722 matches adjusted basis, while the original-price control is 191,700. The archive implies -5.2763% rather than -0.6221%, an artificial difference of -4.6542 percentage points. The May 29 intraday file itself mixes original opening prices and adjusted closing prices. By July 10 original and adjusted daily prices coincide, so a single future adjustment factor applied across the entire OOS window is not a justified repair.',
            'A retained older May 29 minute archive matches original daily OHLC but not total daily volume; fresh minute receipts match the adjusted basis. All 382 timestamps pair and constrain the adjustment factor, and an inverse-price feasibility probe is saved. These are diagnostic controls, not sufficient proof of native share quantities, missing-print placement, point-in-time lookbacks, or corporate-action handling of positions. The provider minute endpoint documents no original/adjusted switch. The earlier OLR forward source delivery also explicitly left this May 29 discrepancy unresolved.',
            'The prepared OOS features and predictions are quarantined as unqualified research artifacts. Their chronology checks still pass; their economic comparability does not. Freeze-replay now fails before creating a validation directory when the source gate is false.',
            'Required repair: obtain one documented, dated price/share convention covering the complete lookback and validation window; prove daily, minute and investor-flow units agree; preserve the actual action dates and account share/cash adjustments for overnight holdings. Rebuild a separate immutable input generation, repeat source conservation and original-development feature bridges, then rerun the same frozen March models. Keep the sealed recipe, June 1 start, liquidation tail, two controls and all acceptance thresholds unchanged. Do not drop the affected issuer, move the start date, invent missing volume or tune on subsequent OOS results.',
            'A qualified source repair can unlock the already implemented fixed six-case acceptance workflow. Until then, no candidate can honestly be called OOS-validated or promoted by this experiment.']
    else:
        lines += ['A failed gate retains the pre-existing selected research portfolio. It does not validate that control for deployment; it only prevents promoting an unsupported challenger. OOS is consumed by this evaluation and cannot be reused as an untouched selection sample.']
    lines += ['', '7. Return attribution, activity, and uncertainty']
    for label,c in e['comparisons'].items():
        lines.append(label+':')
        for sid in ('KALCB','OLR'):
            a=c['attribution'][sid]
            lines.append(f"  {sid}: net PNL change KRW {a['pnl_change']:,.0f}; quantity component {a['quantity_component']:,.0f}; per-share execution/exit/cost component {a['per_share_pnl_component']:,.0f}; new/missing entry component {a['entry_set_component']:,.0f}; new {len(a['new'])}, missing {len(a['missing'])}.")
        for block,u in c['uncertainty']['blocks_in_sessions'].items():
            q=u['return_difference_quantiles_pp'];lines.append(f"  {block}-session paired blocks, 5,000 resamples: difference quantiles {q['p025']:.3f}, {q['p50']:.3f}, {q['p975']:.3f}pp; fraction positive {u['fraction_positive']:.3f}.")
    lines += ['The decomposition is an exact accounting identity: delta(q*u)=delta(q)*mean(u)+delta(u)*mean(q). Quantity includes compounding and cap feedback; per-share PNL includes timing, exit mix and fees. These are not isolated causal effects. New/missing trade outcomes are hindsight diagnostics, not inputs to allocation.',
        'In the screen comparison, nearly all the baseline improvement comes from KALCB: approximately KRW681,761 versus KRW2,272 from OLR. Under combined stress, KALCB adds approximately KRW798,323 and OLR KRW78,282. This is primarily a better KALCB quantity path within the portfolio, with smaller cross-sleeve effects. It must not be presented as proof of maximum cross-strategy synergy.',
        'The baseline 2.5th–97.5th percentile differences cross zero in both block lengths; the combined-stress differences remain positive in those descriptive resamples. Thus the stronger evidence is improved behavior under the declared frictions, while baseline outperformance remains sensitive to the realized sequence. No threshold, recipe or selection decision was changed using these resamples.',
        'Block resampling is conditional on these realized paths and omits strategy re-execution, model selection and regime changes. It cannot turn a short holdout into statistically established expected outperformance. Development intervals also omit selection bias.','']
    for label,m in e['activity_and_costs'].items():
        lines.append(f"{label}: economic entries by strategy {m['strategy_counts']}; active days {m['active_days']}; entries/21 sessions {m['trades_per_21_sessions']:.2f}; explicit fees/tax KRW {m['explicit_fees_and_tax_krw']:,.0f}; two-sided turnover KRW {m['buy_and_sell_turnover_krw']:,.0f}; native-budget blocked/resized orders {m['native_budget_blocked_orders']}/{m['native_budget_resized_orders']}.")
    lines += ['',
        '8. Interpretation and next research boundary',
        'The prior return sacrifice arose because broad OLR budget reductions removed profitable exposure, while much of the worst stressed drawdown came from KALCB. Simply raising KALCB risk breached adverse drawdown limits; reducing OLR alone did not target that loss driver. The targeted experiment attacks quantity increments with weak forecast support instead of assuming every unit of a sleeve has equal value.',
        'There are too few genuinely contested simultaneous cohorts to justify a rich learned cross-strategy ranker here. Native scores are not calibrated on a common expected net KRW scale, and later OLR demand is not known when morning KALCB trades are sized. A realized winner/loser ranking would introduce hindsight. This implementation therefore cannot guarantee that every blocked trade is worse than every admitted trade.',
        'A reusable capital-opportunity audit has been exported for the finalist and control in the evaluated windows/scenarios. Its decision table contains native requested/offered/approved quantities, availability and admission timestamps, ranks, funding constraints, marked equity, budgets, same-event reservations and remaining funding. A separate realized-trade table gives actual filled lifecycle outcomes and label maturity. Rejected orders receive no fabricated zero or counterfactual PNL. Upstream native rejections are outside this table, and admission/partial-fill events are not counted as economic trades.',
        'The next evidence target is prospective decision-time shadow logging with the same schema plus common forecast lineage and actual broker fills. Any future common-value model should estimate net marginal value per scarce resource at 25m, use purged chronological training and mature labels, and be evaluated on a new untouched window. It should compete against these fixed native controls and retain activity as a constraint rather than maximize turnover mechanically.',
        'Do not retune on this reserved window after failure. If validation passes, the result supports only the registered research promotion. A larger untouched period, realistic financing and broker paper execution are still needed before a live-readiness claim. No live configuration was enabled.',
        '', '9. Verification and reproducibility',
        'Focused software suites: '+json.dumps(e['software_verification']['focused_tests'],sort_keys=True)+'. These are unique test identities across the content-bound JUnit receipts; repeated tests are not double-counted. Changed Python modules also compiled successfully.',
        e['software_verification']['broader_suite_issue'],
        'Both development generations were rehashed against their frozen manifests (6,898 distinct file identities). Candidate replays have complete native account, execution and source audits; control reuse requires exact trade/daily/equity identity. A stress test is another path under the same market sample, not another independent sample.',
        'Reconstruct the saved decision with: python scripts/portfolio_targeted_result.py assess. Regenerate this report with: python scripts/portfolio_targeted_result.py report. The publish command records the development finalist separately when source qualification fails; it does not change the canonical pointer.',
        'OOS source repair requires a separate, newly qualified input generation. Rebuild features and fixed-model predictions from that generation before freeze-replay; changing only a source_gate_pass boolean is not a repair. Once valid, run the six predeclared comparisons with full audits and apply the unchanged acceptance rules. Never choose a different finalist from OOS outcomes.',
        '', 'Methodological references',
        'Boyd et al., Multi-Period Trading via Convex Optimization: return forecasts, risk, transaction costs, holding costs and constraints belong in a shared allocation problem, but the optimizer does not create reliable forecasts. https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf',
        'Bailey and Lopez de Prado, The Deflated Sharpe Ratio: repeated strategy selection creates statistical bias. This report does not claim a deflated-Sharpe result or a selection-adjusted significance level; the full historical trial record is unavailable. https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf',
        'Korea Investment official minute-quotation example: documented request fields do not include the original/adjusted switch available on daily quotes. https://github.com/koreainvestment/open-trading-api/blob/main/examples_llm/domestic_stock/inquire_time_dailychartprice/inquire_time_dailychartprice.py',
        '', 'Evidence root: '+str(root),'Machine-readable report: '+str(path.with_suffix('.json')),
        'Registered protocol: '+str(root/'protocol.json'),'Sealed finalist: '+str(root/'development_selection.json'),
        'Acceptance: '+str(root/'acceptance.json')]
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('assess','publish','report'))
    parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args()
    if args.command=='assess':
        result=assess(args.root);write_json(args.root/'acceptance.json',result)
        print(json.dumps({'finalist':result['finalist'],'acceptance':result['acceptance']}),flush=True)
    elif args.command=='publish':publish(args.root)
    else:report(args.root)

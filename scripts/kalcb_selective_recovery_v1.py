"""Four-parent selective recovery: immutable policies and source-audited native accounts.

Register -> build/audit/control bridge -> prepare frozen cases -> supervised run.
No canonical or live changes. April--July remains reused development evidence.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date, datetime, timezone
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.kalcb_full_oos_recovery import ROOT, POLICY, read, write, rows, jsonl, bound, digest, link, env, source_hash
from scripts import kalcb_alignment_walk_forward_v3 as aligned
from scripts import kalcb_recovery_blend_checkpoint_round as blend
from scripts import kalcb_recovery_mechanism_round as mechanism
from backtests.strategies.kalcb import selective_adaptation_v1 as sa

DEST = Path('E:/CodexResearch/k_stock_trader/kalcb_selective_recovery_20261010/v1')
Q = Path('E:/CodexResearch/k_stock_trader/kalcb_recovery_qualification_20261010/v4')
P = Q.parent/'v3'
CANONICAL = REPO/'config/research/kalcb_intraday_advantage_20261007/selected_config.json'
PARENTS = ('expanding', 'expanding_atr5', 'decay60', 'decay60_atr5')
BASES = {'expanding': aligned.DEST/'fit/policies/original_target_full_history',
         'decay60': mechanism.DEST/'fit/policies/original_decay60'}
FORECASTS = {'expanding': aligned.DEST/'fit/forecasts/original_target_full_history',
             'decay60': mechanism.DEST/'fit/forecasts/original_decay60'}
RECIPES = {p+'_conditional': {'family': p.split('_atr')[0], 'mode': 'conditional', 'atr': p.endswith('atr5'), 'parents': [p]} for p in PARENTS}
for atr in (False, True):
    for cap in (.25, .5):
        name = 'expanding_core_decay'+str(int(cap*100))+('_atr5' if atr else '')
        RECIPES[name] = {'family': 'expanding', 'mode': 'hybrid', 'atr': atr, 'satellite_cap': cap,
                        'parents': ['expanding'+('_atr5' if atr else ''), 'decay60'+('_atr5' if atr else '')]}
for family in BASES:
    RECIPES[family+'_quantity_value'] = {'family': family, 'mode': 'quantity', 'atr': False,
                                        'parents': [family, family+'_atr5']}


def check(out):
    p = read(out/'protocol.json')
    if p['recipes'] != RECIPES or p['parents'] != list(PARENTS):
        raise ValueError('Changed registered recipes')
    aligned.verify_bindings(p['bindings']+p['programs'])
    return p


def register(out, tests):
    if out.exists(): raise ValueError('Preserve existing study')
    t = read(tests)
    if not t['complete'] or t['exit_code'] != 0: raise ValueError('Tests not passed')
    aligned.verify_bindings([t['log']])
    for folder in (Q, Path('E:/CodexResearch/k_stock_trader/kalcb_forecast_qualification_20261010/v1')):
        if not read(folder/'completion.json')['complete']: raise ValueError('Unqualified parents')
    inputs = [tests, Q/'completion.json', Q/'preparation.json', P/'preparation.json',
              ROOT/'next_steps_analysis_20261010_v1/completion.json', CANONICAL,
              aligned.DEST/'fit/calendar.json', aligned.ALIGNMENT/'protocol.json', aligned.ALIGNMENT/'completion.json',
              aligned.BOUNDARY/'completion.json', aligned.BOUNDARY/'overlays.json',
              Path('E:/CodexResearch/k_stock_trader/kalcb_forecast_qualification_20261010/v1/completion.json')]
    for base in BASES.values():
        for period in aligned.PERIODS:
            inputs += [p for p in (base/period).iterdir() if p.is_file()]
    for base in FORECASTS.values(): inputs += [base/(k+'.jsonl') for k in ('opening', 'timed')]
    for item in read(P/'preparation.json')['profiles']:
        inputs += [Path(item[k]['path']) for k in ('manifest', 'plan', 'policy')]
    # Binding all local research/core programs prevents drift during policy build.
    programs = set(REPO.glob('scripts/kalcb_*.py')) | set(REPO.glob('backtests/strategies/kalcb/*.py'))
    programs |= set(REPO.glob('strategy_kalcb/**/*.py')) | set(REPO.glob('backtests/auto/shared/*.py'))
    programs |= {REPO/'scripts/portfolio_frozen_inference.py', REPO/'tests/backtests/strategies/test_kalcb_selective_adaptation_v1.py'}
    out.mkdir(parents=True)
    write(out/'protocol.json', {'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'recipes': RECIPES,
        'parents': PARENTS, 'periods': aligned.PERIODS, 'bindings': [bound(p) for p in sorted(set(inputs))],
        'programs': [bound(p) for p in sorted(programs)], 'tests': t,
        'objective': 'Improve IS/OOS net returns, meaningful frequency and MTM DD jointly; preserve strong expanding IS reference.',
        'global_IS_reference': {'baseline_return': 73.1303145472, 'combined_return': 47.8121465627,
                                'sacrifice_reporting_bands_pp': [2, 5, 10]},
        'soft_dd_preferences': {'baseline': 8, 'combined': 10},
        'screening': 'Ten finite variants, both periods, baseline and combined native full source audit. All four parents retained; two quantity variants have two parent comparisons each.',
        'calibration': 'Parent-selected OOF observations, selected-management native labels,60-session history,one-session embargo,100-observation prior,min20 issuer-days/8sessions; rank/disagreement pooling,5-session update on continuous scored clock. Base20-session fit clock unchanged.',
        'hybrid': 'Protect expanding admission slots; small supported Decay-only additions. Candidate priority does not guarantee broker fill/cash priority. Native one-account replay required.',
        'quantity': 'ATR5 maximum total quantity budget; supported .25/.5/1 action grid only, revalue at actual grid quantity. Distinct from multiplicative post-score ATR overlay. No interpolated labels.',
        'deferred_conditional': 'Physical-risk calibration only after rejected-action economic/physical audit supports it; no automatic threshold relaxation. Two or three supported interactions/neighborhoods after first-wave evidence, then qualification. Later confirmation/protection experiments require mechanism evidence.',
        'invariants': 'No model/label future leakage, no timing/cash/fill changes, no date/symbol exclusions, retain losses and failures. Preserve frozen code, continuous boundary history, original fit schedule, all failed trials and OLR work.',
        'scope': 'April1-July10 is reused development; no independent OOS claim. No canonical promotion or live enablement by this program.',
        'next': 'Review all parents/components; qualify credible finalists with four scenarios, direct support, concentration, paired uncertainty and existing full final diagnostics. Do not stop solely because first-wave performance disappoints.'})
    print(json.dumps({'registered': str(out), 'variants': len(RECIPES), 'native_cases': len(RECIPES)*4}), flush=True)


def selected_targets(data):
    p = read(aligned.ALIGNMENT/'protocol.json')
    labels, physical, evidence, summary = aligned.aligned_targets(aligned.ALIGNMENT, data, p['expected_source_generation_hash'])
    for item in read(aligned.BOUNDARY/'overlays.json'):
        if item['family'] != 'selected': continue
        after = item['label']; kind = item['kind']; scenario = item['scenario']; key = after['opportunity_id']
        compact = lambda r: {k:r[k] for k in ('label_valid','label_available_at','net_value_equity_bps')}
        if labels[kind][scenario][key] != compact(item['prior_label']): raise ValueError('Selected boundary label drift')
        aligned.economic_label_contract(after)
        labels[kind][scenario][key] = compact(after)
    evidence[str(aligned.BOUNDARY/'overlays.json')] = digest(aligned.BOUNDARY/'overlays.json')
    return labels, physical, evidence, summary


def choose_day(items, raw, core, support, plan, source, *, priority=(), additions=()):
    from scripts.kalcb_action_quantity_ranker import choose_quantities
    from scripts.kalcb_action_value_study import admit_actions
    from scripts.kalcb_executed_value_study import config_for
    from backtests.strategies.kalcb import executed_value as ev
    cfg = config_for(plan); seed = read(plan['seed'])['mutations']; q3 = read(plan['q3'])['mutations']
    pred = lambda r, rank: ev.eligible(r, seed, q3, rank=rank)
    score = lambda v: {s:{k:r[s+'_value_bps'] for k,r in v.items()} for s in sa.SCENARIOS}
    cs, ns = score(core), score(support)
    choose = lambda rs, vs: choose_quantities(rs, vs['baseline'], vs['combined'], robust=True)[0]
    admit = lambda choices: admit_actions(list(choices.values()), pred, cfg, source, budget=8)[0]
    feasible = [r for r in items if raw[r['opportunity_id']]['failure_probability'] < .001]
    original = [r for r in feasible if min(raw[r['opportunity_id']][s+'_value_bps'] for s in sa.SCENARIOS) > 0]
    selected_original = admit(choose(original, cs)); locked0 = {r['base_opportunity_id']:(r,v) for r,v in selected_original}
    expanded = choose(feasible, cs); expanded.update(locked0)
    selected_core = admit(expanded); locked = {r['base_opportunity_id']:(r,v) for r,v in selected_core}
    eligible = [r for r in feasible if r['base_opportunity_id'] in locked or r['features']['action_quantity_scale'] <= .5]
    supported = choose(eligible, ns); supported.update(locked)
    selected, reasons = admit_actions(list(supported.values()), pred, cfg, source, budget=8)
    origins = {r['opportunity_id']:('frozen_core' if r['base_opportunity_id'] in locked else 'new_forecast') for r,_ in selected}
    if priority:
        selected, reasons = admit_actions(list(priority)+list(additions), pred, cfg, source, budget=8,
                                         priority_keys=[r['opportunity_id'] for r,_ in priority])
        if not {r['opportunity_id'] for r,_ in priority}.issubset({r['opportunity_id'] for r,_ in selected}):
            raise ValueError('Hybrid displaced a protected opening candidate')
        protected = {r['opportunity_id'] for r,_ in priority}
        origins = {r['opportunity_id']:('protected_expanding' if r['opportunity_id'] in protected else 'decay_satellite') for r,_ in selected}
    elif additions:
        selected, reasons = admit_actions(list(additions), pred, cfg, source, budget=8)
        origins = {r['opportunity_id']:'decay_satellite' for r,_ in selected}
    return selected, origins, reasons


def package_day(day, selected, origins, timed, tf, plan, source, tsource, valid_from):
    from backtests.strategies.kalcb import executed_value as ev
    from scripts.kalcb_action_quantity_ranker import choose_quantities
    from scripts.kalcb_checkpoint_fit import native_admit, update_snapshot
    from scripts.kalcb_checkpoint_variant import meets_value_margin
    from scripts.kalcb_executed_value_study import config_for
    snap = ev.snapshot_for(day, selected, source=source, forecast={'policy':'causal_recovery','valid_from':valid_from})
    admissions = [dict(opportunity_id=r['base_opportunity_id'], action_id=r['opportunity_id'], trade_date=day,
        reason='admitted', score=v, quantity_scale=r['features']['action_quantity_scale'], valid_from=valid_from,
        forecast_origin=origins[r['opportunity_id']]) for r,v in selected]
    eligible = [r for r in timed if r['features']['action_quantity_scale'] <= .5 and tf[r['opportunity_id']]['failure_probability'] < .001
                and meets_value_margin(r, tf[r['opportunity_id']], 48)]
    scores = {s:{k:r[s+'_value_bps'] for k,r in tf.items()} for s in sa.SCENARIOS}
    choices = choose_quantities(eligible, scores['baseline'], scores['combined'], robust=True)[0]
    extra = native_admit(snap, choices, config_for(plan), tsource)
    tads = [dict(trade_date=day, symbol=r['symbol'], available_at=r['feature_available_at'], score=v,
                 action_id=r['opportunity_id'], quantity_scale=r['features']['action_quantity_scale'],
                 valid_from=tf[r['opportunity_id']]['valid_from']) for r,v in extra]
    return {'snapshots': {date.fromisoformat(day):snap},
            'updates': {date.fromisoformat(day):[update_snapshot(snap, extra, tsource)]} if extra else {},
            'admissions': admissions, 'timed_admissions': tads}


def empty_result(): return {'snapshots':{}, 'updates':{}, 'admissions':[], 'timed_admissions':[]}


def merge(result, current):
    for k in ('snapshots','updates'): result[k].update(current[k])
    for k in ('admissions','timed_admissions'): result[k].extend(current[k])


def parent_audit(work):
    """Separate cohort tables joined by issuer-day, without duplicating cash PnL."""
    prior=read(P/'preparation.json')['profiles']; records=[]
    for name in PARENTS:
        for period in aligned.PERIODS:
            item=next(i for i in prior if i['name']==name and i['period']==period)
            policy=Path(item['policy']['path']).parent
            admitted={r['opportunity_id']:r for r in rows(policy/(POLICY+'.admission.jsonl'))}
            timed=rows(policy/'timed.admission.jsonl')
            for scenario in sa.SCENARIOS:
                folder=Q/'profiles'/name/period/scenario
                trades=defaultdict(list);codes=defaultdict(Counter)
                for t in read(folder/'trades.json'):
                    key=t['entry_decision_time'][:10]+'/'+t['symbol']
                    trades[key].append({k:t[k] for k in ('qty','entry_price','exit_price','entry_decision_time','entry_fill_time','exit_fill_time','net_pnl','commission','exit_reason')})
                with gzip.open(folder/'decisions.jsonl.gz','rt',encoding='utf-8') as source:
                    for line in source:
                        r=json.loads(line)
                        if r.get('symbol'):
                            codes[r['timestamp'][:10]+'/'+r['symbol']][r['decision_code']+'/'+r.get('reason','')]+=1
                keys=set(admitted)|set(trades)|set(codes)|{r['trade_date']+'/'+r['symbol'] for r in timed}
                for key in sorted(keys):
                    records.append({'parent':name,'period':period,'scenario':scenario,'issuer_day':key,
                        'opening_admission':admitted.get(key),
                        'checkpoint_admissions':[r for r in timed if r['trade_date']+'/'+r['symbol']==key],
                        'native_trades':trades[key],'native_decision_codes':dict(codes[key]),
                        'scope':'Actual account events, one record per parent/scenario/issuer-day. Join candidate table on family,day,symbol; do not sum duplicated quantity joins.'})
    with gzip.open(work/'parent_native_events.jsonl.gz','xt',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
    groups=defaultdict(list)
    with gzip.open(work/'decision_audit.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            gate=('capacity_rejected' if not r['capacity_pass'] else 'risk_rejected' if not r['physical_risk_pass']
                  else 'selected' if r['parent_selected'] else 'economic_or_rank_rejected')
            groups[(r['family'],r['period'],gate)].append(r)
    summary=[]
    for (family,period,gate),rs in groups.items():
        economic={}
        for s in sa.SCENARIOS:
            ls=[r['selected_management_labels'][s] for r in rs if r['selected_management_labels'][s]['label_valid']]
            economic[s]={'valid_actions':len(ls),'mean_isolated_equity_bps':sum(l['net_value_equity_bps'] for l in ls)/len(ls) if ls else None}
        summary.append({'family':family,'period':period,'gate':gate,'quantity_choices':len(rs),
            'issuer_days':len({r['day']+'/'+r['symbol'] for r in rs}),'sessions':len({r['day'] for r in rs}),
            'physical_unsafe_choices':sum(bool(r['physical_outcome']['unsafe']) for r in rs),'outcomes':economic})
    write(work/'opportunity_funnel.json',{'groups':summary,
        'risk_gate_experiment_authorized_by_this_report':False,
        'limits':'Dependent quantity alternatives; isolated common-management labels are not a funded alternative portfolio. Physical failures retained. ATR parent events have actual overlay quantities, distinct from labelled grid actions. Require temporal/risk calibration review before new gate recipes.'})


def build(out):
    from scripts.portfolio_frozen_inference import calibrated
    from scripts.kalcb_recovery_refit import assemble
    from scripts.kalcb_recovery_ablation import transform
    from scripts.kalcb_executed_value_replay import verify_policy, timed_policy
    from scripts.kalcb_action_quantity_ranker import choose_quantities, capacity_ok
    from backtests.strategies.kalcb.executed_value import outer_windows
    check(out); work = out/'build'; work.mkdir(exist_ok=False)
    data, original, sessions, evidence = blend.original_targets()
    target, physical, ep, target_summary = selected_targets(data); evidence.update(ep)
    windows = outer_windows(sessions, min_train=60, block=20)
    calendar = read(aligned.DEST/'fit/calendar.json')
    if sessions != calendar['sessions'] or [list(w) for w in windows] != calendar['windows']: raise ValueError('Global clock changed')
    cells = {(c['period'],c['kind']):c for c in read(aligned.ALIGNMENT/'protocol.json')['cells']}
    plans = {k:read(c['parent_plan']) for k,c in cells.items()}
    sources = {k:digest(Path(c['prepared'])/'opportunities.jsonl') for k,c in cells.items()}
    raw = {f:{k:{r['opportunity_id']:r for r in rows(path/(k+'.jsonl'))} for k in data} for f,path in FORECASTS.items()}
    byday = {k:defaultdict(list) for k in data}
    for kind, rs in data.items():
        for r in rs: byday[kind][r['trade_date']].append(r)
    histories = {f:[] for f in BASES}; observations = {f:[] for f in BASES}
    results = {n:{p:empty_result() for p in aligned.PERIODS} for n in RECIPES}
    controls = {f:{p:empty_result() for p in aligned.PERIODS} for f in BASES}
    proofs = {f:[] for f in BASES}; cproof = {f:[] for f in BASES}; fits = {}; audit_count = 0; scored_index = 0
    decisions = []
    with gzip.open(work/'decision_audit.jsonl.gz', 'xt', encoding='utf-8') as audit:
        for first, last in windows:
            dates = [d for d in sessions if first <= d <= last]
            cal = {f:aligned.calibrate(histories[f], original['opening'], sessions, first, last) for f in BASES}
            for f in BASES: cproof[f].extend(cal[f].values())
            for day in dates:
                period = next(p for p,(a,b,_) in aligned.PERIODS.items() if a <= day <= b)
                items, timed = byday['opening'][day], byday['timed'][day]
                now = {f:{k:{r['opportunity_id']:raw[f][k][r['opportunity_id']] for r in byday[k][day]} for k in data} for f in BASES}
                core = {f:calibrated(now[f]['opening'],cal[f]['frozen_core']) for f in BASES}
                support = {f:calibrated(now[f]['opening'],cal[f]['new_forecast']) for f in BASES}
                ds = {f:sa.descriptors(items,core[f],core['decay60' if f=='expanding' else 'expanding']) for f in BASES}
                selected = {}; origin = {}; reasons = {}
                if scored_index % 5 == 0:
                    for f in BASES:
                        fits[f] = sa.residual_fit(observations[f],target['opening'],sessions,day)
                        fits[f]['valid_through'] = sessions[min(sessions.index(day)+4,len(sessions)-1)]
                        proofs[f].append(deepcopy(fits[f]))
                corrected_core = {f:sa.corrected(core[f],ds[f],fits[f]) for f in BASES}
                corrected_support = {f:sa.corrected(support[f],ds[f],fits[f]) for f in BASES}
                for f in BASES:
                    selected[f], origin[f], reasons[f] = choose_day(items,now[f]['opening'],core[f],support[f],plans[(period,'opening')],sources[(period,'opening')])
                    fit_stamp=max((v['valid_from'] for v in now[f]['opening'].values()),default='2026-04-01')
                    control = package_day(day,selected[f],origin[f],timed,now[f]['timed'],plans[(period,'timed')],sources[(period,'opening')],sources[(period,'timed')],fit_stamp)
                    # Independent unchanged public assembly bridge for every session.
                    reference = assemble(items,timed,now[f]['opening'],now[f]['timed'],cal[f],plans[(period,'opening')],plans[(period,'timed')],[day],sources[(period,'opening')],sources[(period,'timed')])
                    if mechanism.normalized(control) != mechanism.normalized(reference): raise ValueError('Selector no-drift bridge '+f+' '+day)
                    merge(controls[f][period],control)
                    selected_ids = {r['opportunity_id'] for r,_ in selected[f]}
                    for r in items:
                        key = r['opportunity_id']; q = r['features']['action_quantity_scale']; risk = now[f]['opening'][key]['failure_probability']
                        record = {'family':f,'day':day,'period':period,'action_id':key,'symbol':r['symbol'], 'quantity':q,
                            'available_at':r['feature_available_at'],'valid_from':now[f]['opening'][key]['valid_from'],
                            'raw':{s:now[f]['opening'][key][s+'_value_bps'] for s in sa.SCENARIOS},
                            'calibrated_core':{s:core[f][key][s+'_value_bps'] for s in sa.SCENARIOS},
                            'calibrated_support':{s:support[f][key][s+'_value_bps'] for s in sa.SCENARIOS},
                            'failure_probability':risk,'capacity_pass':capacity_ok(r),'physical_risk_pass':risk < .001,
                            'descriptor':ds[f][key],'parent_selected':key in selected_ids,
                            'parent_admission_reason':reasons[f].get(key,'not_final_quantity_or_prior_gate'),
                            'selected_management_labels':{s:target['opening'][s][key] for s in sa.SCENARIOS},
                            'physical_outcome':physical['opening'][key],
                            'scope':'Outcomes only in observational audit; unavailable to decision descriptors.'}
                        audit.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n'); audit_count += 1
                    for r,v in selected[f]:
                        key = r['opportunity_id']; values = core[f] if origin[f][key]=='frozen_core' else support[f]
                        observations[f].append({**values[key],**ds[f][key], 'base_opportunity_id':r['base_opportunity_id'],
                                                'trade_date':day,'opportunity_id':key})
                for name, spec in RECIPES.items():
                    f = spec['family']; mode = spec['mode']; eligible = items
                    cs, ns = core[f], support[f]
                    if mode == 'conditional': cs, ns = corrected_core[f], corrected_support[f]
                    if mode == 'quantity': eligible = [r for r in items if sa.supported_quantity(r)]
                    kwargs = {}
                    if mode == 'hybrid':
                        incumbent = {r['base_opportunity_id'] for r,_ in selected['expanding']}
                        candidates = [r for r in items if r['base_opportunity_id'] not in incumbent
                            and r['features']['action_quantity_scale'] <= spec['satellite_cap']
                            and now['decay60']['opening'][r['opportunity_id']]['failure_probability'] < .001
                            and sa.supported_correction(ds['decay60'][r['opportunity_id']],fits['decay60'])]
                        values = corrected_support['decay60']
                        choices = choose_quantities(candidates,{k:v['baseline_value_bps'] for k,v in values.items()},
                                                    {k:v['combined_value_bps'] for k,v in values.items()},robust=True)[0]
                        kwargs = {'priority':selected['expanding'],'additions':list(choices.values())}
                    chosen, origins, rejected = choose_day(eligible,now[f]['opening'],cs,ns,plans[(period,'opening')],sources[(period,'opening')],**kwargs)
                    # No supported additions and an empty core must stay empty.
                    if mode=='hybrid' and not kwargs['priority'] and not kwargs['additions']: chosen,origins=[],{}
                    titems = [r for r in timed if sa.supported_quantity(r)] if mode=='quantity' else timed
                    result = package_day(day,chosen,origins,titems,now[f]['timed'],plans[(period,'timed')],sources[(period,'opening')],sources[(period,'timed')],first)
                    merge(results[name][period],result)
                    decisions.append({'variant':name,'day':day,'parent_actions':[r['opportunity_id'] for r,_ in selected[f]],
                                      'selected_actions':[r['opportunity_id'] for r,_ in chosen],'rejections':rejected})
                scored_index += 1
            for f in BASES:
                histories[f].extend(v for v in raw[f]['opening'].values() if first <= v['trade_date'] <= last)
            print(json.dumps({'built_through':last,'sessions':scored_index,'audit_rows':audit_count}),flush=True)
    bridges = []
    for f,base in BASES.items():
        for period in aligned.PERIODS:
            m,snap = verify_policy(base/period,POLICY)
            expected = {'snapshots':snap,'updates':timed_policy(base/period,m,snap),
                'admissions':rows(base/period/(POLICY+'.admission.jsonl')),'timed_admissions':rows(base/period/'timed.admission.jsonl')}
            if mechanism.normalized(controls[f][period]) != mechanism.normalized(expected): raise ValueError('Stored parent drift '+f+' '+period)
            bridges.append({'family':f,'period':period,'sessions':len(snap),'passed':True})
    for name,spec in RECIPES.items():
        f = spec['family']; foldproofs = []
        for parent in (list(BASES) if spec['mode']=='hybrid' else [f]):
            for period in aligned.PERIODS:
                for proof in read(BASES[parent]/period/'chronology.json')['folds']:
                    if proof not in foldproofs: foldproofs.append(proof)
        for period in aligned.PERIODS:
            folder = work/'policies'/name/period
            intermediate = work/'before_atr'/name/period if spec['atr'] else folder
            aligned.save_period(intermediate,BASES[f]/period,period,results[name][period],foldproofs,
                                cproof[f]+proofs[f]+(proofs['decay60'] if spec['mode']=='hybrid' else []),evidence,
                                {'name':name,**spec},out/'protocol.json')
            if spec['atr']: transform(intermediate,folder,'atr_quantity_5pct')
    # ATR parent policies must also exactly match the previously qualified controls.
    prior = read(P/'preparation.json')['profiles']
    for f in BASES:
        for period in aligned.PERIODS:
            folder = work/'control_atr'/f/period; transform(BASES[f]/period,folder,'atr_quantity_5pct')
            old = next(x for x in prior if x['name']==f+'_atr5' and x['period']==period)
            parent = Path(old['policy']['path']).parent
            for filename in (POLICY+'.snapshots.json',POLICY+'.admission.jsonl','timed.snapshots.jsonl','timed.admission.jsonl'):
                if digest(folder/filename)!=digest(parent/filename): raise ValueError('ATR parent drift '+filename)
            bridges.append({'family':f+'_atr5','period':period,'passed':True})
    jsonl(work/'selection_changes.jsonl',decisions)
    write(work/'calibration_proofs.json',proofs)
    write(work/'control_bridges.json',{'passed':True,'checks':bridges,'continuous_base_sessions':450,'parents':4})
    write(work/'target_audit.json',target_summary)
    parent_audit(work)
    # Bind full input/label lineage and actual saved model bytes, not only path strings.
    refs = {p:h for p,h in evidence.items()}
    for f,base in BASES.items():
        for period in aligned.PERIODS:
            for fold in read(base/period/'chronology.json')['folds']: refs[fold['model_path']] = fold['model_sha256']
    lineage = []
    for path,sha in sorted(refs.items()):
        if digest(path)!=sha: raise ValueError('Changed label/model lineage '+path)
        targetpath = work/'lineage'/ (sha+'_'+Path(path).name)
        if not targetpath.exists(): link(path,targetpath)
        lineage.append({'source':path,**bound(targetpath)})
    write(work/'lineage.json',lineage)
    check(out)
    write(work/'completion.json',{'complete':True,'audit_rows':audit_count,'control_bridge':bound(work/'control_bridges.json'),
        'calibration':bound(work/'calibration_proofs.json'),'lineage':bound(work/'lineage.json'),
        'opportunity_funnel':bound(work/'opportunity_funnel.json'),'native_parent_events':bound(work/'parent_native_events.jsonl.gz'),
        'policies':[bound(work/'policies'/n/p/'policies_manifest.json') for n in RECIPES for p in aligned.PERIODS]})


def prepare(out):
    """Clone qualified IS/OOS runtimes; replace policy/plan evidence only."""
    from backtests.strategies.kalcb.executed_value import signature
    check(out); done = read(out/'build/completion.json')
    if not done['complete']: raise ValueError('Build incomplete')
    aligned.verify_bindings(done['policies']+[done['control_bridge'],done['calibration'],done['lineage']])
    olditems = read(P/'preparation.json')['profiles']; items=[]; authenticated=set()
    for name,spec in RECIPES.items():
        for period in aligned.PERIODS:
            old = next(i for i in olditems if i['name']=='expanding' and i['period']==period)
            parent_path = Path(old['manifest']['path']); parent=read(parent_path); identity=parent.pop('manifest_hash')
            if signature(parent)!=identity: raise ValueError('Parent source manifest changed')
            folder=out/'cases'/name/period; folder.mkdir(parents=True,exist_ok=False)
            policy=out/'build/policies'/name/period
            plan=read(old['plan']['path']); context=read(plan['audit_configuration']); config=read(plan['config'])
            previous=[plan['config'],plan['audit_configuration']]
            write(folder/'config.json',config)
            context.update(output_root=str(folder),promotion_status='selective_recovery_research_only')
            write(folder/'audit_configuration.json',context)
            for key in previous: plan['input_sha256'].pop(key,None)
            for key in ('config','audit_configuration'):
                plan[key]=str(folder/(key+'.json'));plan['input_sha256'][plan[key]]=digest(plan[key])
            plan['selective_research']={'recipe':name,'period':period,'protocol':bound(out/'protocol.json')}
            write(folder/'plan.json',plan)
            files={}; prefix='research/kalcb_executed_value/policies/'
            replacements={prefix/p.name:p for p in policy.iterdir() if p.is_file()}
            replacements['research/kalcb_executed_value/plan.json']=folder/'plan.json'
            replacements['research/selective_recovery/protocol.json']=out/'protocol.json'
            for n in ('completion.json','control_bridges.json','calibration_proofs.json','lineage.json'):
                replacements['research/selective_recovery/'+n]=out/'build'/n
            for relative,sha in parent['files'].items():
                if relative.startswith(prefix) or relative in replacements: continue
                source=parent_path.parent/'files'/relative
                stat=source.stat(); key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,sha)
                if key not in authenticated:
                    if digest(source)!=sha:raise ValueError('Changed frozen source '+relative)
                    authenticated.add(key)
                link(source,folder/'inputs/files'/relative);files[relative]=sha
            for relative,source in replacements.items():link(source,folder/'inputs/files'/relative);files[relative]=digest(source)
            for key in ('config','audit_configuration'):
                source=Path(plan[key]);relative='research/selective_recovery/'+source.name
                link(source,folder/'inputs/files'/relative);files[relative]=digest(source)
            payload=deepcopy(parent);payload['files']=files
            payload['evidence']['selective_recovery']={'parent_manifest':bound(parent_path),'registration':bound(out/'protocol.json'),
                'policy':bound(policy/'policies_manifest.json'),'control_bridge':done['control_bridge']}
            code=folder/'inputs/files';payload['source_generation_hash']=source_hash(code)
            if payload['source_generation_hash']!=old['source_generation_hash']:raise ValueError('Economic runtime drift')
            write(folder/'inputs/manifest.json',{**payload,'manifest_hash':signature(payload)})
            items.append({'name':name,'period':period,'folder':str(folder),'source_generation_hash':payload['source_generation_hash'],
                          'manifest':bound(folder/'inputs/manifest.json'),'plan':bound(folder/'plan.json'),
                          'policy':bound(policy/'policies_manifest.json')})
            print(json.dumps({'prepared':name,'period':period}),flush=True)
    write(out/'preparation.json',{'complete':True,'items':items,'protocol':bound(out/'protocol.json')})
    # PhaseRunner owns evaluation, gates, phase history and complete round output.
    controls={'control':'expanding',**{'control_'+p:p for p in PARENTS if p!='expanding'}}
    write(out/'study.json',{'recipes':{**{n:{'config':{},'parent':p} for n,p in controls.items()},
                                     **{n:{'config':{},**s} for n,s in RECIPES.items()}},
        'controls':controls,'periods':{p:{'sessions':v[2]} for p,v in aligned.PERIODS.items()},
        'source_generation_hash':signature({p:next(i['source_generation_hash'] for i in items if i['period']==p) for p in aligned.PERIODS}),
        'code':str(REPO),'workers':1,'policy_name':POLICY,
        'bound_inputs':[bound(out/'protocol.json'),bound(out/'preparation.json'),bound(out/'build/completion.json')],
        'source_programs':read(out/'protocol.json')['programs']})


def validate_case(folder,item,scenario):
    from backtests.strategies.kalcb.qualified_artifacts import validate_case as validate
    start,end,count=aligned.PERIODS[item['period']];r=read(folder/'result.json')
    request=read(folder/'request.json')
    if request['plan']!=read(item['plan']['path']):raise ValueError('Replay request drift')
    if Path(request['arguments']['audit_manifest']).resolve()!=Path(item['manifest']['path']).resolve():raise ValueError('Wrong source audit manifest')
    if Path(request['arguments']['policies']).resolve()!=Path(item['folder']).resolve()/'inputs/files/research/kalcb_executed_value/policies':raise ValueError('Wrong policy directory')
    checked=validate(folder,{'execution_source':{'source_generation_hash':item['source_generation_hash']},
        'audit_manifest':item['manifest']['path'],'policy':{'name':POLICY,'manifest_sha256':item['policy']['sha256']},
        'initial_equity_krw':100_000_000,'date_range':{'start':start,'end':end,'sessions':count},'metrics':{scenario:r}},scenario)
    if checked['daily'][0]['day']!=start or checked['daily'][-1]['day']!=end:raise ValueError('Incomplete calendar')
    write(folder/'independent_account_verification.json',checked['validation'])
    return {k:r[k] for k in ('return_pct','drawdown_pct','trades','active_days','audit_status')}


def review(out):
    """Transparent global/local comparisons; never automatically promote a winner."""
    check(out); progress=read(out/'progress.json'); valid={}; failures=[]
    for r in progress:
        if r.get('rejected'): failures.append(r);continue
        valid.setdefault(r['name'],{})[r['period']+'/'+r['scenario']]=r['metrics']
    parents={p:{period+'/'+s:read(Q/'profiles'/p/period/s/'result.json') for period in aligned.PERIODS for s in sa.SCENARIOS} for p in PARENTS}
    comparisons=[]
    for name,cases in valid.items():
        if len(cases)!=4:continue
        delta={p:{k:{m:cases[k][m]-parents[p][k][m] for m in ('return_pct','drawdown_pct','trades','active_days')} for k in cases} for p in RECIPES[name]['parents']}
        sacrifices={s:parents['expanding']['is/'+s]['return_pct']-cases['is/'+s]['return_pct'] for s in sa.SCENARIOS}
        comparisons.append({'name':name,'cases':cases,'parent_deltas':delta,'global_IS_sacrifice_pp':sacrifices,
            'IS_bands':{str(b):all(v<=b for v in sacrifices.values()) for b in (2,5,10)},
            'research_only':True})
    write(out/'first_wave_comparison.json',{'parents':{p:{k:{m:v[m] for m in ('return_pct','drawdown_pct','trades','active_days')} for k,v in ps.items()} for p,ps in parents.items()},
        'comparisons':comparisons,'failed_evidence':failures,'next':'Mechanistic review and conditional second-wave registration, then four-scenario finalist qualification and full existing diagnostics. No automatic promotion.'})
    write(out/'screening_completion.json',{'complete':True,'cases':len(progress),'valid_variants':len(comparisons),'failed_cases':len(failures),
        'comparison':bound(out/'first_wave_comparison.json'),'study_complete':False})


def run(out):
    import psutil, msvcrt
    check(out)
    lock=(out/'run.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':sys.argv,'child':None}
    def status(stage,**kw):
        state.update(stage=stage,updated_at_utc=datetime.now(timezone.utc).isoformat(),**kw)
        write(out/'status.tmp',state);os.replace(out/'status.tmp',out/'status.json')
    try:
        prep=read(out/'preparation.json')
        if not prep['complete'] or (out/'progress.json').exists():raise ValueError('Preserve existing run; use versioned successor')
        progress=[]; controls=read(out/'study.json')['controls']
        def native_case(name,period,scenario):
            if name in controls:
                from scripts.kalcb_recovery_qualification_round_v4 import validate
                from backtests.strategies.kalcb.qualified_artifacts import REPLAY_FILES
                parent=controls[name];old=next(i for i in read(Q/'preparation.json')['profiles'] if i['name']==parent and i['period']==period)
                previous=Path(old['folder'])/scenario;target=out/'cases'/name/period/scenario
                validate(previous,old,scenario,previous,save=False)
                if not target.exists():
                    target.mkdir(parents=True)
                    for filename in REPLAY_FILES:link(previous/filename,target/filename)
                    write(target/'reuse.json',{'source':str(previous),'source_audited':True,'policy_reconstruction_bridge':bound(out/'build/control_bridges.json')})
                return read(previous/'result.json')
            item=next(i for i in prep['items'] if i['name']==name and i['period']==period)
            aligned.verify_bindings([item[k] for k in ('manifest','plan','policy')])
            if True:
                while psutil.virtual_memory().available<2*1024**3:
                    status('waiting_for_memory',child=None);time.sleep(30)
                if (out/'hold.json').exists():raise ValueError('Explicit study hold')
                folder=Path(item['folder']);code=folder/'inputs/files';target=folder/scenario
                if target.exists():raise ValueError('Preserve incomplete case '+str(target))
                check(out)
                command=[sys.executable,'-B','-u','-m','scripts.kalcb_executed_value_replay','--plan',item['plan']['path'],
                    '--policies',str(code/'research/kalcb_executed_value/policies'),'--policy',POLICY,'--scenario',scenario,
                    '--audit-manifest',item['manifest']['path'],'--output',str(target)]
                with (folder/(scenario+'.log')).open('x',encoding='utf-8') as log:
                    child=subprocess.Popen(command,cwd=code,env=env(code),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    status('native_replay',case=item['name']+'/'+item['period']+'/'+scenario,completed=len(progress),total=len(prep['items'])*2,
                        child={'pid':child.pid,'process_created':psutil.Process(child.pid).create_time(),'command':command})
                    while child.poll() is None:time.sleep(10)
                record={'name':item['name'],'period':item['period'],'scenario':scenario,'returncode':child.returncode,'log':bound(folder/(scenario+'.log'))}
                # Genuine implementation exceptions stop the queue; physical failures
                # with completed evidence remain rejected trials and do not hide peers.
                if child.returncode and not (target/'completion.json').exists():
                    write(out/'failed_case.json',record);raise RuntimeError('Native process failed before completion')
                try:
                    record['metrics']=validate_case(target,item,scenario);record['result']=bound(target/'result.json')
                except Exception as error:
                    # Audit/economic failures must never become scores. Preserve
                    # complete evidence, while implementation errors fail closed.
                    r=read(target/'result.json')
                    if r.get('audit_status')=='passed' and r.get('execution_validity_pass'):raise
                    record.update(rejected=True,error=repr(error));write(target/'rejected_evidence.json',record)
                progress.append(record);write(out/'progress.json',progress)
                status('case_completed',child=None,completed=len(progress));print(json.dumps(record),flush=True)
                return ({'research_rejected':True,'error':record['error']} if record.get('rejected') else read(target/'result.json'))
        from backtests.strategies.kalcb.selective_recovery_plugin_v1 import SelectiveRecoveryPlugin
        from backtests.auto.shared.phase_runner import PhaseRunner
        from backtests.auto.shared.round_manager import RoundManager
        plugin=SelectiveRecoveryPlugin(out/'study.json',native_case)
        manager=RoundManager('stock',plugin.name,base_dir=out/'framework')
        PhaseRunner(plugin,manager.get_round_dir(1),round_name='Four-parent selective recovery',round_manager=manager,
                    round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
        check(out);review(out);status('awaiting_mechanism_review',child=None)
    except Exception as error:
        status('attention_required',error=repr(error));raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['register','build','prepare','run','review'])
    p.add_argument('--root',type=Path,default=DEST);p.add_argument('--tests',type=Path);a=p.parse_args()
    if a.stage=='register':register(a.root,a.tests)
    else:globals()[a.stage](a.root)

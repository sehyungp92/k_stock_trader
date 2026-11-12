"""Finite recency, score-calibration and sizing contrasts after target alignment.

Research policies only. No economic-core changes or automatic promotion.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.kalcb_full_oos_recovery import ROOT, read, write, rows, jsonl, bound, digest, env
from scripts import kalcb_alignment_walk_forward_v3 as prior

DEST = Path('E:/CodexResearch/k_stock_trader/kalcb_recovery_mechanisms_20261010/v1')
ROUND = 'mechanism_round_v1'
RECIPES = {
    'original_rolling120': {'family': 'original', 'window': 'rolling120', 'calibration': 'offset'},
    'original_decay60': {'family': 'original', 'window': 'decay60', 'calibration': 'offset'},
    'selected_rolling120': {'family': 'selected', 'window': 'rolling120', 'calibration': 'offset'},
    'selected_decay60': {'family': 'selected', 'window': 'decay60', 'calibration': 'offset'},
    'original_affine': {'family': 'original', 'window': 'expanding', 'calibration': 'affine'},
    'selected_affine': {'family': 'selected', 'window': 'expanding', 'calibration': 'affine'},
}
SIZING = {'selected_atr4': 'atr_quantity_4pct', 'selected_atr5': 'atr_quantity_5pct'}
BASE = {'original': 'original_target_full_history', 'selected': 'selected_target_all'}


def check(out):
    p = read(out/'protocol.json')
    prior.verify_bindings(p['bound_inputs'] + p['source_programs'])
    if p['recipes'] != RECIPES or p['sizing'] != SIZING:
        raise ValueError('Changed declared experiment')
    return p


def register(out):
    if out.exists():
        raise ValueError('Preserve prior registration')
    finished = read(prior.DEST/'replay_completion.json')
    if not finished.get('complete'):
        raise ValueError('Target-alignment main contrasts incomplete')
    prior.protocol_for(prior.DEST)
    inputs = [bound(prior.DEST/'protocol.json'), bound(prior.DEST/'fit/completion.json'),
              bound(prior.DEST/'replay_completion.json'), finished['study'], finished['diagnostics'],
              bound(ROOT/'alignment_outcome_review_v1.json'), bound(ROOT/'stage2_review.json'),
              bound(ROOT/'stage3_review.json'), bound(ROOT/'mechanism_integrity_tests_20261010.log')]
    if '52 passed' not in (ROOT/'mechanism_integrity_tests_20261010.log').read_text(encoding='utf-8'):
        raise ValueError('Missing timing/calibration/recency integrity tests')
    fit = read(prior.DEST/'fit/completion.json')
    inputs += fit['models'] + fit['policies'] + [fit['bridge'], fit['target_audit'], fit['calendar']]
    for family in BASE.values():
        inputs += [bound(prior.DEST/'fit/forecasts'/family/(k+'.jsonl')) for k in ('opening','timed')]
    programs = set(REPO.glob('scripts/kalcb_*.py')) | set(REPO.glob('backtests/strategies/kalcb/*.py'))
    programs |= set(REPO.glob('strategy_kalcb/**/*.py')) | set(REPO.glob('backtests/auto/shared/*.py'))
    programs.add(REPO/'scripts/portfolio_frozen_inference.py')
    out.mkdir(parents=True)
    write(out/'protocol.json', {'registered_at_utc': datetime.now(timezone.utc).isoformat(),
        'recipes': RECIPES, 'sizing': SIZING, 'round_name': ROUND, 'periods': prior.PERIODS,
        'bound_inputs': inputs, 'source_programs': [bound(p) for p in sorted(programs)],
        'rationale': {
            'recency': 'Mature opening outcomes deteriorate after IS and vary strongly over the forward interval. Test hard120-session memory and60-session exponential half-life against the full-history expanding controls, on the same global20-session clock. Existing April1-reset rolling/decay trials are different deployment recipes.',
            'affine': 'High predicted scores remain overoptimistic and have weak/negative admitted rank correlation even with matched targets. Existing past-OOF affine calibration can shrink slope as well as intercept. Test it against conditional intercept offsets; do not flip ranks from observed OOS outcomes.',
            'sizing': 'ATR4/5 previously improved return/DD on original targets. Test this limited neighboring pair on selected-target policies with freshly replayed IS; retain any IS/activity cost.',
        },
        'controls': 'Component controls are the three completed aligned-target contrasts and prior42 retained trials. Eight new variants, no combinatorial expansion or adaptive within-run changes.',
        'invariants': 'Original features/interaction estimator, physical feasibility forecasts, global clock, embargo, mature targets, admission guards/quantity lock and exits preserved. Only declared recency or opening calibration or ATR quantity changes.',
        'calibration': 'Affine is the existing identity-shrunk fit with60-session lookback and100-observation prior, fitted on past OOF raw forecasts and matching targets. Nonnegative fitted slope. Raw-score membership of the incumbent core remains unchanged. The affine map supplies both core and support values; checkpoint scoring remains unchanged.',
        'history': 'Actual new IS and OOS for every trial, prior217-session training prefix available. No reporting-boundary reset. Separate100m accounts. No independent validation claim.',
        'lineage': 'Recency fits additionally record exact training weight hashes and session counts. Physical models reused by byte identity from matching target-family expanding control.',
        'qualification': 'Baseline/combined native screening and existing final diagnostics; four-scenario source/config/model/label and cash audits required for any finalist. DD8/10 preferences are not hard rejects. No canonical or live changes.'})
    print(json.dumps({'registered': str(out), 'trials': len(RECIPES)+len(SIZING)}), flush=True)


def affine_scores(raw, models):
    from backtests.strategies.kalcb.value_calibration import affine_value
    return {k: {**r, **{s+'_value_bps': affine_value(models[s], r[s+'_value_bps'], r['quantity_scale'])
            for s in ('baseline','combined')}} for k, r in raw.items()}


def assemble_scored(opening, timed, raw, forecast_timed, core, new, opening_plan, timed_plan, sessions, source, tsource):
    """Same admission ordering as the control, with explicit calibrated maps.

    Raw-score core membership is intentionally retained. Bridge this function
    against the registered selector using its exact original offset maps.
    """
    from scripts import kalcb_recovery_refit as selector
    cfg = selector.config_for(opening_plan); tcfg = selector.config_for(timed_plan)
    seed = read(opening_plan['seed'])['mutations']; q3 = read(opening_plan['q3'])['mutations']
    predicate = lambda row, rank: selector.ev.eligible(row, seed, q3, rank=rank)
    by_day = defaultdict(list); timed_days = defaultdict(list)
    for r in opening: by_day[r['trade_date']].append(r)
    for r in timed: timed_days[r['trade_date']].append(r)
    result = {'snapshots': {}, 'updates': {}, 'admissions': [], 'timed_admissions': []}
    scores = lambda values: {s: {k: r[s+'_value_bps'] for k,r in values.items()} for s in ('baseline','combined')}
    cs, ns, ts = scores(core), scores(new), scores(forecast_timed)
    for day in sessions:
        items = by_day[day]; feasible = [r for r in items if raw[r['opportunity_id']]['failure_probability'] < .001]
        original = [r for r in feasible if min(raw[r['opportunity_id']][s+'_value_bps'] for s in ('baseline','combined')) > 0]
        choose = lambda items, values: selector.choose_quantities(items, values['baseline'], values['combined'], robust=True)[0]
        admit = lambda choices: selector.admit_actions(list(choices.values()), predicate, cfg, source, budget=8)[0]
        original_locked = {r['base_opportunity_id']: (r,v) for r,v in admit(choose(original,cs))}
        expanded = choose(feasible,cs); expanded.update(original_locked)
        locked = {r['base_opportunity_id']: (r,v) for r,v in admit(expanded)}
        eligible = [r for r in feasible if r['base_opportunity_id'] in locked or r['features']['action_quantity_scale'] <= .5]
        supported = choose(eligible,ns); supported.update(locked); selected = admit(supported)
        fit = max((raw[r['opportunity_id']]['valid_from'] for r in items), default=selector.START)
        snap = selector.ev.snapshot_for(day, selected, source=source, forecast={'policy': 'causal_recovery', 'valid_from': fit})
        result['snapshots'][date.fromisoformat(day)] = snap
        result['admissions'].extend({'opportunity_id': r['base_opportunity_id'], 'action_id': r['opportunity_id'],
            'trade_date': day, 'reason': 'admitted', 'score': v, 'quantity_scale': r['features']['action_quantity_scale'],
            'valid_from': fit, 'forecast_origin': 'frozen_core' if r['base_opportunity_id'] in locked else 'new_forecast'} for r,v in selected)
        candidates = [r for r in timed_days[day] if r['features']['action_quantity_scale'] <= .5
            and forecast_timed[r['opportunity_id']]['failure_probability'] < .001
            and selector.meets_value_margin(r, forecast_timed[r['opportunity_id']], 48)]
        extra = selector.native_admit(snap, choose(candidates,ts), tcfg, tsource)
        if extra:
            result['updates'][date.fromisoformat(day)] = [selector.update_snapshot(snap, extra, tsource)]
            result['timed_admissions'].extend({'trade_date': day, 'symbol': r['symbol'], 'available_at': r['feature_available_at'],
                'score': v, 'action_id': r['opportunity_id'], 'quantity_scale': r['features']['action_quantity_scale'],
                'valid_from': forecast_timed[r['opportunity_id']]['valid_from']} for r,v in extra)
    return result


def normalized(result):
    return {k: {str(d): (v.to_json_dict() if k=='snapshots' else [s.to_json_dict() for s in v])
                for d,v in result[k].items()} if k in ('snapshots','updates') else result[k] for k in result}


def fit(out):
    import joblib
    from threadpoolctl import threadpool_limits
    from scripts.kalcb_recovery_refit import datasets, assemble
    from scripts.kalcb_recovery_continuous import risk_labels
    from scripts.portfolio_frozen_inference import calibrated
    from backtests.strategies.kalcb import executed_value as ev, continuation_value as cv, paired_value as pv, action_value as av
    from backtests.strategies.kalcb.value_calibration import fit_affine
    p = check(out); work = out/'fit'; work.mkdir(exist_ok=False); (work/'models').mkdir()
    data, _, original, sessions, _, evidence = datasets(ROOT, True)
    physical = {}
    for kind in data:
        physical[kind], ep = risk_labels(ROOT, kind, {r['opportunity_id'] for r in data[kind]}); evidence.update(ep)
    selected, selected_physical, ep, _ = prior.aligned_targets(prior.ALIGNMENT, data, read(ROOT/'preparation.json')['source_generation_hash'])
    boundary = prior.apply_boundary_completion(original, physical, selected, selected_physical)
    target = {'original': original, 'selected': selected}; label_proof = {'original': {**evidence, **boundary}, 'selected': {**ep, **boundary}}
    cells = {(c['period'], c['kind']): c for c in read(prior.ALIGNMENT/'protocol.json')['cells']}
    plans = {k: read(c['parent_plan']) for k,c in cells.items()}
    sources = {k: digest(Path(c['prepared'])/'opportunities.jsonl') for k,c in cells.items()}
    calendar = read(prior.DEST/'fit/calendar.json'); windows = ev.outer_windows(sessions,min_train=60,block=20)
    if sessions != calendar['sessions'] or [list(w) for w in windows] != calendar['windows']:
        raise ValueError('Changed global history clock')
    write(work/'calendar.json', calendar)
    cached = {f: {k: {r['opportunity_id']: r for r in rows(prior.DEST/'fit/forecasts'/base/(k+'.jsonl'))}
                 for k in data} for f,base in BASE.items()}
    riskproof = {}
    for family, base in BASE.items():
        proofs = []
        for period in prior.PERIODS:
            for r in read(prior.DEST/'fit/policies'/base/period/'chronology.json')['folds']:
                if r['scenario']=='failure' and r not in proofs: proofs.append(r)
        riskproof[family] = proofs
    bridge = []
    with threadpool_limits(limits=1):
        for name, recipe in RECIPES.items():
            family = recipe['family']; history = []; folds = list(riskproof[family]); calibrations = []; saved = {k: [] for k in data}
            results = {period: {'snapshots': {}, 'updates': {}, 'admissions': [], 'timed_admissions': []} for period in prior.PERIODS}
            for first,last in windows:
                foldrows = {k: [r for r in rs if first<=r['trade_date']<=last] for k,rs in data.items()}
                current = {k: {r['opportunity_id']: deepcopy(cached[family][k][r['opportunity_id']]) for r in rs} for k,rs in foldrows.items()}
                if recipe['window'] != 'expanding':
                    for kind in data:
                        for scenario in ('baseline','combined'):
                            labels = target[family][kind][scenario]
                            model, predictions, proof = cv.fit_forecast(data[kind], labels, sessions, first, last,
                                window=recipe['window'], columns=cv.COLUMNS+(pv.CHECKPOINT_COLUMNS if kind=='timed' else ()))
                            indices, weights = av.training_indices(data[kind], labels, sessions, first, av.ModelSpec('interaction', window=recipe['window']))
                            proof.update(training_weights_hash=ev.signature(weights.tolist()),
                                train_sessions=len({data[kind][i]['trade_date'] for i in indices}), target_family=family, kind=kind, scenario=scenario)
                            path = work/'models'/f'{name}_{kind}_{scenario}_{first}.joblib'; joblib.dump(model,path)
                            proof.update(model_path=str(path), model_sha256=digest(path)); folds.append(proof)
                            if {r['opportunity_id'] for r,_ in predictions} != set(current[kind]):
                                raise ValueError('Incomplete recency forecasts')
                            for r,v in predictions: current[kind][r['opportunity_id']][scenario+'_value_bps'] = v
                else:
                    for period in prior.PERIODS:
                        for r in read(prior.DEST/'fit/policies'/BASE[family]/period/'chronology.json')['folds']:
                            if r['scenario']!='failure' and r['valid_from']==first and r not in folds: folds.append(r)
                cals = prior.calibrate(history, target[family]['opening'], sessions, first, last)
                if recipe['calibration']=='affine':
                    if any(r['trade_date']>=first for r in history): raise ValueError('Future calibration forecast')
                    model, proof = fit_affine(history, target[family]['opening'], sessions, first)
                    proof.update(mode='affine_all', valid_through=last, kind='opening', history_forecasts_sha256=ev.signature(history))
                    calibrations.append(proof); scored = affine_scores(current['opening'], model)
                else:
                    calibrations.extend(cals.values())
                for period, dates in prior.split_dates(sessions,first,last).items():
                    if not dates: continue
                    op = [r for r in foldrows['opening'] if r['trade_date'] in dates]; tp = [r for r in foldrows['timed'] if r['trade_date'] in dates]
                    args = (op,tp,current['opening'],current['timed'])
                    tail = (plans[(period,'opening')],plans[(period,'timed')],dates,sources[(period,'opening')],sources[(period,'timed')])
                    if recipe['calibration']=='affine':
                        # Independently prove the parameterized adapter has no
                        # admission/snapshot drift for the existing control maps.
                        a = assemble(*args,cals,*tail)
                        b = assemble_scored(*args,calibrated(current['opening'],cals['frozen_core']),calibrated(current['opening'],cals['new_forecast']),*tail)
                        if normalized(a) != normalized(b): raise ValueError('Calibration adapter bridge failed')
                        bridge.append({'recipe': name, 'first': first, 'period': period, 'sessions': len(dates), 'passed': True})
                        chosen = assemble_scored(*args,scored,scored,*tail)
                    else:
                        chosen = assemble(*args,cals,*tail)
                    for k in ('snapshots','updates'): results[period][k].update(chosen[k])
                    for k in ('admissions','timed_admissions'): results[period][k].extend(chosen[k])
                history.extend(current['opening'].values())
                for k in data: saved[k].extend(current[k].values())
                print(json.dumps({'recipe': name, 'first': first, 'through': last}),flush=True)
            for period in prior.PERIODS:
                prior.save_period(work/'policies'/name/period, prior.DEST/'fit/policies'/BASE[family]/period,
                    period, results[period], folds, calibrations, label_proof[family], {'name': name, **recipe}, out/'protocol.json')
            folder = work/'forecasts'/name; folder.mkdir(parents=True)
            for k in data: jsonl(folder/(k+'.jsonl'),saved[k])
    write(work/'selector_bridge.json',{'passed': True,'checks':bridge,'sessions':sum(r['sessions'] for r in bridge)})
    check(out)
    write(work/'completion.json',{'complete':True,'protocol':bound(out/'protocol.json'),'bridge':bound(work/'selector_bridge.json'),
        'models':[bound(p) for p in sorted((work/'models').glob('*.joblib'))],
        'policies':[bound(work/'policies'/n/period/'policies_manifest.json') for n in RECIPES for period in prior.PERIODS]})


def replay(out):
    from scripts.kalcb_recovery_experiment_round import prepare,run
    p = check(out); done=read(out/'fit/completion.json')
    if not done.get('complete'): raise ValueError('Incomplete mechanisms fit')
    prior.verify_bindings(done['models']+done['policies']+[done['protocol'],done['bridge']])
    definitions = {n:{'config':{}, **{period+'_policy':str(out/'fit/policies'/n/period) for period in prior.PERIODS}} for n in RECIPES}
    for n,mode in SIZING.items():
        definitions[n]={'config':{},'quantity_rule':mode,**{period+'_policy':str(prior.DEST/'fit/policies/selected_target_all'/period) for period in prior.PERIODS}}
    write(out/'replay_definitions.json',definitions); prepare(ROOT,ROUND,definitions,1)
    path=ROOT/ROUND/'study.json';study=read(path)
    study['is_interpretation']='Every new trial uses a newly replayed IS account from its declared past-only learner/calibration/quantity policy. No cached IS performance for changed learner schedules.'
    study['purpose']='Finite four recency, two opening affine calibration and two selected-target ATR sizing contrasts.'
    study['bound_inputs'] += [bound(out/'protocol.json'),bound(out/'fit/completion.json')] + done['models']
    study['source_programs'] += [bound(__file__)]
    if study['cached_interaction_is']: raise ValueError('Unexpected cached learner IS')
    write(path,study);run(ROOT,ROUND)
    final=list((ROOT/ROUND/'framework').glob('*/round_1/round_final_full_diagnostics.json'))
    if len(final)!=1: raise ValueError('Missing existing full final diagnostics')
    write(out/'replay_completion.json',{'complete':True,'study':bound(path),'diagnostics':bound(final[0]),
        'next_step':'Review all outcomes/components including failed trials; source-qualify credible finalists or diagnose unresolved model ranking/calibration mechanisms. No automatic promotion.'})


def run(out):
    import msvcrt
    import psutil
    check(out)
    lock=(out/'run.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'child':None}
    def status(stage,**kwargs):
        state.update(stage=stage,updated_at_utc=datetime.now(timezone.utc).isoformat(),**kwargs)
        write(out/'status.tmp',state);os.replace(out/'status.tmp',out/'status.json')
    try:
        for stage,receipt in [('fit',out/'fit/completion.json'),('replay',out/'replay_completion.json')]:
            if receipt.exists():raise ValueError('Preserve previous stage; do not duplicate')
            check(out)
            while psutil.virtual_memory().available<2*1024**3:
                status('waiting_for_memory');time.sleep(30)
            if (out/'hold.json').exists():raise ValueError('Explicit hold')
            command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),stage,'--root',str(out)]
            with (out/(stage+'.log')).open('x',encoding='utf-8') as stream:
                child=subprocess.Popen(command,cwd=REPO,env=env(REPO),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                status(stage,child={'pid':child.pid,'process_created':psutil.Process(child.pid).create_time(),'command':command})
                start=time.monotonic()
                while child.poll() is None:
                    if time.monotonic()-start>48*3600:raise TimeoutError('Child preserved after48hours')
                    time.sleep(30)
                if child.returncode:raise RuntimeError('Failed '+stage+'; preserve evidence')
            if not read(receipt).get('complete'):raise ValueError('Incomplete stage')
            status(stage+'_completed',child=None)
        status('ready_for_mechanism_review',child=None)
    except Exception as exc:
        status('attention_required',error=repr(exc));raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['register','fit','replay','run']);parser.add_argument('--root',type=Path,default=DEST)
    args=parser.parse_args();{'register':register,'fit':fit,'replay':replay,'run':run}[args.stage](args.root)

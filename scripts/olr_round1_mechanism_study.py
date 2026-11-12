"""Versioned, mechanism-led replacement for the stopped OLR screen.

Uses the original native audited evaluator and immutable source generations.
The protocol's formerly OOS window is consumed development data. No promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import itertools
import os
from pathlib import Path
import sys
import threading
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import olr_round1_comprehensive_study as native

EXIT = 'olr.trade_plan.exit'
ENTRY = 'olr.trade_plan.entry'
RULES = native.RULES
WINDOWS = ('oos', 'train')
READOUT = 'mechanism_readout.json'


def utc():
    return datetime.now(timezone.utc).isoformat()


def changed(base, edits):
    result = copy.deepcopy(base)
    for path, value in edits:
        node = result
        for key in path[:-1]:
            node = node[key] if isinstance(node, list) else node.setdefault(key, {})
        node[path[-1]] = value
    return result


def family(candidate, base):
    paths = [str(path[0]) for path, _, _ in native.mutation_edits(base, candidate['mutations'])]
    if not paths:
        return 'control'
    if all(p == EXIT for p in paths):
        return 'exit'
    if all('allocation' in p or p.startswith('capacity_') or 'slot_count' in p for p in paths):
        return 'allocation'
    if all(p == ENTRY for p in paths):
        return 'confirmation'
    if all(p in ('olr.signal.min_parent_20d_return_pct', 'olr.research.min_sector_participation') for p in paths):
        return 'admission'
    if all(p.startswith('olr.afternoon.') for p in paths):
        return 'afternoon'
    return 'historical_or_mixed'


def band_identity(rule):
    if 'sector_admission' in rule:
        return 'dynamic_sector'
    if 'allowed_sectors' in rule:
        return 'static_sector'
    if 'max_score' in rule and 'min_score' not in rule:
        return 'low'
    if 'min_score' in rule and 'max_score' not in rule:
        return 'high'
    return 'named:'+str(rule.get('name'))


def build_plan(original):
    """Keep all 414 definitions; stage new hypotheses before the long tail."""
    baseline = original['baseline']
    candidates = copy.deepcopy(original['candidates'])
    known = {c['mutations_hash']: c for c in candidates}
    for c in candidates:
        c['mechanism_family'] = family(c, baseline)
    priority_labels = [
        'current_round1', 'remove_capacity_recycle_liquidity_only',
        'flip_olr.trade_plan.exit/hard_stop_enabled',
        'remove_olr.allocation.max_position_pct',
        'perturb_olr.allocation.max_position_pct_x0.8',
        'remove_olr.allocation.rank_decay',
        'perturb_olr.research.min_sector_participation_x0.8',
        'remove_olr.research.min_sector_participation',
        'perturb_olr.signal.min_parent_20d_return_pct_x1.2',
        'remove_olr.signal.min_parent_20d_return_pct',
        'remove_band_3', 'remove_band_2', 'no_score_bands',
        'remove_olr.trade_plan.exit/mfe_fade_start_r',
    ]
    labels = {c['label']: c for c in candidates}
    for c in candidates:
        for alias in c.get('aliases', []):
            labels.setdefault(alias['label'], c)
    missing = set(priority_labels) - set(labels)
    if missing:
        raise ValueError('Priority coverage missing: ' + str(sorted(missing)))
    priority = list(dict.fromkeys(labels[label]['id'] for label in priority_labels))
    targeted, contrasts = [], []

    def add(label, edits, reason, group):
        mutations = changed(baseline, edits)
        key = native.sig(mutations)
        if key not in known:
            c = {'id': 'c'+key[:16], 'label': label, 'kind': 'mechanism_targeted',
                 'mutations_hash': key, 'mutations': mutations, 'reason': reason,
                 'paths': [list(p) for p, _ in edits], 'aliases': [], 'experiment_group': group}
            c['mechanism_family'] = family(c, baseline)
            known[key] = c
            candidates.append(c)
        cid = known[key]['id']
        if cid not in priority and cid not in targeted:
            targeted.append(cid)
        return cid

    # Activated stops and inactive companions separate protection from changed
    # risk-unit calculations in the other managed exits.
    for mode, parameter, values in [('atr', 'stop_atr_mult', [.4, .6, .8, 1., 1.2]),
                                     ('fixed_pct', 'stop_pct', [.015, .025, .04, .06])]:
        for value in values:
            ids = {}
            for enabled in (False, True):
                ids[str(enabled)] = add(f'{mode}_{parameter}_{value}_stop_{enabled}',
                    [((EXIT, 'stop_mode'), mode), ((EXIT, parameter), value),
                     ((EXIT, 'hard_stop_enabled'), enabled)],
                    'Post-open loss containment; native executable fills, gaps and costs retained. Inactive companion isolates risk-unit effects.', 'stop_activation')
            contrasts.append({'group': 'stop_activation', 'mode': mode, 'value': value,
                              'control_id': ids['False'], 'treatment_id': ids['True']})
    grids = [
        (('olr.allocation.max_position_pct',), [.25, .35, .45, .55], 'allocation', 'Reduce rank-one exposure; quantify foregone IS gains and actual added entries.'),
        (('olr.allocation.rank_decay',), [0., .5, 1., 2.], 'allocation', 'Disperse rank weights without claiming idle cash is additional alpha.'),
        (('olr.allocation.mode',), ['selected_equal_capped'], 'allocation', 'Compare equal and rank-weighted funding.'),
        (('capacity_redistribution_max_multiplier',), [0., 2.], 'allocation', 'Use the supported policy domain; 1.2 and 1.8 from the old generic scalar sweep are invalid.'),
        (('olr.research.min_sector_participation',), [.2, .3, .4], 'admission', 'Research participation bottleneck; require profitable executed additions.'),
        (('olr.signal.min_parent_20d_return_pct',), [-30., -40., -60.], 'admission', 'Parent-return bottleneck under weak market conditions.'),
        ((RULES, 0, 'max_score'), [350., 375.], 'score_coverage', 'Modest widening of the low-score branch, preserving other branches.'),
        ((RULES, 1, 'max_score'), [450.], 'score_coverage', 'Valid single-leaf narrowing; the old 400 upper bound collapses this band.'),
        ((RULES, 2, 'max_score'), [450.], 'score_coverage', 'Valid single-leaf narrowing of the separate dynamic branch.'),
        ((RULES, 3, 'min_score'), [550., 600., 750.], 'score_coverage', 'Test high-band boundary stability; dropping it wholesale removes major IS profits.'),
        (('olr.afternoon.max_range_atr',), [1.5, 2., 3., 4.], 'overextension', 'Broad completed-prefix volatility thresholds, never symbol/date exclusions.'),
        (('olr.afternoon.max_ret',), [.05, .1, .15], 'overextension', 'Bound same-session overextension before entry.'),
        (('olr.afternoon.max_vwap_ret',), [.01, .025, .05], 'overextension', 'Bound distance above completed-prefix VWAP.'),
        (('olr.afternoon.max_score',), [800., 1200., 1600., 2400.], 'overextension', 'Test unbounded momentum scores over a broad range; flag single-event support.'),
        ((EXIT, 'vwap_fail_bars'), [3, 6], 'weakness_exit', 'Exit sustained observed weakness at next executable opportunity.'),
        ((EXIT, 'max_hold_bars'), [12, 24, 48], 'weakness_exit', 'Bound next-session exposure; measure early sale of IS winners.'),
        ((EXIT, 'mfe_fade_gap_r'), [.5, .75, 1., 1.5, 2.], 'weakness_exit', 'Sensitivity of the incumbent profit-retention exit.'),
        ((ENTRY, 'max_signal_bars'), [3, 9, 12], 'confirmation', 'Separate confirmation expiry from research-pool scarcity.'),
        ((ENTRY, 'min_close_location'), [.3, .4, .6], 'confirmation', 'Relax/tighten confirmation with both-period frequency and net return measured.'),
    ]
    for path, values, group, reason in grids:
        for value in values:
            add('/'.join(map(str, path))+'='+str(value), [(path, value)], reason, group)
    # Historical band positions changed when the dynamic middle branch was
    # inserted. Restore accepted values by rule identity, not by list index.
    semantic_coverage = []
    current_bands = baseline.get(RULES, [])
    band_indices = {band_identity(rule): i for i, rule in enumerate(current_bands)}
    if len(band_indices) != len(current_bands):
        raise ValueError('Ambiguous current score-band identity; explicit lineage repair required')
    for lineage in original.get('lineage', []):
        states = [(lineage['name'], lineage['mutations'])]
        states += [(lineage['name']+'_phase_'+str(p['phase']), p['mutations'])
                   for p in lineage.get('accepted_phases', []) if RULES in p['mutations']]
        for historical, mutations in states:
            for old_index, rule in enumerate(mutations.get(RULES, [])):
                identity = band_identity(rule)
                if identity not in band_indices:
                    raise ValueError('Unmatched historical band requires explicit coverage: '+historical+' '+identity)
                index = band_indices[identity]
                for path, value in native.flatten(rule):
                    full_path = (RULES, index, *path)
                    cid = add('semantic_'+historical+'_'+str(full_path), [(full_path, value)],
                        'Restore one historical score-band leaf aligned by rule identity, not changing list position.', 'semantic_history')
                    semantic_coverage.append({'historical':historical,'identity':identity,
                        'historical_index':old_index, 'current_index':index,
                        'leaf':list(path),'value':value,'candidate_id':cid})
    # A small predeclared factorial catches complementary bottlenecks even when
    # either single change is dormant. Every single cell exists above.
    factorial_edits = {
        'stop': [((EXIT, 'hard_stop_enabled'), True)],
        'cap': [(('olr.allocation.max_position_pct',), .45)],
        'parent': [(('olr.signal.min_parent_20d_return_pct',), -40.)],
        'sector': [(('olr.research.min_sector_participation',), .3)],
    }
    factorial = []
    for a, b in [('stop', 'cap'), ('stop', 'parent'), ('stop', 'sector'), ('parent', 'sector')]:
        left = add('factor_'+a, factorial_edits[a], 'Factorial component', 'factorial_component')
        right = add('factor_'+b, factorial_edits[b], 'Factorial component', 'factorial_component')
        both = add('factor_'+a+'_'+b, factorial_edits[a]+factorial_edits[b],
                   'Predeclared complementary risk/admission test; full four-cell attribution.', 'factorial')
        factorial.append({'control_id': original['baseline_id'], 'left_id': left, 'right_id': right, 'both_id': both})
    done = set(priority+targeted)
    remainder = sorted((c for c in original['candidates'] if c['id'] not in done),
                       key=lambda c: (c['kind'] != 'cumulative', c['kind'] != 'historical_leaf_ablation'))
    return {'version': 7, 'registered_utc': utc(), 'baseline_id': original['baseline_id'],
            'baseline': baseline, 'candidates': candidates,
            'original_candidate_ids': [c['id'] for c in original['candidates']],
            'phases': [
                {'name': 'priority_ablation', 'ids': priority},
                {'name': 'targeted_mechanisms', 'ids': targeted},
                {'name': 'early_interactions', 'adaptive': True},
                {'name': 'complete_historical_ablation', 'ids': [c['id'] for c in remainder]},
                {'name': 'historical_interactions', 'adaptive': True},
                {'name': 'finalist_backouts', 'adaptive': True}],
            'activation_contrasts': contrasts, 'factorials': factorial,
            'semantic_band_coverage': semantic_coverage,
            'historical_position_warning': 'Original positional contrasts retained for reproducibility. In legacy round 4 the high band was index 2; it is now index 3. Its old min_score=650 applied to current index 2 is invalid and is not an accepted-mutation ablation. Semantic coverage records the correctly aligned control.',
            'final_gates': {'is_return_floor_pp': -3, 'is_wealth_floor_ratio': .97,
                'oos_gain_floor_pp': 2, 'trade_floor_ratio_each_window': 1., 'dd_increase_ceiling_pp': 2},
            'adaptive_rules': {'max_components_per_family': 3, 'component_is_floor_pp': -10,
                'component_oos_floor_pp': -2, 'max_interactions_each_stage': 60,
                'backouts': 'Each changed leaf of top three strict finalists restored independently to baseline; no new threshold fitting.'},
            'warning': 'OOS-informed development. No global optimum or independent expected-return claim. No automatic promotion.'}


def register(root):
    path = root/'mechanism_plan.json'
    if path.exists():
        raise ValueError('Registered mechanism plan already exists; version rather than overwrite it')
    original = native.read(root/'candidate_plan.json')
    plan = build_plan(original)
    plan['source_evidence'] = {name: native.sha(root/name) for name in
        ('candidate_plan.json', 'mechanism_diagnostics.json', 'mechanism_followup_plan.json',
         'lineage_attribution.json', 'historical_leaf_coverage_audit.json')}
    plan['runner_sha256'] = native.sha(__file__)
    plan['native_evaluator_sha256'] = native.sha(native.__file__)
    native.write(path, plan)
    protocol = native.read(root/'protocol.json')
    protocol.update(harness_revision=7, harness_sha256=native.sha(native.__file__),
        orchestration_script=str(Path(__file__).resolve()), orchestration_sha256=native.sha(__file__),
        mechanism_plan_sha256=native.sha(path),
        phases=[p['name'] for p in plan['phases']]+['paired_robustness', 'support_and_uncertainty'],
        revision={'at_utc': utc(), 'reason': 'User stopped the original screen. Mechanism-led staged replacement, paired activation controls, factorials, complete retained lineage and leaf coverage, failure gates and finalist backouts.'})
    native.write(root/'protocol.json', protocol)
    native.event(root, 'mechanism_revision_registered', counts={p['name']:len(p.get('ids', [])) for p in plan['phases']},
                 total_fixed_candidates=len(plan['candidates']))


@contextmanager
def process_lock(root):
    """OS-held lock is automatically released if this supervisor is killed."""
    import msvcrt
    with (root/'mechanism_supervisor.lock').open('a+b') as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('Another mechanism supervisor holds this study lock') from exc
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def validate_cached(root, candidate, window, capital=100_000_000, scenario='baseline', capped=False):
    folder = root/'cases'/candidate['id']/window/(f'{capital}_{scenario}'+('_capped' if capped else ''))
    if not (folder/'assessment.json').exists():
        return
    assessment = native.read(folder/'assessment.json')
    if assessment['status'] in ('invalid_configuration', 'unsupported_execution_contract'):
        return
    if not assessment.get('qualified'):
        if reviewed_capacity_rejection(root, candidate, window, capital, scenario, capped):
            return
        if reviewed_source_unavailability(root, candidate, window, capital, scenario, capped):
            return
        raise ValueError('Unresolved failed evidence requires repair, not automatic skipping: '+str(folder))
    receipt = native.read(folder/'completion.json')
    result = native.read(folder/'result.json')
    audit = native.read(folder/'execution_audit.json')
    if native.sha(folder/'result.json') != receipt['result_sha256'] or native.sha(folder/'execution_audit.json') != receipt['audit_sha256']:
        raise ValueError('Cached native evidence changed: '+str(folder))
    if not audit.get('audit_pass') or result.get('audit_status') != 'passed' or not result.get('execution_validity_pass'):
        raise ValueError('Cached audit is invalid: '+str(folder))
    for key in ('return_pct', 'trades', 'drawdown_pct'):
        if assessment[key] != result[key]:
            raise ValueError('Cached assessment no longer matches native result: '+str(folder))
    registered = root/'registrations'/window/(candidate['id']+'_'+str(capital))
    bound_study = native.execution_registration(registered/'study.json', receipt['source_generation_hash'])
    if result['source_generation_hash'] != receipt['source_generation_hash']:
        raise ValueError('Cached native result source identity changed')
    native.verify_window_contract(root, window, bound_study, folder)
    if native.read(registered/'candidate.json')['mutations_hash'] != candidate['mutations_hash']:
        raise ValueError('Cached candidate identity changed')
    if result.get('terminal_inventory') or any(result.get(k, 0) for k in
        ('terminal_pending', 'terminal_deferred_intents', 'terminal_rotation_intents')):
        raise ValueError('Unfinished terminal account state')


def reviewed_capacity_rejection(root, candidate, window, capital=100_000_000, scenario='baseline', capped=False):
    """A manually reviewed physical rejection stays failed, never a performance row.

    Only exact, registered evidence may pass this orchestration gate. New failures
    still stop dispatch. Neither simulated fills nor native audits are rewritten.
    """
    plan_path = root/'mechanism_plan.json'
    if not plan_path.exists():
        return None
    plan = native.read(plan_path)
    key = '/'.join((candidate['id'], window, f'{capital}_{scenario}'+('_capped' if capped else '')))
    registration = plan.get('reviewed_capacity_rejections', {}).get(key)
    if registration is None:
        return None
    review_path = root/registration['path']
    if native.sha(review_path) != registration['sha256']:
        raise ValueError('Capacity rejection review changed')
    review = native.read(review_path)
    if review['case_key'] != key or review['mutations_hash'] != candidate['mutations_hash']:
        raise ValueError('Capacity rejection candidate identity changed')
    folder = root/'cases'/key
    for name, digest in review['evidence_sha256'].items():
        if native.sha(folder/name) != digest:
            raise ValueError('Reviewed failed evidence changed: '+name)
    receipt, result, audit = (native.read(folder/n) for n in
        ('completion.json', 'result.json', 'execution_audit.json'))
    if (not receipt.get('complete') or receipt['result_sha256'] != native.sha(folder/'result.json')
            or receipt['audit_sha256'] != native.sha(folder/'execution_audit.json')):
        raise ValueError('Incomplete reviewed capacity failure')
    if (result.get('execution_validity_pass') or audit.get('audit_pass')
            or result.get('audit_status') != 'failed'):
        raise ValueError('A capacity rejection cannot be qualified')
    allowed = {'shared_bar_volume_exceeded'}
    capacity = audit.get('capacity_model_audit') or {}
    if (set(audit['audit_receipt']['errors']) != allowed or set(capacity.get('errors', [])) != allowed
            or capacity.get('contract', {}).get('realized_volume') != 'ex_post_impossibility_check_only'
            or not capacity.get('impossible_volume_events')):
        raise ValueError('Unreviewed audit defect in capacity rejection')
    for event in capacity['impossible_volume_events']:
        if not event['bar_total_filled_shares'] > event['printed_shares']:
            raise ValueError('Capacity rejection lacks physical evidence')
    registered = root/'registrations'/window/(candidate['id']+'_'+str(capital))
    study = native.execution_registration(registered/'study.json', receipt['source_generation_hash'])
    if (native.read(registered/'candidate.json')['mutations_hash'] != candidate['mutations_hash']
            or study['source_generation_hash'] != receipt['source_generation_hash']
            or result['source_generation_hash'] != receipt['source_generation_hash']):
        raise ValueError('Capacity rejection registration changed')
    native.verify_window_contract(root, window, study, folder)
    if result.get('terminal_inventory') or any(result.get(k, 0) for k in
        ('terminal_pending', 'terminal_deferred_intents', 'terminal_rotation_intents')):
        raise ValueError('Unfinished rejected account')
    return {'candidate_id':candidate['id'], 'label':candidate['label'], 'window':window,
        'capital':capital, 'scenario':scenario, 'capped':capped, 'qualified':False,
        'status':'execution_infeasible', 'error':'shared_bar_volume_exceeded',
        'output':str(folder), 'review':str(review_path),
        'disposition':'Native audit failed. No accepted performance estimate. Exact evidence retained; related admission experiments continue.'}


def reviewed_source_unavailability(root, candidate, window, capital=100_000_000, scenario='baseline', capped=False):
    """Retain a specific unrecoverable-data trial without accepting its PnL.

    A source gap is not a bad strategy result or an executable counterfactual.
    New gaps still stop dispatch and require independent recovery investigation.
    """
    path = root/'mechanism_plan.json'
    if not path.exists():
        return None
    key = '/'.join((candidate['id'], window, f'{capital}_{scenario}'+('_capped' if capped else '')))
    registration = native.read(path).get('reviewed_source_unavailability', {}).get(key)
    if registration is None:
        return None
    review_path = root/registration['path']
    if native.sha(review_path) != registration['sha256']:
        raise ValueError('Source unavailability review changed')
    review = native.read(review_path)
    if review['case_key'] != key or review['mutations_hash'] != candidate['mutations_hash']:
        raise ValueError('Source unavailability candidate identity changed')
    folder = root/'cases'/key
    for name, digest in review['evidence_sha256'].items():
        if native.sha(folder/name) != digest:
            raise ValueError('Reviewed unverified evidence changed: '+name)
    for name, digest in review['recovery_evidence_sha256'].items():
        if native.sha(root/name) != digest:
            raise ValueError('Source recovery evidence changed')
    completion, result, audit = (native.read(folder/n) for n in
        ('completion.json', 'result.json', 'execution_audit.json'))
    receipt = audit['audit_receipt']
    if (not completion.get('complete') or completion['result_sha256'] != native.sha(folder/'result.json')
            or completion['audit_sha256'] != native.sha(folder/'execution_audit.json')):
        raise ValueError('Incomplete reviewed source failure')
    if (not result.get('execution_validity_pass') or audit.get('audit_pass')
            or result.get('audit_status') != 'unverified' or receipt.get('errors')):
        raise ValueError('Unreviewed execution defect in source unavailability')
    capacity = audit.get('capacity_model_audit') or {}
    if capacity.get('status') != 'model_consistent' or capacity.get('errors'):
        raise ValueError('Unreviewed capacity defect in source unavailability')
    reviewed_coverage = review.get('reviewed_unverified_coverage', {
        'daily_marks':'unverified', 'replay_completeness':'unverified'})
    allowed_source_coverage = (
        {'daily_marks':'unverified', 'replay_completeness':'unverified'},
        {'replay_completeness':'unverified'},
    )
    if (reviewed_coverage not in allowed_source_coverage or
            {k:v for k,v in receipt['coverage'].items() if v not in ('pass','not_applicable')}
            != reviewed_coverage):
        raise ValueError('Additional unverified audit coverage')
    if reviewed_coverage == {'replay_completeness':'unverified'}:
        daily = receipt['daily_equity_coverage']
        if (receipt['coverage'].get('daily_marks') != 'pass' or daily.get('status') != 'complete'
                or daily.get('missing_sessions') or daily.get('stale_or_unknown_mark_sessions')
                or daily.get('cash_only_padded_sessions')):
            raise ValueError('Candidate-feed-only review requires complete daily marks')
    if (receipt['coverage'] != review['expected_coverage']
            or receipt['daily_equity_coverage'] != review['expected_daily_coverage']
            or receipt['qualification']['shared_scopes'] != review['expected_qualification_scopes']):
        raise ValueError('Unreviewed source coverage defect')
    registered = root/'registrations'/window/(candidate['id']+'_'+str(capital))
    study = native.execution_registration(registered/'study.json', completion['source_generation_hash'])
    if (native.read(registered/'candidate.json')['mutations_hash'] != candidate['mutations_hash']
            or study['source_generation_hash'] != completion['source_generation_hash']
            or result['source_generation_hash'] != completion['source_generation_hash']):
        raise ValueError('Source unavailability registration changed')
    native.verify_window_contract(root, window, study, folder)
    if result.get('terminal_inventory') or any(result.get(k, 0) for k in
        ('terminal_pending', 'terminal_deferred_intents', 'terminal_rotation_intents')):
        raise ValueError('Unfinished source-unverified account')
    return {'candidate_id':candidate['id'], 'label':candidate['label'], 'window':window,
        'capital':capital, 'scenario':scenario, 'capped':capped, 'qualified':False,
        'status':'source_unverified', 'output':str(folder), 'review':str(review_path),
        'error':'Missing management/closing observation; recovery evidence reviewed',
        'disposition':'No accepted performance estimate. Missing source remains unresolved; other registered leaf experiments and the other window continue.'}


def unsupported_exit_orders(configuration):
    """The frozen causal capacity model supports open/auction, not intrabar fills."""
    plan = configuration.get('trade_exit_plan') or {}
    if plan.get('mode', 'next_close') != 'managed':
        return []
    unsupported = []
    if plan.get('hard_stop_enabled', False):
        unsupported.append('intrabar_protective_stop')
    if plan.get('target_r', 0) > 0:
        unsupported.append('intrabar_limit_target')
    if plan.get('partial_trigger_r', 0) > 0 and plan.get('partial_fraction', 0) > 0:
        unsupported.append('intrabar_partial_limit_target')
    return unsupported


def evaluate(root, candidate, window, capital=100_000_000, scenario='baseline', capped=False):
    validate_cached(root, candidate, window, capital, scenario, capped)
    rejected = reviewed_capacity_rejection(root, candidate, window, capital, scenario, capped)
    if rejected:
        return rejected
    unavailable = reviewed_source_unavailability(root, candidate, window, capital, scenario, capped)
    if unavailable:
        return unavailable
    try:
        configuration = native.candidate_preflight(root, window, candidate)
    except ValueError as exc:
        # A documented impossible single-leaf combination is evidence, but is
        # never reported as an audited replay or a strategy improvement.
        out = root/'cases'/candidate['id']/window/(f'{capital}_{scenario}'+('_capped' if capped else ''))
        row = {'candidate_id': candidate['id'], 'label': candidate['label'], 'window': window,
               'capital': capital, 'scenario': scenario, 'capped': capped, 'qualified': False,
               'status': 'invalid_configuration', 'error': str(exc), 'output': str(out),
               'disposition': 'Retained invalid single-leaf trial; review dependency companions, never a performance result.'}
        native.write(out/'assessment.json', row)
        return row
    unsupported = unsupported_exit_orders(configuration)
    if unsupported:
        info = native.read(native.generation_folder(root, window)/'ready.json')
        qualification = native.read(Path(info['files'])/info['manifest']['evidence']['qualification_path'])
        if qualification.get('capacity', {}).get('kind') != 'declared_bar_participation_model':
            raise ValueError('Execution capability disposition must be reviewed for the changed capacity contract')
        out = root/'cases'/candidate['id']/window/(f'{capital}_{scenario}'+('_capped' if capped else ''))
        row = {'candidate_id':candidate['id'],'label':candidate['label'],'window':window,'capital':capital,
            'scenario':scenario,'capped':capped,'qualified':False,'status':'unsupported_execution_contract',
            'error':','.join(unsupported),'output':str(out),
            'evidence':'revisions/revision_8_supported_exit_contract/failed_intrabar_stop_oos/execution_audit.json',
            'disposition':'Not an audited performance trial. Intrabar stops/limits are unsupported by this declared capacity contract. Completed-bar follow-through/VWAP/MFE exits with audited next-open fills test the mitigation mechanism separately.'}
        native.write(out/'assessment.json',row)
        return row
    row = native.evaluate(root, candidate, window, capital, scenario, capped)
    if not row.get('qualified'):
        raise RuntimeError('Native replay/audit failed; repair before continuing this category: '+row['output'])
    validate_cached(root, candidate, window, capital, scenario, capped)
    return row


def batch(root, candidates, workers, phase):
    jobs = [dict(candidate=candidate, window=window, index=index+1)
            for index, candidate in enumerate(candidates) for window in WINDOWS]
    run_evaluation_jobs(root, jobs, workers, phase)
    native.write(root/(phase+'_results.json'), native.results(root))


def run_evaluation_jobs(root, jobs, workers, phase):
    """Bounded work queue: short OOS jobs no longer leave an idle worker later.

    Submit only active slots, stop dispatch immediately after any new failure,
    and let already-running native jobs finish. Preparation is serialized per window by
    the evaluator; full native replay/audit is parallel outside that lock.
    """
    if not 1 <= workers <= 4:
        raise ValueError('Reviewed study concurrency is 1 to 4 workers')
    failed = threading.Event()
    totals = Counter(j['window'] for j in jobs)
    completed, active = Counter(), {}
    pending = iter(jobs)
    plan_path = root/'mechanism_plan.json'
    admission = native.read(plan_path).get('resource_admission') if plan_path.exists() else None
    identities = [(j['candidate']['id'], j['window'], j.get('capital',100_000_000),
                   j.get('scenario','baseline'),j.get('capped',False)) for j in jobs]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate evaluation jobs')

    def execute(job):
        if failed.is_set() or (root/'STOP_MECHANISM.json').exists():
            raise RuntimeError('Mechanism study stopped before next dispatch')
        try:
            return evaluate(root, **{k:v for k,v in job.items() if k != 'index'})
        except Exception:
            failed.set()
            raise

    def progress(job, value=None):
        window = job['window']
        row = {'phase':phase, 'window':window, 'index':job['index'], 'total':totals[window],
               'candidate':job['candidate']['id'], 'label':job['candidate']['label'],
               'status':value['status'] if value else 'running', 'utc':utc(),
               'completed_evaluations':completed[window],
               'active_candidates':[j['candidate']['id'] for j in active.values() if j['window']==window]}
        if value:
            row['qualified'] = value['qualified']
        native.write(root/f'progress_{window}.json', row)
        print(__import__('json').dumps(row), flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        exhausted = False
        while active or not exhausted:
            if (root/'STOP_MECHANISM.json').exists():
                failed.set()
                raise RuntimeError('Mechanism study stopped before next dispatch')
            while len(active) < workers and not exhausted and not failed.is_set():
                if admission and active:
                    from scripts.olr_native_acceleration import available_memory_mb, additional_job_allowed
                    available = available_memory_mb()
                    if not additional_job_allowed(admission, len(active), available):
                        native.write(root/'resource_admission_status.json', {
                            'utc':utc(), 'active_jobs':len(active), 'maximum_workers':workers,
                            'available_mb':available, 'policy':admission,
                            'status':'waiting_for_memory_headroom', 'active_jobs_interrupted':False})
                        break
                job = next(pending, None)
                if job is None:
                    exhausted = True
                    break
                active[pool.submit(execute, job)] = job
                progress(job)
            if not active:
                break
            done, _ = wait(active, timeout=5 if admission else None, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try:
                    value = future.result()
                except Exception as exc:
                    failed.set()
                    native.event(root, 'mechanism_needs_repair', phase=phase, window=job['window'],
                                 candidate=job['candidate']['id'], error=str(exc))
                    raise
                completed[job['window']] += 1
                progress(job, value)


def catalog(root):
    plan = native.read(root/'mechanism_plan.json')
    all_candidates = {c['id']:c for c in plan['candidates']}
    for path in sorted((root/'adaptive_plans').glob('*.json')):
        all_candidates.update({c['id']:c for c in native.read(path)['candidates']})
    return all_candidates


def historical_first_phases(plan, predeclared_ids=()):
    """Finish accepted-history contrasts before testing new mechanisms."""
    historical = set(plan['original_candidate_ids'])
    historical.update(r['candidate_id'] for r in plan.get('semantic_band_coverage', []))
    known = {c['id'] for c in plan['candidates']}
    if not historical <= known:
        raise ValueError('Historical coverage references an unregistered candidate')
    ordered = list(dict.fromkeys(cid for p in plan['phases'] for cid in p.get('ids', [])))
    # Some supported MFE companions live exclusively in the retained,
    # predeclared interaction plan, rather than a fixed phase's ID list.
    if not set(ordered) <= known or not known <= (set(ordered) | set(predeclared_ids)):
        raise ValueError('Every registered definition must have a phase')
    ordered.extend(sorted(historical-set(ordered)))
    priority = [cid for p in plan['phases'] if p['name'] == 'priority_ablation'
                for cid in p['ids'] if cid in historical]
    return [
        {'name':'priority_ablation', 'ids':priority},
        {'name':'complete_historical_ablation', 'ids':[cid for cid in ordered if cid in historical and cid not in priority]},
        {'name':'targeted_mechanisms', 'ids':[cid for cid in ordered if cid not in historical]},
        *[copy.deepcopy(p) for p in plan['phases'] if p.get('adaptive')],
    ]


def component_screen(known, pairs, baseline_id):
    """Screen complements using each window, without confusing final gates."""
    control = pairs[baseline_id]
    rows = []
    for cid, pair in pairs.items():
        if cid == baseline_id or cid not in known or set(pair) != set(WINDOWS):
            continue
        gain = {w:pair[w]['return_pct']-control[w]['return_pct'] for w in WINDOWS}
        frequency = {w:pair[w]['trades']/control[w]['trades'] for w in WINDOWS}
        useful = ([w+'_return' for w in WINDOWS if gain[w] > 1e-9]
                  + [w+'_frequency' for w in WINDOWS if frequency[w] > 1])
        rows.append({'id':cid, 'gain':gain, 'frequency_by_window':frequency,
            'frequency_ratio':min(frequency.values()), 'family':known[cid]['mechanism_family'],
            'eligible_component':gain['train'] >= -10 and gain['oos'] >= -2 and bool(useful),
            'positive_dimensions':useful})
    return rows


def select_components(rows):
    # Reserve a representative for each return/frequency objective so a positive
    # IS, unchanged-OOS complement cannot disappear behind OOS-only ranking.
    objectives = [('gain','oos'), ('gain','train'),
                  ('frequency_by_window','oos'), ('frequency_by_window','train')]
    ids = []
    for field, window in objectives:
        ranked = sorted(rows, key=lambda r:(r[field][window], r['gain']['oos'],
                                           r['gain']['train'], r['id']), reverse=True)
        ids.append(ranked[0]['id'])
    ids.append(max(rows,key=lambda r:(r['frequency_ratio'],r['gain']['oos'],r['id']))['id'])
    return list(dict.fromkeys(ids))


def adaptive_plan(root, phase):
    path = root/'adaptive_plans'/(phase+'.json')
    if path.exists():
        return native.read(path)['candidates']
    known = catalog(root)
    plan = native.read(root/'mechanism_plan.json')
    base = plan['baseline']
    leaders, pairs = native.rank_pairs(root)
    control = pairs[plan['baseline_id']]
    additions, seen = [], {c['mutations_hash'] for c in known.values()}
    evidence, screened = [], []
    def add(mutations, label, parents, reason):
        key = native.sig(mutations)
        if key in seen:
            return
        seen.add(key)
        additions.append({'id':'c'+key[:16], 'label':label, 'kind':phase, 'mutations':mutations,
            'mutations_hash':key, 'aliases':[], 'parents':parents, 'reason':reason,
            'mechanism_family':family({'mutations':mutations}, base)})
    if phase == 'finalist_backouts':
        for leader in leaders[:3]:
            candidate = known[leader['id']]
            # Undo each edit independently on the winning composition. Unlike a
            # baseline ablation, this tests whether its accepted complexity earns its place.
            for path_, value, remove in native.mutation_edits(candidate['mutations'], base):
                mutations = native.set_leaf(candidate['mutations'], path_, value, remove)
                add(mutations, 'backout_'+candidate['id']+'_'+str(path_), [candidate['id']],
                    'Undo one finalist edit against baseline; test marginal necessity and simplify unsupported complexity.')
    else:
        groups = defaultdict(list)
        screened = component_screen(known, pairs, plan['baseline_id'])
        for row in screened:
            if row['eligible_component']:
                groups[row['family']].append(row)
        selected = []
        for group in sorted(groups):
            rows = groups[group]
            ids = select_components(rows)
            selected.extend(ids)
            evidence.extend(r for r in rows if r['id'] in ids)
        # Interleave family pairs so the cap cannot exhaust the budget on the
        # first alphabetic family and omit exit/allocation interactions.
        by_family = defaultdict(list)
        for cid in selected:
            by_family[known[cid]['mechanism_family']].append(cid)
        queues = [list(itertools.product(by_family[a],by_family[b]))
                  for a,b in itertools.combinations(sorted(by_family),2)]
        balanced = [pair for layer in itertools.zip_longest(*queues) for pair in layer if pair]
        for left, right in balanced:
            mutations = native.combine_disjoint(base, known[left]['mutations'], known[right]['mutations'])
            if mutations is not None:
                add(mutations, phase+'_'+left+'_'+right, [left,right],
                    'Disjoint complementary mechanisms selected from matched paired evidence; final gates unchanged.')
            if len(additions) >= 60:
                break
    native.write(path, {'registered_utc':utc(), 'phase':phase, 'candidates':additions,
        'all_component_screen_evidence':screened,
        'selected_component_evidence':evidence, 'parent_results_sha256':native.sig(native.results(root)),
        'warning':'Adaptive development trials, counted in full; component gates are not finalist acceptance gates.'})
    return additions


def support_comparison(control, treatment):
    """Account delta and issuer/day execution attribution, not trade deletion backtests."""
    left = native.read(Path(control['output'])/'trades.json')
    right = native.read(Path(treatment['output'])/'trades.json')
    def aggregate(rows):
        out = {}
        for t in rows:
            key = (t['entry_fill_time'][:10], t['symbol'])
            row = out.setdefault(key, {'pnl':0., 'qty':0, 'entry_notional':0., 'executions':[]})
            row['pnl'] += t['net_pnl']; row['qty'] += t['qty']
            row['entry_notional'] += t['qty']*t['entry_price']
            row['executions'].append({k:t.get(k) for k in ('entry_fill_time','exit_fill_time','qty','entry_price','exit_price','net_pnl','exit_reason')})
        return out
    a, b = aggregate(left), aggregate(right)
    rows = []
    for key in sorted(set(a)|set(b)):
        x, y = a.get(key, {}), b.get(key, {})
        if x == y:
            continue
        rows.append({'day':key[0], 'symbol':key[1], 'control':x, 'treatment':y,
            'delta_net_pnl':y.get('pnl',0)-x.get('pnl',0),
            'type':'new' if not x else 'removed' if not y else 'changed'})
    gains = sorted((r['delta_net_pnl'] for r in rows if r['delta_net_pnl']>0), reverse=True)
    total = sum(r['delta_net_pnl'] for r in rows)
    months = defaultdict(float)
    for row in rows:
        months[row['day'][:7]] += row['delta_net_pnl']
    return {'changed_issuer_days':len(rows), 'changed_days':len({r['day'] for r in rows}),
        'new_issuer_days':sum(r['type']=='new' for r in rows),
        'removed_issuer_days':sum(r['type']=='removed' for r in rows),
        'pnl_delta':total, 'positive_contribution_top2_share':sum(gains[:2])/sum(gains) if gains else None,
        'pnl_delta_excluding_two_best_contributors_descriptive':total-sum(gains[:2]),
        'entry_month_delta_pnl':dict(months),
        'support_flag':'sparse' if len(rows)<10 else 'at_least_10_changed_issuer_days',
        'rows':rows, 'warning':'Post-selection attribution. Removing observed contributors is arithmetic sensitivity, not executable performance.'}


def readout(root):
    native.report(root)
    plan = native.read(root/'mechanism_plan.json')
    leaders, pairs = native.rank_pairs(root)
    control = pairs.get(plan['baseline_id'], {})
    all_candidates = catalog(root)
    paired, contrasts, factorials = [], [], []
    for cid, pair in pairs.items():
        if cid not in all_candidates or set(pair) != set(WINDOWS) or set(control) != set(WINDOWS):
            continue
        paired.append({'id':cid, 'label':all_candidates[cid]['label'],
            'family':all_candidates[cid]['mechanism_family'],
            'return_gain_pp':{w:pair[w]['return_pct']-control[w]['return_pct'] for w in WINDOWS},
            'trades':{w:pair[w]['trades'] for w in WINDOWS},
            'activity_delta':{w:pair[w]['trades']-control[w]['trades'] for w in WINDOWS},
            'drawdown':{w:pair[w]['drawdown_pct'] for w in WINDOWS}})
    for item in plan['activation_contrasts']:
        a, b = pairs.get(item['control_id'], {}), pairs.get(item['treatment_id'], {})
        if set(a)==set(b)==set(WINDOWS):
            contrasts.append({**item, 'return_gain_pp':{w:b[w]['return_pct']-a[w]['return_pct'] for w in WINDOWS},
                'trade_delta':{w:b[w]['trades']-a[w]['trades'] for w in WINDOWS}})
    for item in plan['factorials']:
        cells = [pairs.get(item[k], {}) for k in ('control_id','left_id','right_id','both_id')]
        if all(set(p)==set(WINDOWS) for p in cells):
            factorials.append({**item, 'interaction_pp':{w:cells[3][w]['return_pct']-cells[1][w]['return_pct']-
                cells[2][w]['return_pct']+cells[0][w]['return_pct'] for w in WINDOWS}})
    dispositions = {(r['candidate_id'],r['window']):r['status'] for r in native.results(root)}
    rejected = []
    for key in plan.get('reviewed_capacity_rejections', {}):
        cid, window, account = key.split('/')
        if account == '100000000_baseline':
            row = reviewed_capacity_rejection(root, all_candidates[cid], window)
            rejected.append(row)
            dispositions[(cid,window)] = row['status']
    native.write(root/'execution_infeasible_trials.json', rejected)
    unavailable = []
    for key in plan.get('reviewed_source_unavailability', {}):
        cid, window, account = key.split('/')
        if account == '100000000_baseline':
            row = reviewed_source_unavailability(root, all_candidates[cid], window)
            unavailable.append(row)
            dispositions[(cid,window)] = row['status']
    native.write(root/'source_unverified_trials.json', unavailable)
    coverage = {cid:{w:dispositions.get((cid,w),'pending') for w in WINDOWS} for cid in plan['original_candidate_ids']}
    native.write(root/READOUT, {'updated_utc':utc(), 'paired_candidates':paired,
        'activation_contrasts':contrasts, 'factorials':factorials, 'strict_finalists':[r['id'] for r in leaders],
        'all_original_trial_coverage':coverage, 'registered_unique_candidates':len(all_candidates),
        'completed_paired_candidates':len(paired), 'invalid_configuration_count':sum(v=='invalid_configuration' for v in dispositions.values()),
        'unsupported_execution_contract_count':sum(v=='unsupported_execution_contract' for v in dispositions.values()),
        'execution_infeasible_count':len(rejected),
        'source_unverified_count':len(unavailable),
        'warning':plan['warning']})


def finish(root, finalists):
    native.finish_analytics(root, finalists)
    plan = native.read(root/'mechanism_plan.json')
    _, pairs = native.rank_pairs(root)
    base = pairs[plan['baseline_id']]
    support = {c['id']:{w:support_comparison(base[w],pairs[c['id']][w]) for w in WINDOWS} for c in finalists}
    native.write(root/'finalist_support.json', support)
    native.write(root/'finalist_trade_diagnostics.json', {c['id']:{w:native.concentration(pairs[c['id']][w]['output'])
        for w in WINDOWS} for c in finalists})
    native.write(root/'block_length_sensitivity.json', {c['id']:{w:{str(block):native.paired_uncertainty(
        base[w],pairs[c['id']][w],block_sessions=block) for block in (1,3,5,10)}
        for w in WINDOWS} for c in finalists})
    # Compare stress results with the matched baseline, never with its un-stressed result.
    robustness = native.read(root/'robustness_results.json')
    controls = {(r['window'],r['capital'],r['scenario'],r['capped']):r for r in robustness if r['candidate_id']==plan['baseline_id']}
    comparisons = []
    for row in robustness:
        b = controls[(row['window'],row['capital'],row['scenario'],row['capped'])]
        comparisons.append({**row, 'return_gain_pp':row['return_pct']-b['return_pct'],
            'trade_delta':row['trades']-b['trades'], 'drawdown_delta_pp':row['drawdown_pct']-b['drawdown_pct']})
    native.write(root/'paired_robustness.json', comparisons)
    opportunity = {}
    for candidate in finalists:
        opportunity[candidate['id']] = {}
        for window in WINDOWS:
            row = pairs[candidate['id']][window]
            registration = native.read(root/'registrations'/window/(candidate['id']+'_100000000')/'study.json')
            days = {r['day'] for r in native.read(Path(row['output'])/'daily.json')}
            snapshots = [r for r in native.read(Path(registration['code'])/registration['olr_snapshots']) if r['trade_date'] in days]
            counts, regimes = Counter(), Counter()
            rejects = defaultdict(Counter)
            for snap in snapshots:
                meta = snap.get('metadata') or {}
                regimes[str(meta.get('market',{}).get('regime'))] += 1
                counts['final_candidate_rows'] += len(snap['candidates'])
                counts['days_with_candidates'] += bool(snap['candidates'])
                for key in ('requested_universe_count','complete_universe_count','candidate_pool_count',
                            'afternoon_candidate_pool_count','selected_symbol_count'):
                    counts[key] += meta.get(key,0)
                for stage in ('build_rejected_symbols','rejected_symbols','afternoon_rejected_symbols'):
                    for reasons in (meta.get(stage) or {}).values():
                        rejects[stage].update(reasons if isinstance(reasons,list) else [reasons])
            trades = native.read(Path(row['output'])/'trades.json')
            notional = sum(t['qty']*t['entry_price'] for t in trades)
            opportunity[candidate['id']][window] = {'entry_snapshot_sessions':len(snapshots),
                'counts':dict(counts), 'regime_entry_sessions':dict(regimes),
                'rejections_overlapping':{k:dict(v) for k,v in rejects.items()},
                'executed_trades':len(trades), 'active_entry_days':row['active_days'],
                'entry_turnover':notional/row['capital'],
                'net_pnl_per_entry_notional':sum(t['net_pnl'] for t in trades)/notional if notional else None,
                'explicit_costs':sum(t['commission'] for t in trades),
                'warning':'Turnover-adjusted realized PnL is descriptive, not a forecast. Rejection reasons overlap.'}
    native.write(root/'finalist_opportunity_funnels.json', opportunity)


def run(root, workers):
    if (root/'superseded_mechanism.json').exists():
        raise ValueError('This mechanism window is superseded; use the corrected root in superseded_mechanism.json')
    plan = native.read(root/'mechanism_plan.json')
    if native.sha(__file__) != plan['runner_sha256'] or native.sha(native.__file__) != plan['native_evaluator_sha256']:
        raise ValueError('Registered runner bytes changed; version the repair')
    for name, digest in plan['source_evidence'].items():
        if native.sha(root/name) != digest:
            raise ValueError('Registered source evidence changed: '+name)
    known = catalog(root)
    baseline = known[plan['baseline_id']]
    for window in WINDOWS:
        native.prepare_window(root, window)
        row = evaluate(root, baseline, window)
        if window == 'train' and (abs(row['return_pct']-55.0465907665475)>1e-7 or row['trades']!=176):
            raise ValueError('Selected IS control is not reproduced')
        native.reconstruction_bridge(root, window, baseline)
    if native.read(root/'protocol.json').get('refresh_baseline_mechanism_diagnostics'):
        from scripts import olr_round1_mechanism_diagnostics as diagnostics
        diagnostics.main(root)
    readout(root)
    for phase in plan['phases']:
        candidates = (adaptive_plan(root, phase['name']) if phase.get('adaptive') else [known[cid] for cid in phase['ids']])
        native.event(root, phase['name']+'_started', candidates=len(candidates), revision=plan['version'])
        batch(root, candidates, workers, phase['name'])
        readout(root)
    leaders, _ = native.rank_pairs(root)
    known = catalog(root)
    finalists = [baseline, *[known[r['id']] for r in leaders[:5]]]
    native.event(root, 'paired_robustness_started', finalists=len(finalists))
    robustness_jobs = []
    for candidate in finalists:
        for window in WINDOWS:
            for capital in (10_000_000, 25_000_000, 50_000_000, 100_000_000):
                for scenario in ('baseline', 'combined'):
                    robustness_jobs.append(dict(candidate=candidate,window=window,capital=capital,scenario=scenario))
            for scenario in ('cost', 'participation'):
                robustness_jobs.append(dict(candidate=candidate,window=window,capital=100_000_000,scenario=scenario))
            robustness_jobs.append(dict(candidate=candidate,window=window,capital=100_000_000,scenario='baseline',capped=True))
    robustness_indices = Counter()
    for job in robustness_jobs:
        robustness_indices[job['window']] += 1
        job['index'] = robustness_indices[job['window']]
    run_evaluation_jobs(root, robustness_jobs, workers, 'paired_robustness')
    readout(root)
    finish(root, finalists)
    native.write(root/'completion.json', {'completed_utc':utc(), 'revision':plan['version'],
        'registered_candidates':len(known), 'finalists':[c['id'] for c in finalists],
        'all_original_candidates_retained':True, 'research_only':True,
        'independent_performance_established':False, 'requires_final_interpretation':True})
    native.event(root, 'mechanism_study_computed', finalists=len(finalists), requires_final_interpretation=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register','run','report'])
    parser.add_argument('--root',type=Path,default=native.DEFAULT_ROOT)
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    if args.command == 'register':
        register(args.root)
    elif args.command == 'report':
        readout(args.root)
    else:
        with process_lock(args.root):
            native.write(args.root/'process.json', {'pid':os.getpid(), 'started_utc':utc(),
                'script':str(Path(__file__).resolve()), 'revision':native.read(args.root/'mechanism_plan.json')['version'],
                'timeout_seconds_per_process':172800, 'overall_timeout':None})
            try:
                run(args.root,args.workers)
            except Exception as exc:
                native.event(args.root, 'mechanism_needs_repair', error=str(exc), traceback=traceback.format_exc())
                raise


if __name__ == '__main__':
    main()

"""Prepared independent support census; requires completed V33 review and exit."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
BASE = Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
SOURCE = BASE/'v33_checkpoint_contrast_labels'
HELPER = REPO/'backtests/strategies/kalcb/checkpoint_contrast_support_v1.py'
TEST = REPO/'tests/backtests/strategies/test_kalcb_checkpoint_contrast_support_v1.py'
CALENDAR = BASE.parent/'kalcb_aligned_walk_forward_20261010/v3/fit/calendar.json'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def bound(p):
    with Path(p).open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'path': str(Path(p).resolve()), 'sha256': sha}


def verify(items):
    for item in items:
        if bound(item['path'])['sha256'] != item['sha256']:
            raise ValueError('Changed registered source: ' + item['path'])


def write(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def lines(p):
    with gzip.open(p, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def origins():
    rows = []
    for name, module in tuple(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if file and Path(file).resolve().is_relative_to(REPO):
            if Path(file).resolve() not in (Path(__file__).resolve(), HELPER.resolve()):
                raise ValueError('Unexpected application import: ' + name)
            rows.append({'module': name, **bound(file)})
    return rows


def closed_review():
    import psutil
    receipt = read(SOURCE/'independent_review_launch_v1/launch_receipt.json')
    try:
        process = psutil.Process(receipt['pid'])
        if process.cmdline() == receipt['command'] and abs(process.create_time()-receipt['process_created']) < .001:
            raise ValueError('Independent review still active')
    except psutil.NoSuchProcess:
        pass
    done = read(SOURCE/'review_v1/completion.json')
    verify([done['review']])
    if not done['complete'] or not read(done['review']['path'])['passed']:
        raise ValueError('Unpassed independent review')


def register(root, tests):
    closed_review()
    test = read(tests)
    if test['exit_code'] != 0:
        raise ValueError('Tests failed')
    verify(test['programs'] + [test['log']])
    p30 = read(BASE/'v30_checkpoint_information/protocol.json')
    cases = [c for c in p30['cases'] if c['key'].split('/')[-1] in ('baseline', 'combined')]
    expected = {str(Path(b['path']).resolve()).lower(): b for b in p30['bindings']}
    bindings = [bound(SOURCE/n) for n in ('protocol.json', 'completion.json', 'labels/completion.json',
                 'review_v1/protocol.json', 'review_v1/completion.json', 'review_v1/review.json')]
    bindings += [bound(CALENDAR), bound(tests)]
    source_protocol = read(SOURCE/'protocol.json')
    bindings += [source_protocol['plan']]
    source_plan = read(source_protocol['plan']['path'])
    bindings += [{'path': source_plan['config'], 'sha256': source_plan['input_sha256'][source_plan['config']]}]
    bindings += [bound(BASE/'v30_checkpoint_information/protocol.json'),
                 bound(BASE/'v30_checkpoint_information/assessment.json')]
    for folder in ('v31_checkpoint_context_support', 'v26_opening_decision_audit_v2'):
        done = read(BASE/folder/'completion.json')
        verify(done['artifacts'])
        bindings += [bound(BASE/folder/'completion.json')] + done['artifacts']
    p31 = read(BASE/'v31_checkpoint_context_support/protocol.json')
    bindings += [bound(BASE/'v31_checkpoint_context_support/protocol.json')]
    for case in cases:
        for name in ('fills.json', 'orders.jsonl.gz'):
            path = Path(case['folder'])/name
            original = expected.get(str(path.resolve()).lower())
            if original is None:
                raise ValueError('Missing historical native source binding: '+str(path))
            bindings.append(original)
        account = BASE/'v30_checkpoint_information/accounts'/Path(case['key']).with_suffix('.json')
        previous = next(x for x in read(BASE/'v30_checkpoint_information/assessment.json')['artifacts']
                        if Path(x['path']).resolve() == account.resolve())
        bindings += [previous, bound(case['assessment'])]
        recipe, period, _ = case['key'].split('/')
        profile_path = BASE/f'v25_pending_funding_screen/cases/{recipe}/{period}/input_profile.json'
        profile = read(profile_path)
        bindings += [bound(profile_path), profile['plan']]
        plan = read(profile['plan']['path'])
        bindings += [{'path': plan['config'], 'sha256': plan['input_sha256'][plan['config']]}]
    bindings = list({x['path']: x for x in bindings}.values())
    verify(bindings)
    root.mkdir(exist_ok=False)
    write(root/'protocol.json', {
        'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'read_only': True,
        'programs': [bound(__file__), bound(HELPER), bound(TEST)], 'bindings': bindings,
        'tests': bound(tests), 'cases': cases, 'source': str(SOURCE),
        'context_features': p31['context_features'], 'own_features': p31['own_features'],
        'population': 'Every V33 pair, all original 286 sessions, both scenarios and three dependent quantities. Eight unchanged native prior100/125 baseline/combined accounts. No outcome filtering or payoff-based policy selection.',
        'chronology': 'Exact original20session windows, 60 unique mature sessions, one-session embargo. Paired target matures only after BOTH arms and BOTH scenarios; count issuer-days/sessions once. Unknown pre-IS parent membership stays unknown.',
        'comparability': 'Reconstruct native held/planned/pending entry shares before completed45 using unique lifetime fills and actual submit/cancel/reject journals. Compare selected reference dose and all existing dose envelopes. Equal quantities do not establish full state/portfolio exchangeability. No post-hoc scaling of PnL.',
        'risk': 'Payoff availability, HOLD deadline observations and additional-liquidation physical/deadline validity are separate. Inactive/no-position aliases do not count as new action failures or support. Invalid/unresolved pairs never acquire payoff labels.',
        'no_fit': True, 'native_cases': 0, 'policy_changes': 0, 'study_complete': False})


def run(root):
    import psutil
    closed_review()
    p = read(root/'protocol.json')
    verify(p['programs'] + p['bindings'])
    process = psutil.Process()
    write(root/'supervisor_start.json', {'pid': process.pid, 'process_created': process.create_time(),
          'command': process.cmdline(), 'at': datetime.now(timezone.utc).isoformat()})
    spec = importlib.util.spec_from_file_location('registered_contrast_support', HELPER)
    h = importlib.util.module_from_spec(spec); sys.modules[spec.name] = h; spec.loader.exec_module(h)
    before = origins()
    contexts = {(r['day'], r['symbol']): r for r in lines(BASE/'v31_checkpoint_context_support/contexts.jsonl.gz')}
    selected = defaultdict(set); actual = {}
    for row in lines(BASE/'v26_opening_decision_audit_v2/decisions.jsonl.gz'):
        for recipe, admission in row['selected'].items():
            if admission.get('reason') == 'admitted':
                selected[recipe].add(row['action_id'])
        for key, positions in row['actual'].items():
            for position in positions:
                k = (key, position['position_id'])
                if k in actual:
                    raise ValueError('Duplicate actual lifetime / action join')
                actual[k] = {'action_id': row['action_id'], 'quantity_scale': row['quantity_scale']}
    labels = {}; groups = defaultdict(dict); by_day_symbol = defaultdict(list); counts = Counter(); risk = []
    done = read(SOURCE/'labels/completion.json')
    for item in done['artifacts']:
        if '.events.' in item['path']:
            continue
        verify([item])
        for raw_value in lines(item['path']):
            h.validate_source_context(raw_value, contexts[raw_value['day'], raw_value['symbol']])
            v = {k: raw_value[k] for k in ('opportunity_id', 'day', 'symbol', 'scenario', 'quantity_scale',
                 'planned_quantity', 'paired_label_valid', 'label_available_at', 'eligibility')}
            for arm in ('hold', 'liquidate'):
                v[arm] = {k: raw_value[arm][k] for k in ('independent_execution_pass', 'independent_account_pass',
                          'deadline_violation', 'checkpoint', 'actual_changed_exit_quantity', 'unresolved', 'physical_fill_failures')}
            key = (v['opportunity_id'], v['scenario'])
            if key in labels:
                raise ValueError('Duplicate paired label')
            labels[key] = v; groups[v['opportunity_id']][v['scenario']] = v
            for arm in ('hold', 'liquidate'):
                d = v[arm]; active = d['checkpoint']['action_submitted']
                for flag, failure in [('execution', not d['independent_execution_pass']),
                                      ('account', not d['independent_account_pass']), ('deadline', d['deadline_violation'])]:
                    counts[arm+'_'+flag+'_failures'] += bool(failure)
                    if arm == 'liquidate':
                        counts['active_liquidation_'+flag+'_failures'] += bool(active and failure)
                if not d['independent_account_pass'] or not d['independent_execution_pass'] or d['deadline_violation']:
                    risk.append({'action_id': v['opportunity_id'], 'scenario': v['scenario'], 'arm': arm,
                                 'active': active, 'eligible': v['eligibility'], 'unresolved': d['unresolved'],
                                 'physical_failures': d['physical_fill_failures'], 'deadline': d['deadline_violation'],
                                 'account_pass': d['independent_account_pass'], 'paired_payoff_valid': v['paired_label_valid']})
    if len(labels) != 38508 or any(set(g) != {'baseline', 'combined'} for g in groups.values()):
        raise ValueError('Incomplete finite paired population')
    joint = [h.joint_row(g['baseline'], g['combined']) for g in groups.values()]
    for group in groups.values():
        for scenario, value in group.items():
            by_day_symbol[value['day'], value['symbol'], scenario].append(value)
    calendar = read(CALENDAR)
    chronology = {'original_population': h.mature_support(joint, calendar['sessions'], calendar['windows'])}
    for recipe in ('control', 'control_prior125'):
        chronology[recipe+'_known_selected_only'] = h.mature_support(joint, calendar['sessions'], calendar['windows'], selected[recipe])
    write(root/'chronology.json', chronology)
    write(root/'risk_accounting.json', {'counts': dict(counts), 'rows': risk,
        'interpretation': 'Retained dependent observations, not physical-risk certification or funded portfolio results.'})
    comparisons = []; assessments = {}; wanted = defaultdict(set)
    for case in p['cases']:
        recipe, period, scenario = case['key'].split('/')
        folder = Path(case['folder']); fills = read(folder/'fills.json')
        submissions = {}; terminals = defaultdict(list)
        for event in lines(folder/'orders.jsonl.gz'):
            order = event['order']; oid = order['order_id']
            if event['event'] == 'submit':
                if oid in submissions:
                    raise ValueError('Duplicate native submission')
                submissions[oid] = order
            elif event['event'] in ('cancel', 'reject'):
                terminals[oid].append(event['timestamp'])
            elif event['event'] == 'amend' and event['reason'] != 'partial_remainder':
                raise ValueError('Unimplemented native amendment clock')
        if case['assessment'] not in assessments:
            assessments[case['assessment']] = read(case['assessment'])
        positions = assessments[case['assessment']]['accounts'][case['assessment_key']]['positions']
        byid = {r['position_id']: r for r in positions}
        # Position IDs already authenticated as unique physical lifetimes. Restrict
        # fills to the lifetime interval before any quantity/state comparison.
        observations = read(BASE/'v30_checkpoint_information/accounts'/Path(case['key']).with_suffix('.json'))['observations']
        for obs in observations:
            pid = obs['position_id']; pos = byid[pid]; context = contexts.get((obs['day'], obs['symbol']))
            row = {'case': case['key'], 'position_id': pid, 'day': obs['day'], 'symbol': obs['symbol'],
                   'parent_exit_already_decided': obs['parent_exit_already_decided'],
                   'context_supported': bool(context and context['context_supported'])}
            match = actual.get((recipe+'/'+scenario, pid))
            if not match:
                row['unmatched_selected_action'] = True; comparisons.append(row); continue
            clock = context['available_at'] if context else obs['features']['available_at']
            physical = [f for f in fills if f['symbol'] == obs['symbol']
                        and h.stamp(pos['entry_fill_time']) <= h.stamp(f['timestamp']) <= h.stamp(pos['exit_fill_time'])]
            prefix = h.native_prefix(physical, submissions, terminals, pos['entry_order_ids'], obs['symbol'], clock)
            if prefix['held_quantity'] != obs['held_qty']:
                raise ValueError('Native held quantity does not match independently audited census')
            reference = labels.get((match['action_id'], scenario))
            row.update(match, native=prefix, reference_present=reference is not None)
            if reference:
                row['reference'] = {'held_quantity': reference['hold']['checkpoint']['held_quantity'],
                                    'planned_quantity': reference['planned_quantity'],
                                    'eligibility': reference['eligibility'], 'paired_payoff_valid': reference['paired_label_valid']}
                wanted[(obs['day'], scenario)].add(match['action_id'])
                same_day = by_day_symbol[obs['day'], obs['symbol'], scenario]
                holds = [x['hold']['checkpoint']['held_quantity'] for x in same_day if x['eligibility'] == 'eligible']
                row['within_current_day_dose_held_envelope'] = bool(holds and min(holds) <= prefix['held_quantity'] <= max(holds))
            comparisons.append(row)
        print(json.dumps({'case': case['key'], 'compared_lifetimes': len(observations)}), flush=True)
    pending_reference = {}
    event_bindings = {Path(item['path']).name: item for item in done['artifacts'] if '.events.' in item['path']}
    for (day, scenario), ids in sorted(wanted.items()):
        tokens = ['"opportunity_id":"'+key+'"' for key in ids]
        path = SOURCE/'labels'/f'{day}_{scenario}.events.jsonl.gz'
        verify([event_bindings[path.name]])
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                if not any(token in line for token in tokens):
                    continue
                event = json.loads(line); key = event['pair']['opportunity_id']
                if key in ids:
                    state = event['hold']['checkpoint']['state']
                    pending_reference[key, scenario] = sum(o['qty'] for o in state['orders'] if o['side'] == 'BUY')
    for row in comparisons:
        if row.get('reference_present'):
            row['reference']['pending_buy_quantity'] = pending_reference[row['action_id'], row['case'].split('/')[-1]]
            row['state_comparison'] = h.state_comparison(row['native'], row['reference'])
    summaries = {}
    for case in p['cases']:
        rows = [r for r in comparisons if r['case'] == case['key']]
        active = [r for r in rows if not r['parent_exit_already_decided']]
        match = [r for r in active if r.get('reference_present')]
        summaries[case['key']] = {'native_observations': len(rows), 'without_parent_exit': len(active),
             'context_supported': sum(r['context_supported'] for r in active), 'reference_matched': len(match),
             'exact_held_planned_pending': sum(all(r['state_comparison'].values()) for r in match),
             'inside_same_day_held_dose_envelope': sum(r['within_current_day_dose_held_envelope'] for r in match),
             'native_pending_entry_at_checkpoint': sum(r['native']['pending_buy_quantity'] > 0 for r in match),
             'pending_presence_disagrees': sum((r['native']['pending_buy_quantity']>0) != (r['reference']['pending_buy_quantity']>0) for r in match)}
    write(root/'selected_parent_comparability.json', {'summaries': summaries, 'rows': comparisons,
          'limits': 'Current-day dose envelope is a descriptive same-source mechanics comparison, not a prior-only training domain or exchangeability proof. Quantities/scenarios/parents are dependent. Full core/account state remains different; no proportional PnL or funded profit claim.'})
    source_protocol = read(SOURCE/'protocol.json')
    source_plan = read(source_protocol['plan']['path'])
    source_config = read(source_plan['config'])
    management = {}
    prefixes = ('exit_', 'failure_stop_', 'partial_', 'profit_', 'time_stop_')
    for recipe in ('control', 'control_prior125'):
        for period in ('is', 'oos'):
            profile = read(BASE/f'v25_pending_funding_screen/cases/{recipe}/{period}/input_profile.json')
            cfg = read(read(profile['plan']['path'])['config'])
            keys = sorted(k for k in set(cfg) | set(source_config) if k.startswith(prefixes))
            management[recipe+'/'+period] = {'compared_fields': len(keys), 'differences': [
                {'field': k, 'isolated': source_config.get(k), 'native': cfg.get(k)}
                for k in keys if source_config.get(k) != cfg.get(k)]}
    write(root/'management_configuration.json', {'cases': management,
        'limits': 'Equality of management settings is necessary, not equality of selected/funded native state or paths.'})
    verify(p['programs'] + p['bindings'])
    write(root/'completion.json', {'complete': True, 'at': datetime.now(timezone.utc).isoformat(),
          'programs': p['programs'], 'bindings': p['bindings'], 'origins_before': before, 'origins_after': origins(),
          'artifacts': [bound(root/n) for n in ('chronology.json', 'risk_accounting.json', 'selected_parent_comparability.json', 'management_configuration.json')],
          'study_complete': False, 'models': 0, 'native_cases': 0, 'exchangeability_proven': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'run'))
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--tests', type=Path)
    args = parser.parse_args()
    if args.mode == 'register':
        register(args.root, args.tests)
    else:
        try:
            run(args.root)
        except Exception as error:
            write(args.root/'failure.json', {'at': datetime.now(timezone.utc).isoformat(), 'error': repr(error), 'study_complete': False})
            raise

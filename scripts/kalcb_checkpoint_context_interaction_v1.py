"""Register two fixed context-slope interactions against an authenticated additive reference."""
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
SOURCE = BASE/'v36_checkpoint_holding_targets'
HELPER = REPO/'backtests/strategies/kalcb/checkpoint_context_interaction_v1.py'
INHERITED = BASE/'v37_checkpoint_holding_models'
TEST = REPO/'tests/backtests/strategies/test_kalcb_checkpoint_context_interaction_v1.py'
CALENDAR = BASE.parent/'kalcb_aligned_walk_forward_20261010/v3/fit/calendar.json'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def bound(p):
    with Path(p).open('rb') as f: digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'path': str(Path(p).resolve()), 'sha256': digest}


def verify(items):
    for item in items:
        if bound(item['path'])['sha256'] != item['sha256']:
            raise ValueError('Changed registered source '+item['path'])


def write(p, value):
    with Path(p).open('x', encoding='utf-8') as f: json.dump(value, f, indent=2, allow_nan=False)


def lines(p):
    with gzip.open(p, 'rt', encoding='utf-8') as f:
        yield from map(json.loads, f)


def load():
    spec = importlib.util.spec_from_file_location('registered_holding_models', HELPER)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def origins():
    result = []
    for name, m in tuple(sys.modules.items()):
        f = getattr(m, '__file__', None)
        if f and Path(f).resolve().is_relative_to(REPO):
            if Path(f).resolve() not in (Path(__file__).resolve(), HELPER.resolve()):
                raise ValueError('Unregistered application import '+name)
            result.append({'module': name, **bound(f)})
    return result


def closed():
    import psutil
    receipt = read(INHERITED/'launch_receipt.json')
    try:
        p = psutil.Process(receipt['pid'])
        if p.cmdline() == receipt['command'] and abs(p.create_time()-receipt['process_created']) < .001:
            raise ValueError('Target extraction is still running')
    except psutil.NoSuchProcess:
        pass
    if not read(INHERITED/'completion.json')['complete'] or not read(INHERITED/'review_v1/post_completion_health_v1.json')['passed']:
        raise ValueError('Inherited additive models/review not authenticated')


def register(root, tests):
    closed(); receipt = read(tests)
    if receipt['exit_code'] != 0: raise ValueError('Failed preflight')
    verify(receipt['programs']+[receipt['log'], receipt['historical_preflight']])
    historical = read(receipt['historical_preflight']['path'])
    if not historical['passed'] or historical['fits'] != 0:
        raise ValueError('Historical causal feature preflight failed')
    verify(historical['source_bindings'])
    source = read(SOURCE/'completion.json')
    paths = [SOURCE/'protocol.json', SOURCE/'completion.json', SOURCE/'post_completion_health_v1.json',
             BASE/'v33_checkpoint_contrast_labels/labels/completion.json',
             BASE/'v34_checkpoint_contrast_support/selected_parent_comparability.json',
             BASE/'v34_checkpoint_contrast_support/completion.json',
             BASE/'v34_checkpoint_contrast_support/post_completion_health_v1.json',
             BASE/'v31_checkpoint_context_support/contexts.jsonl.gz', CALENDAR, tests]
    paths += [INHERITED/'protocol.json',INHERITED/'completion.json',INHERITED/'chronology.json',INHERITED/'review_v1/review.json',INHERITED/'review_v1/post_completion_health_v1.json',INHERITED/'economic_assessment_v1/economic_decision.json']
    bindings = source['artifacts'] + [bound(p) for p in paths] + [receipt['historical_preflight']]
    verify(bindings)
    h = load(); root.mkdir(exist_ok=False)
    write(root/'protocol.json', {
        'at': datetime.now(timezone.utc).isoformat(), 'programs': [bound(__file__),bound(HELPER),bound(TEST)],
        'bindings': bindings, 'tests': bound(tests), 'recipes': list(h.RECIPES),
        'own_state_features': list(h.OWN+h.STATE+h.CONTEXT), 'incremental_context_features': list(h.INTERACTIONS),
        'interaction_formulas': {'own_return_x_market_breadth_change':'post_entry_return_atr * peer_market_vwap_breadth_change', 'own_return_x_sector_breadth_change':'post_entry_return_atr * peer_sector_vwap_breadth_change'},
        'target': 'V36 full-cost current-holding LIQUIDATE minus HOLD bps per actual held notional. Original conservative whole-episode maturity. No descendant-credit fitting or PnL scaling.',
        'population': 'Both scenarios causal eligible held/no-pending original population, independently valid mature paired targets only for payoff. All failures retained separately for action risk. Every source row retained in input; unsupported current rows abstain.',
        'estimator': 'Authenticate/reuse V37 additive15-feature coefficients. Fit ONLY two new interaction slopes alongside the same15inputs, Ridge120 all slopes/unpenalized intercept, identical training and weights. Exactly24fresh scenario/fold fits; no feature, penalty, threshold, quantity or clock search. Both models use SAME17-feature prior-only domain and separately refit selected forward calibration on that common domain. Interactions test context modifying own-return slope, not another additive context term.',
        'dependence': 'One total weight per issuer-day split over existing quantities, then equal session weight, normalized to issuer-day count. Quantities, scenarios, parents are dependent; no summing isolated PnL.',
        'chronology': 'Original calendar20-session windows, one-session embargo,60 unique mature sessions. Same-quantity support20issuer-days/8sessions. Unknown preIS selected membership stays unknown.',
        'risk': 'Active submitted LIQUIDATE no-pending exposures only. Pair scenarios and all existing quantities into one issuer-day; unsafe if any such action has execution/account/deadline/unresolved/quantity-close failure. Use inherited zero-failure1/(N+100) branch,60mature sessions,strictly<.001. If nonzero failures, abstain pending separately registered conditional risk work. No payoff-validity conditioning of risk. Prior feature envelope is necessary, not exchangeability or true-risk certification.',
        'native_features': 'V34 actual held/planned/pending integer state and actual mean entry fill price; raw V31 completed45/context/volume/ATR. Preserve parent-complete-exit blockers. No isolated reference quantities silently substituted.',
        'calibration': 'Per parent,recipe,scenario identity-prior100 affine slope[0,1],20distinctissuer-days/8sessions. Only earlier genuinely forward, raw-domain/risk supported observations with mature exact same predecision integer state and feature match to isolated labels in BOTH scenarios. Never future-HOLD matching. Current unmatched states may receive forecasts but cannot supply calibration/evaluation labels.',
        'economic_diagnostic': 'BOTH calibrated full-cost current-holding advantages strictly>0. Zero is the action-advantage break-even, not an entry reserve. No policy/admission threshold search or portfolio orders. Native physical/capacity/shared-volume/account validation remains pending.',
        'next_gate': 'Require incremental chronological economic discrimination, at least20ISpositive-both issuer-days/8sessions, known-selected calibration support, both-scenario benefit and ISwinner preservation before any separately frozen native protocol. If this fixed interaction contrast fails, close this information branch; do not repeat more complex learner/feature/penalty/threshold sweeps. No automatic replay.',
        'reporting': 'Keep original-population and actual selected-parent evidence separate. Compare raw own/context on identical support; calibrated diagnostics never imply funded returns or DD. Retain no-support reasons, unknown labels, all months and conditional paired sensitivity for any credible finalist.',
        'runtime_estimate': 'Approximately3-8minutes including hashes and finite fits; no native replay.',
        'native_cases': 0, 'policies': 0, 'study_complete': False})


def risk_records(h, groups, artifacts):
    raw = {}; consumed = []
    for item in artifacts:
        if '.events.' in item['path']: continue
        verify([item]); consumed.append(item)
        for v in lines(item['path']):
            arm = v['liquidate']; k = (v['opportunity_id'], v['scenario'])
            if k in raw: raise ValueError('Duplicate risk action')
            raw[k] = {'at': arm['risk_available_at'],
                      'active': arm['checkpoint']['action_submitted'],
                      'unsafe': not arm['independent_execution_pass'] or not arm['independent_account_pass'] or
                      arm['deadline_violation'] or arm['unresolved'] or
                      arm['actual_changed_exit_quantity'] != v['hold']['checkpoint']['held_quantity']}
    result = []
    for g in groups:
        row = g['baseline']; pair = [raw[g[s]['opportunity_id'],s] for s in h.SCENARIOS]
        result.append({'day': row['day'], 'symbol': row['symbol'], 'opportunity_id': row['opportunity_id'],
                       'active_no_pending_both': all(h.causal_eligible(g[s]) and raw[g[s]['opportunity_id'],s]['active'] for s in h.SCENARIOS),
                       'risk_available_at': max(r['at'] for r in pair) if all(r['at'] for r in pair) else None,
                       'unsafe': any(r['unsafe'] for r in pair)})
    if len(raw) != 38508: raise ValueError('Incomplete physical population')
    return result, consumed


def native_pairs(h, labels):
    contexts = {(r['day'],r['symbol']): r for r in lines(BASE/'v31_checkpoint_context_support/contexts.jsonl.gz')}
    values = defaultdict(dict); metadata = defaultdict(dict); rejected = Counter()
    for row in read(BASE/'v34_checkpoint_contrast_support/selected_parent_comparability.json')['rows']:
        if not row.get('native') or not row.get('action_id') or not row['context_supported']:
            rejected[row['case']+'/missing_source_or_action'] += 1; continue
        prior, _, scenario = row['case'].split('/'); key = (prior,row['day'],row['symbol'])
        if scenario in values[key]: raise ValueError('Duplicate actual checkpoint lifetime')
        native = h.native_row(row, contexts[row['day'],row['symbol']]); values[key][scenario] = native
        ref = labels.get((row['action_id'], scenario))
        exact = bool(ref and h.exact_causal_match(native,ref))
        metadata[key][scenario] = {'position_id': row['position_id'], 'exact_predecision': exact,
            'label': {'valid': bool(exact and ref['current_holding_label_valid']),
                      'available_at': ref['label_available_at'] if exact else None,
                      'value': ref['current_holding_advantage_bps'] if exact else None,
                      'episode_value': ref['original_episode_advantage_bps'] if exact else None,
                      'held_notional_krw': native['held_notional_krw']}}
    return values, metadata, dict(rejected)


def run(root):
    import psutil
    from threadpoolctl import threadpool_limits
    closed(); protocol = read(root/'protocol.json'); verify(protocol['programs']+protocol['bindings'])
    proc = psutil.Process(); write(root/'supervisor_start.json', {'pid':proc.pid,'process_created':proc.create_time(),
        'command':proc.cmdline(),'at':datetime.now(timezone.utc).isoformat()})
    h = load(); before = origins(); grouped = defaultdict(dict); labels = {}
    for row in lines(SOURCE/'targets.jsonl.gz'):
        key = (row['opportunity_id'],row['scenario'])
        if key in labels: raise ValueError('Duplicate target')
        labels[key] = row; grouped[key[0]][key[1]] = row
    if len(labels) != 38508 or any(set(g)!=set(h.SCENARIOS) for g in grouped.values()):
        raise ValueError('Incomplete target population')
    groups = [grouped[k] for k in sorted(grouped)]
    risks, consumed = risk_records(h, groups, read(BASE/'v33_checkpoint_contrast_labels/labels/completion.json')['artifacts'])
    actual, metadata, missing = native_pairs(h,labels); calendar = read(CALENDAR)
    inherited = {r['payoff']['valid_from']:r['payoff'] for r in read(INHERITED/'chronology.json')}
    history = defaultdict(list); proofs = []; native_output = []; original_output = []
    with threadpool_limits(limits=1):
        for first, through in calendar['windows']:
            proof = h.fit(groups,calendar['sessions'],first,through,inherited[first])
            risk = h.risk_proof(risks,calendar['sessions'],first)
            maps = {prior: {recipe: {s: h.calibration(history[prior,recipe],calendar['sessions'],first,s)
                     for s in h.SCENARIOS} for recipe in h.RECIPES} for prior in ('control','control_prior125')}
            proofs.append({'payoff':proof,'risk':risk,'calibrations':maps})
            current = [g for g in groups if first <= g['baseline']['day'] <= through]
            for pair in current:
                row = pair['baseline']
                original_output.append({'opportunity_id':row['opportunity_id'],'day':row['day'],'symbol':row['symbol'],
                    'valid_from':first,'forecasts':{recipe:h.forecast(pair,proof,recipe) for recipe in h.RECIPES},
                    'risk_pass':h.risk_pass(risk),
                    'labels':{s:{'valid':pair[s]['current_holding_label_valid'],'value':pair[s]['current_holding_advantage_bps']} for s in h.SCENARIOS}})
            for key in sorted(actual):
                prior,day,symbol = key
                if not first <= day <= through: continue
                pair = actual[key]; meta = metadata[key]
                exact = set(meta)==set(h.SCENARIOS) and all(x['exact_predecision'] for x in meta.values())
                for recipe in h.RECIPES:
                    raw = h.forecast(pair,proof,recipe); calibrated = h.calibrated(raw,maps[prior][recipe],risk)
                    record = {'prior':prior,'recipe':recipe,'day':day,'symbol':symbol,'valid_from':first,
                        'raw':raw,'calibrated':calibrated,'risk_pass':h.risk_pass(risk),
                        'matched_predecision_both':exact,'labels':{s:v['label'] for s,v in meta.items()},
                        'position_ids':{s:v['position_id'] for s,v in meta.items()},
                        'causal_states':pair,'policy_admission':False}
                    native_output.append(record); history[prior,recipe].append(record)
            print(json.dumps({'window':first,'mature_sessions':proof['mature_sessions'],
                'risk_unique_issuer_days':risk['issuer_days'],'native_forecasts':len(native_output)}),flush=True)
    for name, rows in [('original_population_forecasts.jsonl.gz',original_output),('native_forecasts.jsonl.gz',native_output)]:
        with gzip.open(root/name,'xt',encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    write(root/'chronology.json',proofs); write(root/'input_summary.json',{'targets':len(labels),'risk_action_pairs':len(risks),
         'native_issuer_day_parent_pairs':len(actual),'missing_native_source':missing,'original_forecast_pairs':len(original_output),
         'native_forecast_records':len(native_output),'native_cases':0,'policy_changes':0,'study_complete':False})
    verify(protocol['programs']+protocol['bindings']+consumed)
    write(root/'completion.json',{'complete':True,'at':datetime.now(timezone.utc).isoformat(),
        'programs':protocol['programs'],'bindings':protocol['bindings'],'consumed':consumed,
        'origins_before':before,'origins_after':origins(),
        'artifacts':[bound(root/n) for n in ('original_population_forecasts.jsonl.gz','native_forecasts.jsonl.gz','chronology.json','input_summary.json')],
        'native_cases':0,'policies':0,'study_complete':False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('mode',choices=('register','run'))
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--tests',type=Path)
    args = parser.parse_args()
    if args.mode == 'register': register(args.root,args.tests)
    else:
        try: run(args.root)
        except Exception as error:
            write(args.root/'failure.json',{'at':datetime.now(timezone.utc).isoformat(),'error':repr(error),'study_complete':False})
            raise

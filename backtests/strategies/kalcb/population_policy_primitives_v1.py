"""Extracted deterministic policy primitives; no runner imports or orchestration.
Historical algorithms are preserved and bridged against frozen parent artifacts.
"""

from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timedelta
import math
import json
from pathlib import Path
from backtests.strategies.kalcb import executed_value as ev, selective_adaptation_v1 as sa
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, digest, bound, link, PERIODS, POLICY
from backtests.strategies.kalcb.oos_ablation import transform_snapshot

def rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def jsonl(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str, allow_nan=False)+"\n")

# Extracted from scripts/kalcb_action_quantity_labels.py
def scaled_row(row, scale):
    if scale not in (0.25, 0.5, 1.0):
        raise ValueError('Undeclared quantity')
    candidate = {**row['candidate'], 'metadata': {**row['candidate']['metadata'], 'action_quantity_scale': scale}}
    features = dict(row['features'])
    old_qty = row['action']['quantity']
    ratio = int(old_qty * scale) / old_qty if old_qty else 0
    for key in ('action_notional_equity', 'action_quantity_last_volume', 'action_quantity_opening_volume', 'action_low_volume_exit_slices'):
        features[key] *= ratio
    features['action_quantity_scale'] = scale
    return {**row, 'candidate': candidate, 'features': features, 'action': {**row['action'], 'quantity': int(old_qty * scale)}, 'base_opportunity_id': row['opportunity_id'], 'opportunity_id': row['opportunity_id'] + f'/q{scale:g}'}

# Extracted from scripts/kalcb_action_quantity_ranker.py
def capacity_ok(row):
    """Conservative forecast from completed opening volume, not a fill guarantee."""
    from kis_core.trading_calendar import get_trading_calendar
    cutoff = datetime.fromisoformat(row['feature_available_at'])
    closing = get_trading_calendar().session_hours(cutoff.date())[1]
    deadline = datetime.combine(cutoff.date(), closing, cutoff.tzinfo) - timedelta(minutes=10)
    remaining = max(0, int((deadline - cutoff).total_seconds() // 300))
    volumes = sorted((b['volume'] for b in row['opening_bars']))
    low_volume = volumes[min(1, len(volumes) - 1)]
    capacity = math.floor(low_volume * 0.005) * 0.5
    return capacity > 0 and 0 < row['action']['quantity'] <= remaining * capacity

# Extracted from scripts/kalcb_action_quantity_ranker.py
def choose_quantities(items, baseline, stress, *, robust=False):
    choices = {}
    dispositions = {}
    for row in items:
        key = row['opportunity_id']
        base = row['base_opportunity_id']
        value = baseline[key]
        if value <= 0:
            dispositions[key] = 'nonpositive_expected_value'
            continue
        if robust and (stress[key] <= 0 or not capacity_ok(row)):
            dispositions[key] = 'stress_or_capacity'
            continue
        previous = choices.get(base)
        if previous is None or (value, -row['features']['action_quantity_scale']) > (previous[1], -previous[0]['features']['action_quantity_scale']):
            choices[base] = (row, value)
        dispositions[key] = 'feasible_action'
    return (choices, dispositions)

# Extracted from scripts/kalcb_action_value_study.py
def recheck_actions(day, selected, cfg, source):
    """Check actual selected ranks/context using only saved completed prefix bars."""
    from strategy_common.market import MarketBar
    from strategy_kalcb.core.preview import preview_entry
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    snapshot = ev.snapshot_for(day, selected, source=source)
    bars = [MarketBar(r['symbol'], datetime.fromisoformat(b['timestamp']), '5m', *[b[k] for k in ('open', 'high', 'low', 'close', 'volume')]) for r, _ in selected for b in r['opening_bars']]
    preview = preview_entry(bars, snapshot, cfg, KALCBPortfolioView(cash=ev.REFERENCE_EQUITY, equity=ev.REFERENCE_EQUITY), available_at=snapshot.generated_at)
    return {a['symbol'] for a in preview['intents']}

# Extracted from scripts/kalcb_action_value_study.py
def admit_actions(items, predicate, cfg, source, *, budget=8, threshold=0.0, ready=True, priority_keys=()):
    priority_keys = frozenset(priority_keys)
    ordered = sorted(items, key=lambda pair: (pair[0]['opportunity_id'] not in priority_keys, -pair[1], pair[0]['symbol']))
    selected, reasons = ([], {})
    for row, score in ordered:
        key = row['opportunity_id']
        if not math.isfinite(score):
            raise ValueError('Invalid forecast')
        if ready and (not row['action']['ready']):
            reasons[key] = 'opening_not_ready'
        elif not predicate(row, len(selected) + 1):
            reasons[key] = 'guard'
        elif threshold is not None and score <= threshold:
            reasons[key] = 'net_value_threshold'
        elif len(selected) >= budget:
            reasons[key] = 'admission_budget'
        else:
            proposed = selected + [(row, score)]
            if ready:
                ready_symbols = recheck_actions(row['trade_date'], proposed, cfg, source)
                if any((r['symbol'] not in ready_symbols for r, _ in proposed)):
                    reasons[key] = 'selected_rank_or_context_not_ready'
                    continue
            selected = proposed
            reasons[key] = 'admitted'
    return (selected, reasons)

# Extracted from scripts/kalcb_executed_value_study.py
def config_for(plan, scenario='baseline'):
    from strategy_kalcb.config import KALCBConfig
    raw = json.loads(Path(plan['config']).read_text())
    raw.update(slippage_bps=ev.SCENARIOS[scenario][1], fast_replay_suppress_rejections=False)
    return KALCBConfig.from_mapping(raw)

# Extracted from scripts/kalcb_checkpoint_fit.py
def update_snapshot(base, selected, source):
    snap = ev.snapshot_for(str(base.trade_date), selected, source=source)
    new = tuple((replace(c, metadata={**c.metadata, 'candidate_rank': len(base.candidates) + i, 'frontier_rank': len(base.candidates) + i}) for i, c in enumerate(snap.candidates, 1)))
    return replace(snap, candidates=base.candidates + new, metadata={**snap.metadata, 'active_symbols': [c.symbol for c in base.candidates + new], 'active_symbol_count': len(base.candidates + new), 'candidate_pool_count': len(base.candidates + new), 'selection_count': len(base.candidates + new), 'frontier_symbol_count': len(base.candidates + new)})

# Extracted from scripts/kalcb_checkpoint_fit.py
def native_admit(base, choices, cfg, source):
    from strategy_common.market import MarketBar
    from strategy_kalcb.core.preview import preview_entry
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    selected = []
    existing = set(base.by_symbol())
    for row, value in sorted(choices.values(), key=lambda x: (-x[1], x[0]['symbol'])):
        if row['symbol'] in existing:
            continue
        trial = selected + [(row, value)]
        snap = update_snapshot(base, trial, source)
        bars = [MarketBar(r['symbol'], datetime.fromisoformat(b['timestamp']), '5m', *[b[k] for k in ('open', 'high', 'low', 'close', 'volume')]) for r, _ in trial for b in r['opening_bars']]
        preview = preview_entry(bars, snap, cfg, KALCBPortfolioView(cash=ev.REFERENCE_EQUITY, equity=ev.REFERENCE_EQUITY), available_at=snap.generated_at)
        actual = {a['symbol'] for a in preview['intents']}
        if all((r['symbol'] in actual for r, _ in trial)):
            selected = trial
        if len(selected) >= 4:
            break
    return selected

# Extracted from scripts/kalcb_checkpoint_variant.py
def meets_value_margin(row, prediction, roundtrip_bps):
    """Same economic hurdle across sizes, expressed in reference-equity bps."""
    import math
    notional = row['features']['action_notional_equity']
    if not math.isfinite(notional) or notional <= 0:
        return False
    hurdle = notional * roundtrip_bps
    return min(prediction['baseline_value_bps'], prediction['combined_value_bps']) > hurdle

# Extracted from scripts/kalcb_selective_recovery_v2.py
def choose_day(items, raw, core, support, plan, source, *, priority=(), additions=()):
    from backtests.strategies.kalcb import executed_value as ev
    cfg = config_for(plan)
    seed = read(plan['seed'])['mutations']
    q3 = read(plan['q3'])['mutations']
    pred = lambda r, rank: ev.eligible(r, seed, q3, rank=rank)
    score = lambda v: {s: {k: r[s + '_value_bps'] for k, r in v.items()} for s in sa.SCENARIOS}
    cs, ns = (score(core), score(support))
    choose = lambda rs, vs: choose_quantities(rs, vs['baseline'], vs['combined'], robust=True)[0]
    admit = lambda choices: admit_actions(list(choices.values()), pred, cfg, source, budget=8)[0]
    feasible = [r for r in items if raw[r['opportunity_id']]['failure_probability'] < 0.001]
    original = [r for r in feasible if min((raw[r['opportunity_id']][s + '_value_bps'] for s in sa.SCENARIOS)) > 0]
    selected_original = admit(choose(original, cs))
    locked0 = {r['base_opportunity_id']: (r, v) for r, v in selected_original}
    expanded = choose(feasible, cs)
    expanded.update(locked0)
    selected_core = admit(expanded)
    locked = {r['base_opportunity_id']: (r, v) for r, v in selected_core}
    eligible = [r for r in feasible if r['base_opportunity_id'] in locked or r['features']['action_quantity_scale'] <= 0.5]
    supported = choose(eligible, ns)
    supported.update(locked)
    selected, reasons = admit_actions(list(supported.values()), pred, cfg, source, budget=8)
    origins = {r['opportunity_id']: 'frozen_core' if r['base_opportunity_id'] in locked else 'new_forecast' for r, _ in selected}
    if priority:
        selected, reasons = admit_actions(list(priority) + list(additions), pred, cfg, source, budget=8, priority_keys=[r['opportunity_id'] for r, _ in priority])
        if not {r['opportunity_id'] for r, _ in priority}.issubset({r['opportunity_id'] for r, _ in selected}):
            raise ValueError('Hybrid displaced a protected opening candidate')
        protected = {r['opportunity_id'] for r, _ in priority}
        origins = {r['opportunity_id']: 'protected_expanding' if r['opportunity_id'] in protected else 'decay_satellite' for r, _ in selected}
    elif additions:
        selected, reasons = admit_actions(list(additions), pred, cfg, source, budget=8)
        origins = {r['opportunity_id']: 'decay_satellite' for r, _ in selected}
    return (selected, origins, reasons)

# Extracted from scripts/kalcb_selective_recovery_v2.py
def package_day(day, selected, origins, timed, tf, plan, source, tsource, valid_from):
    from backtests.strategies.kalcb import executed_value as ev
    snap = ev.snapshot_for(day, selected, source=source, forecast={'policy': 'causal_recovery', 'valid_from': valid_from})
    admissions = [dict(opportunity_id=r['base_opportunity_id'], action_id=r['opportunity_id'], trade_date=day, reason='admitted', score=v, quantity_scale=r['features']['action_quantity_scale'], valid_from=valid_from, forecast_origin=origins[r['opportunity_id']]) for r, v in selected]
    eligible = [r for r in timed if r['features']['action_quantity_scale'] <= 0.5 and tf[r['opportunity_id']]['failure_probability'] < 0.001 and meets_value_margin(r, tf[r['opportunity_id']], 48)]
    scores = {s: {k: r[s + '_value_bps'] for k, r in tf.items()} for s in sa.SCENARIOS}
    choices = choose_quantities(eligible, scores['baseline'], scores['combined'], robust=True)[0]
    extra = native_admit(snap, choices, config_for(plan), tsource)
    tads = [dict(trade_date=day, symbol=r['symbol'], available_at=r['feature_available_at'], score=v, action_id=r['opportunity_id'], quantity_scale=r['features']['action_quantity_scale'], valid_from=tf[r['opportunity_id']]['valid_from']) for r, v in extra]
    return {'snapshots': {date.fromisoformat(day): snap}, 'updates': {date.fromisoformat(day): [update_snapshot(snap, extra, tsource)]} if extra else {}, 'admissions': admissions, 'timed_admissions': tads}

# Extracted from scripts/kalcb_selective_recovery_v2.py
def empty_result():
    return {'snapshots': {}, 'updates': {}, 'admissions': [], 'timed_admissions': []}

# Extracted from scripts/kalcb_selective_recovery_v2.py
def merge(result, current):
    for k in ('snapshots', 'updates'):
        result[k].update(current[k])
    for k in ('admissions', 'timed_admissions'):
        result[k].extend(current[k])

# Extracted from scripts/kalcb_executed_value_replay.py
def verify_policy(directory, name):
    completion = json.loads((directory / 'completion.json').read_text())
    if not completion.get('complete'):
        raise ValueError('Uncompleted policy generation')
    manifest = json.loads((directory / 'policies_manifest.json').read_text())
    if name not in manifest['policies']:
        raise ValueError('Undeclared candidate policy')
    for path, expected in manifest['files'].items():
        if ev.digest(directory / path) != expected:
            raise ValueError(f'Changed policy artifact: {path}')
    chronology = json.loads((directory / 'chronology.json').read_text())
    from datetime import datetime
    for fold in chronology['folds']:
        if fold['train_through'] >= fold['valid_from'] or datetime.fromisoformat(fold['label_available_through']) >= datetime.fromisoformat(fold['fit_at']):
            raise ValueError('Noncausal fit provenance')
    for calibration in chronology.get('calibration', []):
        latest = calibration.get('latest_label_available_at')
        cutoff = datetime.fromisoformat(calibration['valid_from'] + 'T00:00:00+09:00')
        if latest and datetime.fromisoformat(latest) >= cutoff:
            raise ValueError('Noncausal admission calibration')
    from strategy_kalcb.models import KALCBDailySnapshot
    raw = json.loads((directory / (name + '.snapshots.json')).read_text())
    snapshots = {}
    for row in raw:
        snap = KALCBDailySnapshot.from_json_dict(row)
        if snap.artifact_hash != row['artifact_hash'] or snap.trade_date in snapshots:
            raise ValueError('Changed/duplicate snapshot')
        snapshots[snap.trade_date] = snap
    decisions = ev.read_jsonl(directory / (name + '.admission.jsonl'))
    if len({r['opportunity_id'] for r in decisions}) != len(decisions):
        raise ValueError('Duplicate admission decision')
    admitted = {r['opportunity_id']: r for r in decisions if r['reason'] == 'admitted'}
    actual = {f'{day}/{c.symbol}': c for day, s in snapshots.items() for c in s.candidates}
    if actual.keys() != admitted.keys():
        raise ValueError('Snapshot/causal admission mismatch')
    for key, candidate in actual.items():
        if candidate.selection_score != admitted[key]['score']:
            raise ValueError('Snapshot/forecast score mismatch')
    for snapshot in snapshots.values():
        if len(snapshot.candidates) > manifest['contract']['budget']:
            raise ValueError('Candidate budget exceeded')
    return (manifest, snapshots)

# Extracted from scripts/kalcb_executed_value_replay.py
def timed_policy(directory, manifest, snapshots):
    """Validate additional decision-time snapshots without changing the opening pool."""
    from datetime import datetime
    from strategy_kalcb.models import KALCBDailySnapshot
    filename = manifest.get('timed_snapshot_file')
    if not filename:
        return {}
    if filename not in manifest['files']:
        raise ValueError('Unbound intraday candidates')
    admissions = manifest.get('timed_admission_file')
    if not admissions or admissions not in manifest['files']:
        raise ValueError('Unbound intraday decisions')
    decisions = ev.read_jsonl(directory / admissions)
    expected = {(r['trade_date'], r['symbol'], r['available_at']): r for r in decisions}
    if len(expected) != len(decisions):
        raise ValueError('Duplicate intraday decision')
    used = set()
    result = {}
    for raw in ev.read_jsonl(directory / filename):
        snap = KALCBDailySnapshot.from_json_dict(raw)
        if snap.artifact_hash != raw['artifact_hash']:
            raise ValueError('Changed intraday snapshot')
        if snap.trade_date not in snapshots or snap.generated_at is None:
            raise ValueError('Undeclared intraday session')
        prior = result.setdefault(snap.trade_date, [])
        if prior and prior[-1].generated_at >= snap.generated_at:
            raise ValueError('Unordered intraday updates')
        if len(snap.candidates) > manifest['contract']['timed_budget']:
            raise ValueError('Intraday budget exceeded')
        base = snapshots[snap.trade_date].by_symbol()
        symbols = [c.symbol for c in snap.candidates]
        if len(set(symbols)) != len(symbols) or not set(base).issubset(symbols):
            raise ValueError('Intraday update removes or duplicates opening candidates')
        for c in snap.candidates:
            if c.symbol in base and c.to_json_dict() == base[c.symbol].to_json_dict():
                continue
            if c.symbol in base:
                raise ValueError('Intraday update changes locked opening candidate')
            if c.symbol not in manifest.get('observation_symbols', []):
                raise ValueError('Undeclared watch symbol')
            key = (str(snap.trade_date), c.symbol, str(c.metadata.get('entry_checkpoint_at')))
            if key not in expected or key in used:
                raise ValueError('Missing/duplicate timed admission')
            row = expected[key]
            if c.selection_score != row['score'] or datetime.fromisoformat(key[2]) != snap.generated_at:
                raise ValueError('Intraday score/clock mismatch')
            if datetime.fromisoformat(c.metadata['feature_available_at']) > snap.generated_at:
                raise ValueError('Future features in intraday selection')
            used.add(key)
        prior.append(snap)
    if used != set(expected):
        raise ValueError('Unused intraday admission')
    return result

# Extracted from scripts/kalcb_recovery_ablation.py
def transform(parent, dest, mode):
    parent = Path(parent)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    pm = read(parent / 'policies_manifest.json')
    opening = read(parent / (POLICY + '.snapshots.json'))
    admissions = rows(parent / (POLICY + '.admission.jsonl'))
    timed = rows(parent / 'timed.snapshots.jsonl')
    tads = rows(parent / 'timed.admission.jsonl')
    changed = {POLICY + '.snapshots.json', POLICY + '.admission.jsonl', 'timed.snapshots.jsonl', 'timed.admission.jsonl', 'policies_manifest.json', 'completion.json'}
    for p in parent.iterdir():
        if p.is_file() and p.name not in changed:
            link(p, dest / p.name)
    excluded = {}
    if mode == 'no_support_additions':
        for r in admissions:
            if r.get('forecast_origin') == 'new_forecast':
                excluded.setdefault(r['trade_date'], set()).add(r['opportunity_id'].split('/')[1])
    osn = [transform_snapshot(r, mode, excluded.get(r['trade_date'], ())) for r in opening]
    tsn = [] if mode == 'no_checkpoint' else [transform_snapshot(r, mode, excluded.get(r['trade_date'], ())) for r in timed]

    def adjust(admissions, snaps, timed=False):
        lookup = {(r['trade_date'], c['symbol']): c for r in snaps for c in r['candidates']}
        result = []
        for r in admissions:
            key = (r['trade_date'], r['symbol'] if timed else r['opportunity_id'].split('/')[1])
            if key not in lookup:
                continue
            q = float(lookup[key]['metadata'].get('action_quantity_scale', 1))
            v = dict(r)
            if q != float(r.get('quantity_scale', q)):
                v.update(source_action_id=r.get('action_id'), quantity_scale=q, action_id=f'{key[0]}/{key[1]}/diagnostic_q{q:.12g}')
            result.append(v)
        return result
    write(dest / (POLICY + '.snapshots.json'), osn)
    jsonl(dest / (POLICY + '.admission.jsonl'), adjust(admissions, osn))
    jsonl(dest / 'timed.snapshots.jsonl', tsn)
    jsonl(dest / 'timed.admission.jsonl', adjust(tads, tsn, True))
    manifest = {**pm, 'diagnostic_transform': {'mode': mode, 'parent_manifest': bound(parent / 'policies_manifest.json'), 'no_new_fit_or_outcome_condition': True, 'quantity_estimates_not_rescored': mode.startswith(('atr_quantity', 'q1_'))}}
    manifest['files'] = {p.name: digest(p) for p in dest.iterdir() if p.is_file()}
    write(dest / 'policies_manifest.json', manifest)
    write(dest / 'completion.json', {'complete': True})
    m, s = verify_policy(dest, POLICY)
    timed_policy(dest, m, s)
    write(dest / 'completion.json', {'complete': True, 'transformation_verified': True})

# Extracted from scripts/kalcb_alignment_walk_forward_v3.py
def save_period(out, parent, period, result, folds, calibrations, evidence, recipe, registration):
    first, last, count = PERIODS[period]
    dates = [str(d) for d in result['snapshots']]
    if len(dates) != count or dates[0] != first or dates[-1] != last:
        raise ValueError('Wrong scored policy coverage')
    out.mkdir(parents=True, exist_ok=False)
    write(out / (POLICY + '.snapshots.json'), [s.to_json_dict() for s in result['snapshots'].values()])
    jsonl(out / (POLICY + '.admission.jsonl'), result['admissions'])
    jsonl(out / 'timed.snapshots.jsonl', (s.to_json_dict() for items in result['updates'].values() for s in items))
    jsonl(out / 'timed.admission.jsonl', result['timed_admissions'])
    relevant = lambda x: x['valid_from'] <= last and x['valid_through'] >= first
    pf = [p for p in folds if relevant(p)]
    write(out / 'chronology.json', {'folds': pf, 'calibration': [p for p in calibrations if relevant(p)], 'scope': 'Full-history causal development; reporting split does not refit or reset calibration.'})
    write(out / 'lineage.json', {'label_manifests_and_receipts': evidence, 'recipe': recipe, 'registration': bound(registration), 'risk_models': 'Expanding paired physical-feasibility models, not loss classifiers.', 'models': [{'path': p['model_path'], 'sha256': p['model_sha256']} for p in pf]})
    old = read(parent / 'policies_manifest.json')
    manifest = {k: deepcopy(old[k]) for k in ('contract', 'timed_snapshot_file', 'timed_admission_file', 'observation_symbols', 'observation_minutes')}
    manifest.update(version='kalcb-aligned-prequential-v3', policies=[POLICY], evaluation_from=first, evaluation_through=last, files={p.name: digest(p) for p in out.iterdir()}, scope='Reused development; actual newly fitted IS and OOS policies.', reference_policy_manifest=bound(parent / 'policies_manifest.json'), recipe=recipe)
    write(out / 'policies_manifest.json', manifest)
    write(out / 'completion.json', {'complete': True})
    m, snapshots = verify_policy(out, POLICY)
    timed_policy(out, m, snapshots)

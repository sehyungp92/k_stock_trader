"""Read-only complete opening decision audit; no model or portfolio experiments."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

BASE = Path('E:/CodexResearch/k_stock_trader')
ROOT = BASE/'kalcb_late_entry_20261011'
OUT = ROOT/'v26_opening_decision_audit'
SCENARIOS = ('baseline', 'combined')
FEATURES = ('daily_atr_pct', 'first30_range_atr', 'first30_rel_volume_log',
            'first30_close_location', 'path_efficiency', 'path_above_vwap_share',
            'sector_intraday_breadth', 'market_adjusted_first30_ret')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, obj):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)


def exact_denominator(row, scale):
    original = int(row['action']['quantity'])
    quantity = math.floor(original*scale)
    close = float(row['opening_bars'][-1]['close'])
    original_fraction = float(row['features']['action_notional_equity'])
    if original <= 0 or quantity <= 0 or close <= 0:
        raise ValueError('Nonpositive action denominator')
    if not math.isclose(original*close/100_000_000, original_fraction, abs_tol=1e-12):
        raise ValueError('Original quantity/notional mismatch')
    return quantity, quantity*close/100_000_000


def summary(rows):
    result = {'actions': len(rows), 'issuer_days': len({r['base_id'] for r in rows}),
              'sessions': len({r['day'] for r in rows})}
    for scenario in SCENARIOS:
        valid = [r for r in rows if r['labels'][scenario]['label_valid']]
        y = [r['labels'][scenario]['net_value_equity_bps']/r['notional_equity'] for r in valid]
        predicted = [r['calibrated_core'][scenario]/r['notional_equity'] for r in valid]
        positive = [v for v in y if v > 0]
        negative = [v for v in y if v < 0]
        result[scenario] = {'valid': len(valid), 'invalid': len(rows)-len(valid),
            'mean_net_notional_bps': statistics.mean(y) if y else None,
            'median_net_notional_bps': statistics.median(y) if y else None,
            'positive_fraction': len(positive)/len(y) if y else None,
            'mean_positive_bps': statistics.mean(positive) if positive else None,
            'mean_negative_bps': statistics.mean(negative) if negative else None,
            'mean_forecast_notional_bps': statistics.mean(predicted) if y else None,
            'mean_overprediction_notional_bps': statistics.mean(p-v for p,v in zip(predicted,y)) if y else None}
    return result


def main():
    OUT.mkdir(exist_ok=False)
    bindings = {}
    def take(path, expected=None):
        path = Path(path)
        actual = sha(path)
        if expected and actual != expected:
            raise ValueError('Input hash mismatch: '+str(path))
        bindings[str(path)] = actual
        return json.loads(path.read_text(encoding='utf-8-sig'))
    def lines(path, expected=None, compressed=False):
        path = Path(path)
        actual = sha(path)
        if expected and actual != expected: raise ValueError('Input hash mismatch: '+str(path))
        bindings[str(path)] = actual
        opener = gzip.open if compressed else open
        with opener(path, 'rt', encoding='utf-8-sig') as stream:
            return [json.loads(line) for line in stream if line.strip()]

    save(OUT/'protocol.json', {'registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'kind':'read_only_descriptive_diagnostic', 'model_fits':0, 'native_replays':0,
        'features':FEATURES, 'cohorts':'full actions, physical/capacity eligible, positive core forecasts, current prior100/125 selected and actually filled; monthly and exact quantity views',
        'context_bins':'IS-only quartiles applied unchanged OOS, descriptive only; no threshold or mechanism selection from bucket outcomes',
        'limits':'Quantity alternatives/scenarios are dependent. Isolated labels cannot be summed as funded portfolio PnL. April-July reused development.'})
    dependency = take(BASE/'kalcb_selective_recovery_20261010/v5_population_setup/build_v1/input_bindings.json')
    expected = {str(Path(v['path']).resolve()).lower():v['sha256'] for v in dependency}
    audit_path = BASE/'kalcb_selective_recovery_20261010/v2/build/decision_audit.jsonl.gz'
    audit = lines(audit_path, expected[str(audit_path.resolve()).lower()], compressed=True)
    alignment = take(BASE/'kalcb_selected_target_alignment_20261010/v2/protocol.json')
    opportunities = {}
    for cell in alignment['cells']:
        if cell['kind'] != 'opening': continue
        path = Path(cell['prepared'])/'opportunities.jsonl'
        meta = take(Path(cell['prepared'])/'prepared.json')
        for row in lines(path, meta['opportunities_sha256']):
            key = row['opportunity_id']
            if key in opportunities and opportunities[key] != row: raise ValueError('Contradictory source row')
            opportunities[key] = row
    records = []
    seen = set()
    for row in audit:
        if row['family'] != 'expanding': continue
        key = row['action_id']; base_id = key.rsplit('/q', 1)[0]
        if key in seen: raise ValueError('Duplicate exact action')
        seen.add(key)
        source = opportunities[base_id]
        if row['available_at'] != source['feature_available_at'] or row['valid_from'] > row['day']:
            raise ValueError('Forecast/source chronology mismatch')
        quantity, denominator = exact_denominator(source, row['quantity'])
        records.append(dict(action_id=key, base_id=base_id, day=row['day'], period=row['period'],
            quantity_scale=row['quantity'], reference_quantity=quantity, notional_equity=denominator,
            available_at=row['available_at'], base_valid_from=row['valid_from'],
            raw=row['raw'], calibrated_core=row['calibrated_core'], calibrated_support=row['calibrated_support'],
            labels=row['selected_management_labels'], failure_probability=row['failure_probability'],
            physical_risk_pass=row['physical_risk_pass'], capacity_pass=row['capacity_pass'],
            physical_outcome=row['physical_outcome'], features=source['features'],
            descriptor=row['descriptor'], original_parent_selected=row['parent_selected'],
            original_parent_reason=row['parent_admission_reason'], selected={}, actual={}))
    lookup = {r['action_id']:r for r in records}
    accounts = take(ROOT/'v25_pending_funding_screen/assessment_v1/assessment.json')['accounts']
    reconciliation = []
    management = take(BASE/'kalcb_selected_target_alignment_20261010/v2/selected_management.json')
    management_differences = []
    for recipe in ('control','control_prior125'):
        for period in ('is','oos'):
            profile = take(ROOT/f'v25_pending_funding_screen/cases/{recipe}/{period}/input_profile.json')
            plan = take(profile['plan']['path'], profile['plan']['sha256'])
            cfg = take(plan['config'], plan['input_sha256'][plan['config']])
            for k in management:
                if k.startswith(('exit_', 'failure_stop_', 'partial_', 'profit_', 'time_stop_')) and cfg.get(k) != management[k]:
                    management_differences.append({'recipe':recipe,'period':period,'key':k,'label':management[k],'native':cfg.get(k)})
            manifest_path = Path(profile['policy']['path'])
            manifest = take(manifest_path, profile['policy']['sha256'])
            admission_name='interaction_cost_margin.admission.jsonl'
            admissions = lines(manifest_path.parent/admission_name, manifest['files'][admission_name])
            bybase = {}
            for admission in admissions:
                r = lookup[admission['action_id']]
                if r['period'] != period or r['base_id'] in bybase: raise ValueError('Selection join mismatch')
                r['selected'][recipe] = admission
                bybase[r['base_id']] = r
            for scenario in SCENARIOS:
                account = accounts[f'{recipe}/{period}/{scenario}']
                if not account['bridge']['passed']: raise ValueError('Unverified reference account')
                positions = [p for p in account['positions'] if p['route']=='first30_open_anchor']
                if len({p['position_id'] for p in positions}) != len(positions): raise ValueError('Duplicated position lifetime')
                for position in positions:
                    r = bybase[position['entry_day']+'/'+position['symbol']]
                    r['actual'].setdefault(recipe+'/'+scenario, []).append(position)
                reconciliation.append({'account':f'{recipe}/{period}/{scenario}', 'positions':len(positions),
                    'net_pnl_krw':sum(p['net_pnl_krw'] for p in positions), 'joined_once':True})
    gates = {}
    for period in ('is','oos'):
        for q in (.25,.5,1.):
            rows = [r for r in records if r['period']==period and r['quantity_scale']==q]
            eligible = [r for r in rows if r['physical_risk_pass'] and r['capacity_pass']]
            positive = [r for r in eligible if min(r['calibrated_core'].values()) > 0]
            for name, values in [('all',rows),('physical_capacity',eligible),('positive_core',positive)]:
                gates[f'{period}/q{q}/{name}'] = summary(values)
            for recipe in ('control','control_prior125'):
                selected = [r for r in rows if recipe in r['selected']]
                gates[f'{period}/q{q}/{recipe}'] = summary(selected)
    context = {}
    for feature in FEATURES:
        train = [r for r in records if r['period']=='is' and 'control' in r['selected'] and r['features'].get(feature) is not None]
        boundaries = statistics.quantiles([r['features'][feature] for r in train], n=4, method='inclusive')
        context[feature] = {'is_boundaries':boundaries, 'bins':{}}
        for period in ('is','oos'):
            rows = [r for r in records if r['period']==period and 'control' in r['selected']]
            for n in range(4):
                values = [r for r in rows if r['features'].get(feature) is not None and
                          sum(r['features'][feature] > c for c in boundaries)==n]
                context[feature]['bins'][period+'/'+str(n+1)] = summary(values)
    monthly = {}
    for month in sorted({r['day'][:7] for r in records}):
        monthly[month] = summary([r for r in records if r['day'].startswith(month) and 'control' in r['selected']])
    with gzip.open(OUT/'decisions.jsonl.gz','xt',encoding='utf-8') as stream:
        for r in records: stream.write(json.dumps(r,allow_nan=False,sort_keys=True)+'\n')
    save(OUT/'assessment.json', {'scope':'descriptive only, no fits or policy change',
        'actions':len(records),'issuer_days':len(opportunities),'gates':gates,'context':context,
        'monthly_selected':monthly,'actual_account_reconciliation':reconciliation,
        'management_field_differences':management_differences,
        'label_semantics':'Decision audit is selected-management labels from selected_targets, not original-management labels. Existing population-calibration target text is imprecise; original immutable artifacts preserved.',
        'source_binding_count':len(bindings)})
    for path, expected_hash in bindings.items():
        if sha(path) != expected_hash: raise ValueError('Input mutated during audit')
    save(OUT/'completion.json', {'complete':True,'study_complete':False,'fitted_models':0,'native_cases':0,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'program':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
        'sources':[{'path':p,'sha256':h} for p,h in bindings.items()],
        'artifacts':[{'path':str(OUT/f),'sha256':sha(OUT/f)} for f in ('protocol.json','decisions.jsonl.gz','assessment.json')]})
    print(json.dumps({'complete':True,'actions':len(records),'accounts':len(reconciliation),'management_differences':management_differences,'root':str(OUT)}))


if __name__ == '__main__': main()

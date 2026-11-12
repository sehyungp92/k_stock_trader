"""Fail-closed reading and comparison of completed funded experiments.

An intermediate NAV, a fitted model, or a passing component comparison is not
a qualified strategy successor. Preserve fine-event and common-grid risk.
"""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import hashlib
import json
from statistics import median


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def opportunity_key(trade):
    """OLR's one confirmed opportunity per name/session, including waiting.

    A delayed submission of the same opportunity is not a newly discovered
    trade. Multiple lifecycles per name/session need explicit lineage rather
    than this bounded-family identity contract.
    """
    stamp = datetime.fromisoformat(trade['entry_decision_time'])
    if stamp.tzinfo is None:
        raise ValueError('Aware opportunity clock required')
    return trade['symbol'],stamp.astimezone(timezone(timedelta(hours=9))).date().isoformat()


def acquisition_by_opportunity(trades):
    acquired = {}
    for trade in trades:
        key = opportunity_key(trade)
        if key in acquired:
            raise ValueError('Multiple lifecycles require explicit opportunity lineage')
        acquired[key] = trade['qty']*trade['entry_price']
    return acquired


def completed_case(path):
    path = Path(path)
    receipt = read(path/'completion.json')
    if receipt.get('complete') is not True:
        raise ValueError('Incomplete native replay')
    required={'result.json','execution_audit.json','minute_nav_evidence.json','trades.json','equity.json'}
    if not required.issubset(receipt.get('artifacts',{})):
        raise ValueError('Required economic artifacts are not receipt-bound')
    for name, digest in receipt.get('artifacts', {}).items():
        if Path(name).name != name or sha(path/name) != digest:
            raise ValueError('Changed or invalid native artifact: '+name)
    for name, key in [('result.json','result_sha256'), ('execution_audit.json','audit_sha256')]:
        if sha(path/name) != receipt.get(key):
            raise ValueError('Changed completed evidence: '+name)
    result = read(path/'result.json')
    audit = read(path/'execution_audit.json')
    if result['source_generation_hash'] != receipt['source_generation_hash']:
        raise ValueError('Source identity mismatch')
    if audit['audit_receipt']['source_generation_hash'] != receipt['source_generation_hash']:
        raise ValueError('Auditor source identity mismatch')
    nav = read(path/'minute_nav_evidence.json')
    # The generic broker metric and the independently reconstructed complete
    # opening/closing/fill-cost path have different event grids. Keep both;
    # equality would incorrectly reject legitimate intraminute risk evidence.
    trades = read(path/'trades.json')
    if len(trades) != result['trades']:
        raise ValueError('Funded trade count mismatch')
    volume = audit.get('capacity_model_audit', {})
    valid = (result.get('execution_validity_pass') is True
             and audit.get('audit_pass') is True
             and audit.get('execution_validity_pass') is True
             and audit.get('audit_status') == 'passed'
             and not volume.get('errors')
             and not volume.get('impossible_volume_events')
             and not result.get('terminal_inventory')
             and result.get('terminal_pending') == 0)
    acquired = acquisition_by_opportunity(trades)
    return dict(path=str(path), receipt_sha256=sha(path/'completion.json'),
        source=result['source_generation_hash'], valid=valid,
        return_pct=result['return_pct'], reported_dd=result['drawdown_pct'],
        full_dd=nav['all_event_drawdown_pct'],
        common_dd=nav['common_original_drawdown_pct'], grid=nav['original_grid_sha256'],
        trades=result['trades'], days=result['active_days'], acquired=acquired,
        phase_liquidity_verified=volume.get('phase_liquidity_verified') is True)


def compare(control, candidate, *, window='train', scenario='baseline', initial_equity=100_000_000):
    if control['grid'] != candidate['grid']:
        raise ValueError('Controls and treatments use different comparison grids')
    is_train = window == 'train'
    gates = dict(
        execution=control['valid'] and candidate['valid'],
        return_preservation=candidate['return_pct'] >= control['return_pct']-(1.5 if is_train else 0),
        completed_entries=candidate['trades'] >= max(control['trades'],186 if is_train else 47),
        active_entry_days=candidate['days'] >= max(control['days'],88 if is_train else 29))
    # Report strict risk improvement separately even in IS, where older
    # protocols allowed modest deterioration. Never hide a worse fine path.
    gates.update(full_dd_lower=candidate['full_dd'] < control['full_dd'],
                 common_dd_lower=candidate['common_dd'] < control['common_dd'])
    added = [v for k,v in candidate['acquired'].items() if k not in control['acquired']]
    removed = [v for k,v in control['acquired'].items() if k not in candidate['acquired']]
    notional = dict(added=sum(added), removed=sum(removed),
        net=sum(added)-sum(removed), median_added=median(added) if added else 0.)
    if not is_train:
        gates['meaningful_incremental_notional'] = notional['net'] >= .01*initial_equity
        gates['meaningful_median_added_entry'] = notional['median_added'] >= .01*initial_equity
    return dict(window=window, scenario=scenario, gates=gates,
        full_objective_cell_pass=all(gates.values()), notional=notional,
        delta=dict(return_pp=candidate['return_pct']-control['return_pct'],
            full_dd_pp=candidate['full_dd']-control['full_dd'],
            common_dd_pp=candidate['common_dd']-control['common_dd'],
            trades=candidate['trades']-control['trades'], days=candidate['days']-control['days']))


def suite_status(comparisons, *, retained_reference_gates_pass=False, forecast_validation_pass=False):
    required = {(w,s) for w in ('train','oos') for s in ('baseline','combined','baseline_capped')}
    keys = [(r['window'],r['scenario']) for r in comparisons]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate economic comparison')
    missing = sorted(required-set(keys))
    return dict(missing_comparisons=missing,
        funded_successor_qualified=bool(not missing and set(keys)==required
            and all(r['full_objective_cell_pass'] for r in comparisons)
            and retained_reference_gates_pass and forecast_validation_pass),
        retained_reference_gates_pass=retained_reference_gates_pass,
        forecast_validation_pass=forecast_validation_pass)

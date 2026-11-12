"""Immutable publication of an exact-parent runtime plus explicit new services."""
from copy import deepcopy
from pathlib import Path

from backtests.auto.shared.cache_keys import stable_signature
from .retention_research_io_v1 import read, write, bound, digest, verify, source_hash
from .late_entry_io_v5 import cached_link

OVERLAYS = (
    'strategy_kalcb/core/account_allocation_v1.py',
    'strategy_kalcb/core/pending_entry_revalidation_v1.py',
    'backtests/strategies/kalcb/account_decision_v1.py',
    'backtests/strategies/kalcb/account_history_v1.py',
    'backtests/strategies/kalcb/opening_book_risk_v1.py',
    'backtests/strategies/kalcb/opening_account_replay_v1.py',
    'backtests/strategies/kalcb/pending_entry_account_pilot_v1.py',
    'scripts/kalcb_opening_account_worker_v1.py',
)


def patch_core(text):
    anchor = '    if qty <= 0:\n        gates.append(_gate("qty_sizing", 1, qty, False))\n        return _rejected(state, bar, "qty_zero", gates, {**entry_meta, "candidate_rank": candidate_rank})'
    if text.count(anchor) != 1:
        raise ValueError('Exact parent integer decision anchor changed')
    return text.replace(anchor,
        '    if qty > 0:\n'
        '        from strategy_kalcb.core.account_allocation_v1 import allocate_quantity\n'
        '        qty = allocate_quantity(bar.symbol, qty, entry_meta)\n' + anchor)


def freeze(root, name, period, old):
    root = Path(root)
    p = read(root/'protocol.json')
    folder = root/'runtime'/name/period
    receipt = folder/'profile.json'
    if receipt.exists():
        item = read(receipt)
        verify([item[k] for k in ('plan','policy','manifest')])
        return item
    verify([old[k] for k in ('plan','policy','manifest')])
    manifest_path = Path(old['manifest']['path'])
    parent = read(manifest_path)
    sig = parent.pop('manifest_hash')
    if stable_signature(parent) != sig:
        raise ValueError('Invalid frozen parent generation')
    folder.mkdir(parents=True, exist_ok=False)
    plan = read(old['plan']['path'])
    cfg = read(plan['config'])
    if cfg.get('risk_max_gross_equity_pct',0) or cfg.get('risk_daily_vol_target',0):
        raise ValueError('Unexpected parent risk policy')
    write(folder/'config.json', cfg)
    context = read(plan['audit_configuration'])
    context.update(output_root=str(folder), promotion_status='opening_account_research_only',
                   opening_account_allocation_mode=name)
    write(folder/'audit_configuration.json', context)
    for key in ('config','audit_configuration'):
        plan['input_sha256'].pop(plan[key],None)
        plan[key] = str((folder/(key+'.json')).resolve())
        plan['input_sha256'][plan[key]] = digest(plan[key])
    plan['opening_account_research'] = {'protocol':bound(root/'protocol.json'), 'parent':old,
        'databases':p['databases'], 'database_bindings':p['database_bindings'],
        'mode':name, 'period':period, 'config_unchanged':True}
    write(folder/'plan.json', plan)
    code = folder/'inputs/files'
    replacements = {rel:Path(b['path']) for rel,b in p['overlays'].items()}
    replacements.update({'research/kalcb_executed_value/plan.json':folder/'plan.json',
        'research/opening_account/protocol.json':root/'protocol.json',
        'research/opening_account/config.json':folder/'config.json',
        'research/opening_account/audit_configuration.json':folder/'audit_configuration.json'})
    patched = folder/'patched_logic.py'
    patched.write_text(patch_core((manifest_path.parent/'files/strategy_kalcb/core/logic.py').read_text(encoding='utf-8')),encoding='utf-8')
    replacements['strategy_kalcb/core/logic.py'] = patched
    files = {}
    for rel,sha in parent['files'].items():
        if rel in replacements:
            continue
        cached_link(manifest_path.parent/'files'/rel, code/rel, sha, root/'content_cache')
        files[rel] = sha
    for rel,source in replacements.items():
        sha = digest(source)
        cached_link(source, code/rel, sha, root/'content_cache')
        files[rel] = sha
    payload = deepcopy(parent)
    payload['files'] = files
    payload['source_generation_hash'] = source_hash(code)
    payload['evidence']['opening_account_research'] = {'parent_manifest':old['manifest'],
        'protocol':bound(root/'protocol.json'), 'new_services':p['overlays']}
    write(folder/'inputs/manifest.json', {**payload,'manifest_hash':stable_signature(payload)})
    policy = code/'research/kalcb_executed_value/policies/policies_manifest.json'
    if digest(policy) != old['policy']['sha256']:
        raise ValueError('Changed parent forecasts')
    item = {'name':name,'period':period,'folder':str(folder),
        'source_generation_hash':payload['source_generation_hash'],
        'manifest':bound(folder/'inputs/manifest.json'),'plan':bound(folder/'plan.json'),'policy':bound(policy)}
    write(receipt,item)
    return item

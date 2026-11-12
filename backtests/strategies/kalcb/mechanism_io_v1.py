"""Immutable native publication; patches only new successor source copies."""
from copy import deepcopy
from pathlib import Path
from backtests.auto.shared.cache_keys import stable_signature
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, digest, bound, verify, source_hash
from backtests.strategies.kalcb.late_entry_io_v4 import cached_link, exactly

FIELD = 'research_opening_failure_min_hold_bars'


def patch_logic(text):
    start = text.index('def _exit_reason(')
    end = text.index('\ndef ', start+4)
    body = text[start:end]
    anchor = '    return ""\n'
    at = body.rfind(anchor)
    if at < 0 or body[at+len(anchor):].strip():
        raise ValueError('Missing final parent exit fallthrough')
    code = ('    if config.'+FIELD+':\n'
            '        from strategy_kalcb.opening_failure_v1 import evidence\n'
            '        proof = evidence(symbol_state, position, bar, config.'+FIELD+')\n'
            '        if proof:\n'
            '            position.metadata["opening_failure_signal"] = proof\n'
            '            return "opening_thesis_failure"\n')
    return text[:start] + body[:at] + code + body[at:] + text[end:]


def make_overrides(code, target, programs):
    code, target = Path(code), Path(target)
    target.mkdir(parents=True, exist_ok=False)
    for rel, binding in programs.items():
        verify([binding]); p = target/rel; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(Path(binding['path']).read_bytes())
    rel = 'strategy_kalcb/core/logic.py'
    p = target/rel; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(patch_logic((code/rel).read_text(encoding='utf-8')), encoding='utf-8')
    rel = 'strategy_kalcb/config.py'
    text = (code/rel).read_text(encoding='utf-8')
    text = exactly(text, '    strategy_id: str = STRATEGY_ID\n',
                   '    '+FIELD+': int = 0\n    strategy_id: str = STRATEGY_ID\n')
    (target/rel).write_text(text, encoding='utf-8')
    return {p.relative_to(target).as_posix(): bound(p) for p in target.rglob('*.py')}


def freeze(root, name, period, old):
    protocol = read(root/'protocol.json'); recipe = protocol['recipes'][name]
    folder = Path(protocol['runtime_root'])/'cases'/name/period
    receipt = root/'cases'/name/period/'input_profile.json'
    if receipt.exists():
        item = read(receipt); verify([item[k] for k in ('manifest', 'plan', 'policy')]); return item
    verify([old[k] for k in ('manifest', 'plan', 'policy')]+list(protocol['overrides'][period].values()))
    pp = Path(old['manifest']['path']); parent = read(pp); identity = parent.pop('manifest_hash')
    if stable_signature(parent) != identity: raise ValueError('Changed parent manifest')
    folder.mkdir(parents=True, exist_ok=False)
    plan = read(old['plan']['path']); config = read(plan['config']); context = read(plan['audit_configuration'])
    if recipe['hold'] not in (0,6,8): raise ValueError('Unregistered hold')
    config[FIELD] = recipe['hold']; write(folder/'config.json', config)
    context.update(output_root=str(folder), promotion_status='opening_mechanism_research_only')
    write(folder/'audit_configuration.json', context)
    for key in ('config', 'audit_configuration'):
        plan['input_sha256'].pop(plan[key], None)
        plan[key] = str((folder/(key+'.json')).resolve()); plan['input_sha256'][plan[key]] = digest(plan[key])
    plan['opening_mechanism_research'] = {'protocol':bound(root/'protocol.json'), 'parent':old,
        'recipe':recipe, 'selection_unchanged':True, 'native_observer_only':recipe['hold']==0}
    write(folder/'plan.json', plan)
    replacements = {rel:Path(b['path']) for rel,b in protocol['overrides'][period].items()}
    replacements.update({'research/kalcb_executed_value/plan.json':folder/'plan.json',
        'research/mechanism/protocol.json':root/'protocol.json', 'research/mechanism/config.json':folder/'config.json',
        'research/mechanism/audit_configuration.json':folder/'audit_configuration.json',
        'research/mechanism/alternatives.json':root/'alternatives.json'})
    files={}; cache=Path(protocol['runtime_root'])/'content_cache'
    for rel,sha in parent['files'].items():
        if rel in replacements: continue
        cached_link(pp.parent/'files'/rel, folder/'inputs/files'/rel, sha, cache); files[rel]=sha
    for rel,source in replacements.items():
        sha=digest(source); cached_link(source, folder/'inputs/files'/rel, sha, cache); files[rel]=sha
    payload=deepcopy(parent); payload['files']=files
    payload['evidence']['opening_mechanism']={'registration':bound(root/'protocol.json'),
        'parent_manifest':bound(pp), 'config_change':{FIELD:recipe['hold']},
        'policy_unchanged':old['policy'], 'overrides':protocol['overrides'][period]}
    code=folder/'inputs/files'; actual=source_hash(code); payload['source_generation_hash']=actual
    if actual==old['source_generation_hash']: raise ValueError('Expected new registered source identity')
    write(folder/'inputs/manifest.json',{**payload,'manifest_hash':stable_signature(payload)})
    policy=code/'research/kalcb_executed_value/policies/policies_manifest.json'
    if digest(policy)!=old['policy']['sha256']: raise ValueError('Changed parent policy')
    item={'name':name,'period':period,'folder':str(folder),'source_generation_hash':actual,
          'manifest':bound(folder/'inputs/manifest.json'),'plan':bound(folder/'plan.json'),'policy':bound(policy)}
    write(receipt,item); return item

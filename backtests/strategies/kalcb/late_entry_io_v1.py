"""Versioned, explicit source publication for the first-hour research contract."""
from copy import deepcopy
from pathlib import Path
from backtests.auto.shared.cache_keys import stable_signature
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,digest,bound,verify,link,source_hash


def exactly(text,old,new):
    if text.count(old)!=1:raise ValueError('Patch anchor is not unique: '+old[:80])
    return text.replace(old,new)


def make_overrides(code,target,programs):
    """Never change archived files or live/shared repository source in place."""
    code=Path(code);target=Path(target);target.mkdir(parents=True,exist_ok=False)
    for relative,binding in programs.items():
        verify([binding]);p=target/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(binding['path']).read_bytes())
    relative='scripts/kalcb_executed_value_replay.py';text=(code/relative).read_text(encoding='utf-8')
    start=text.index('def timed_policy(');end=text.index('\ndef observation_prefix(',start)
    text=text[:start]+"def timed_policy(directory,manifest,snapshots):\n    from backtests.strategies.kalcb.late_entry_v1 import cumulative_timed_policy\n    return cumulative_timed_policy(directory,manifest,snapshots)\n\n"+text[end:]
    text=exactly(text,"if policy_manifest.get('observation_minutes')!=45:raise ValueError('Unsupported audited candidate decision horizon')",
        "if policy_manifest.get('observation_minutes') not in (45,60):raise ValueError('Unsupported audited candidate decision horizon')")
    text=exactly(text,"{'version':'causal-timed-candidates-v1','strategy':'kalcb',\n                'feature_intraday_before':'09:45','candidate_intraday_before':'15:20'}",
        "{'version':('causal-timed-candidates60-v1' if policy_manifest['observation_minutes']==60 else 'causal-timed-candidates-v1'),'strategy':'kalcb',\n                'feature_intraday_before':('10:00' if policy_manifest['observation_minutes']==60 else '09:45'),'candidate_intraday_before':'15:20'}")
    p=target/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')
    relative='backtests/engine/qualification_evidence.py';text=(code/relative).read_text(encoding='utf-8')
    text=exactly(text,"'causal-feature-and-management-v2','causal-timed-candidates-v1')","'causal-feature-and-management-v2','causal-timed-candidates-v1','causal-timed-candidates60-v1')")
    text=exactly(text,"    if (version not in SOURCE_SCOPE_VERSIONS", "    if version=='causal-timed-candidates60-v1':\n        if strategy!='kalcb':raise ValueError('unsupported_source_dependency_scope')\n        expected=('10:00','15:20')\n    if (version not in SOURCE_SCOPE_VERSIONS")
    text=exactly(text,"{'causal-feature-and-management-v2','causal-timed-candidates-v1'}","{'causal-feature-and-management-v2','causal-timed-candidates-v1','causal-timed-candidates60-v1'}")
    p=target/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')
    relative='strategy_kalcb/core/logic.py';text=(code/relative).read_text(encoding='utf-8')
    text=exactly(text,'    if not symbol_state.opening_range_built:\n',
        '    if checkpoint and candidate.metadata.get("later_entry_flat_only"):\n'
        '        from backtests.strategies.kalcb.late_entry_v1 import flat_blocked\n'
        '        if flat_blocked(state, portfolio):\n'
        '            return KALCBCoreResult(state, decisions=[_decision(bar, "entry_blocked", "later_entry_requires_flat_account")])\n\n'
        '    if not symbol_state.opening_range_built:\n')
    # The action decision belongs to the shared core. Inline the tiny predicate
    # here to avoid introducing a backtest dependency into strategy source.
    text=text.replace('        from backtests.strategies.kalcb.late_entry_v1 import flat_blocked\n        if flat_blocked(state, portfolio):',
        '        if (portfolio.open_positions or any(portfolio.positions.values()) or\n'
        '                any(s.pending_entry_order_id or s.position is not None for s in state.symbols.values())):')
    p=target/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')
    return {p.relative_to(target).as_posix():bound(p) for p in target.rglob('*.py')}


def freeze(root,name,period,old,policy,overrides,authenticated):
    folder=root/'cases'/name/period;receipt=folder/'input_profile.json'
    if receipt.exists():
        item=read(receipt);verify([item[k] for k in ('manifest','plan','policy')]);return item
    verify([old[k] for k in ('manifest','plan','policy')]+list(overrides.values()))
    pp=Path(old['manifest']['path']);parent=read(pp);identity=parent.pop('manifest_hash')
    if stable_signature(parent)!=identity:raise ValueError('Changed parent generation')
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'inputs').exists() or (folder/'plan.json').exists():raise ValueError('Preserve partial publication')
    plan=read(old['plan']['path']);context=read(plan['audit_configuration']);write(folder/'config.json',read(plan['config']))
    context.update(output_root=str(folder),promotion_status='first_hour_research_only');write(folder/'audit_configuration.json',context)
    for k in ('config','audit_configuration'):
        plan['input_sha256'].pop(plan[k],None);plan[k]=str((folder/(k+'.json')).resolve());plan['input_sha256'][plan[k]]=digest(plan[k])
    plan['first_hour_research']={'protocol':bound(root/'protocol.json'),'parent':old,'config_unchanged':True,'name':name,'period':period}
    write(folder/'plan.json',plan)
    prefix='research/kalcb_executed_value/policies/'
    replacements={prefix+p.name:p for p in Path(policy).iterdir() if p.is_file()}
    replacements.update({k:Path(v['path']) for k,v in overrides.items()})
    replacements.update({'research/kalcb_executed_value/plan.json':folder/'plan.json','research/later60/protocol.json':root/'protocol.json',
        'research/later60/config.json':folder/'config.json','research/later60/audit_configuration.json':folder/'audit_configuration.json'})
    files={}
    for rel,sha in parent['files'].items():
        if rel.startswith(prefix) or rel in replacements:continue
        src=pp.parent/'files'/rel;st=src.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,sha)
        if key not in authenticated:
            if digest(src)!=sha:raise ValueError('Changed inherited file '+rel)
            authenticated.add(key)
        link(src,folder/'inputs/files'/rel);files[rel]=sha
    for rel,src in replacements.items():link(src,folder/'inputs/files'/rel);files[rel]=digest(src)
    payload=deepcopy(parent);payload['files']=files
    payload['evidence']['first_hour_research']={'registration':bound(root/'protocol.json'),'parent_manifest':bound(pp),
        'source_overrides':overrides,'policy':bound(Path(policy)/'policies_manifest.json')}
    code=folder/'inputs/files';actual=source_hash(code);payload['source_generation_hash']=actual
    if actual==old['source_generation_hash']:raise ValueError('New source must have new identity')
    write(folder/'inputs/manifest.json',{**payload,'manifest_hash':stable_signature(payload)})
    item={'name':name,'period':period,'folder':str(folder),'source_generation_hash':actual,'manifest':bound(folder/'inputs/manifest.json'),
        'plan':bound(folder/'plan.json'),'policy':bound(code/prefix/'policies_manifest.json')}
    write(receipt,item);return item

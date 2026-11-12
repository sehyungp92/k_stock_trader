"""Freeze selector-only experiments without weakening the management-only packager."""
from copy import deepcopy
from pathlib import Path
from backtests.auto.shared.cache_keys import stable_signature
from backtests.strategies.kalcb.retention_research_io_v1 import (
    read,write,digest,bound,verify,link,source_hash,validate_native,env,PERIODS,POLICY)


def freeze_case(out,name,period,recipe,old,policy,authenticated):
    out=Path(out);policy=Path(policy);folder=out/'cases'/name/period
    receipt=folder/'input_profile.json'
    if receipt.exists():
        item=read(receipt);verify([item[k] for k in ('manifest','plan','policy')]);return item
    if recipe.get('config'):raise ValueError('Population experiment must not change management')
    verify([old[k] for k in ('manifest','plan','policy')])
    pp=Path(old['manifest']['path']);parent=read(pp);identity=parent.pop('manifest_hash')
    if stable_signature(parent)!=identity:raise ValueError('Changed parent manifest')
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'plan.json').exists() or (folder/'inputs').exists():raise ValueError('Preserve partial frozen case')
    plan=read(old['plan']['path']);config=read(plan['config']);context=read(plan['audit_configuration'])
    oldpaths=[plan['config'],plan['audit_configuration']]
    write(folder/'config.json',config)
    context.update(output_root=str(folder),promotion_status='population_calibration_research_only')
    write(folder/'audit_configuration.json',context)
    for p in oldpaths:plan['input_sha256'].pop(p,None)
    for key in ('config','audit_configuration'):
        plan[key]=str(folder/(key+'.json'));plan['input_sha256'][plan[key]]=digest(plan[key])
    plan['population_research']={'recipe':name,'period':period,'protocol':bound(out/'protocol.json'),
        'parent':recipe['parent'],'selection_only':True,'continuous_history':True}
    write(folder/'plan.json',plan)
    prefix='research/kalcb_executed_value/policies/'
    replacements={prefix+p.name:p for p in policy.iterdir() if p.is_file()}
    replacements.update({'research/kalcb_executed_value/plan.json':folder/'plan.json',
        'research/population_research/protocol.json':out/'protocol.json',
        'research/population_research/config.json':folder/'config.json',
        'research/population_research/audit_configuration.json':folder/'audit_configuration.json'})
    files={}
    for relative,sha in parent['files'].items():
        if relative.startswith(prefix) or relative in replacements:continue
        source=pp.parent/'files'/relative;st=source.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,sha)
        if key not in authenticated:
            if digest(source)!=sha:raise ValueError('Changed frozen source '+relative)
            authenticated.add(key)
        link(source,folder/'inputs/files'/relative);files[relative]=sha
    for relative,source in replacements.items():
        link(source,folder/'inputs/files'/relative);files[relative]=digest(source)
    payload=deepcopy(parent);payload['files']=files
    payload['evidence']['population_research']={'parent_manifest':bound(pp),'registration':bound(out/'protocol.json'),
        'policy':bound(policy/'policies_manifest.json'),'original_policy':old['policy']}
    code=folder/'inputs/files';actual=source_hash(code)
    if actual!=old['source_generation_hash']:raise ValueError('Economic runtime changed')
    payload['source_generation_hash']=actual
    write(folder/'inputs/manifest.json',{**payload,'manifest_hash':stable_signature(payload)})
    item={'name':name,'period':period,'folder':str(folder),'source_generation_hash':actual,
        'manifest':bound(folder/'inputs/manifest.json'),'plan':bound(folder/'plan.json'),
        'policy':bound(code/prefix/'policies_manifest.json')}
    write(receipt,item);return item


def bridge_accounts(actual,reference):
    """Compare economic event fields while ignoring documentary source paths."""
    actual=Path(actual);reference=Path(reference)
    for name in ('daily.json','equity.json'):
        if read(actual/name)!=read(reference/name):raise ValueError('Native account bridge drift '+name)
    def physical(row):
        return {**{k:row[k] for k in ('order_id','strategy_id','symbol','side','qty','price','timestamp','reason')},
            'execution':{k:row['metadata'].get(k) for k in ('execution_phase','base_price','fee','decision_available_at','submitted_at','execution_bar')}}
    if list(map(physical,read(actual/'fills.json')))!=list(map(physical,read(reference/'fills.json'))):
        raise ValueError('Native fill bridge drift')
    return {'passed':True,'reference':str(reference),'equity_marks':len(read(actual/'equity.json')),
        'scope':'Every timestamped mark, daily balance and physical fill matches unchanged parent.'}


def native_support(actual,reference,policy,parent_policy):
    """Real filled additions; removals remain paired observations, not fake probes."""
    def admissions(folder):
        import json
        with (Path(folder)/(POLICY+'.admission.jsonl')).open(encoding='utf-8') as f:
            return {r['opportunity_id']:r for line in f if (r:=json.loads(line))['reason']=='admitted'}
    a,p=admissions(policy),admissions(parent_policy)
    added=set(a)-set(p);removed=set(p)-set(a)
    def entries(folder):
        result={}
        for t in read(Path(folder)/'trades.json'):
            k=t['entry_decision_time'][:10]+'/'+t['symbol']
            result.setdefault(k,[]).append({f:t[f] for f in ('entry_decision_time','entry_fill_time','qty','net_pnl','exit_reason')})
        return result
    actual_trades,parent_trades=entries(actual),entries(reference)
    def opening_fill(key):
        # An issuer admitted at opening may only execute at a later checkpoint.
        # Match the candidate's own entry phase from actual fill metadata below.
        return key in opening_fills
    opening_fills=set()
    for fill in read(Path(actual)/'fills.json'):
        if str(fill['side']).upper() not in ('BUY','ORDERSIDE.BUY'):continue
        md=fill.get('metadata',{});route=md.get('entry_route_mode') or md.get('entry_route')
        # Missing route is not promoted into evidence; audit actual native trade context next.
        if route in ('first30_open','first30'):
            opening_fills.add(str(fill['timestamp'])[:10]+'/'+fill['symbol'])
    changes={k:{'parent':p.get(k),'variant':a.get(k),'actual_trades':actual_trades.get(k,[]),
                'parent_trades':parent_trades.get(k,[])} for k in sorted(set(a)|set(p)) if a.get(k)!=p.get(k)}
    return {'scope':'Actual native accounts. Filled added issuer-days are observed exposure, not isolated incremental profit. Removed parent fills and later funding descendants are dependent; no same-state hypothetical fill is credited.',
        'opening_admission_additions':len(added),'opening_admission_removals':len(removed),
        'added_issuer_days_with_actual_trade':len(added&set(actual_trades)),
        'added_issuer_days_absent_from_parent_account_with_actual_trade':len((added&set(actual_trades))-set(parent_trades)),
        'added_issuer_days_with_explicit_opening_fill_route':sum(opening_fill(k) for k in added),
        'removed_admissions_with_parent_trade':len(removed&set(parent_trades)),
        'removed_admissions_still_traded_by_variant':len(removed&set(actual_trades)),
        'same_state_removal_support_proven':False,'changes':changes}

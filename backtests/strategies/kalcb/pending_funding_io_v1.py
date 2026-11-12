"""Opening-only research reservation and immutable native publication."""
from copy import deepcopy
from pathlib import Path
from backtests.auto.shared.cache_keys import stable_signature
from .retention_research_io_v1 import read,write,bound,digest,verify,source_hash


def patch_core(text):
    old='open_notional=(_governed_open_notional(state,portfolio)\n            if config.risk_max_gross_equity_pct else portfolio.open_notional),'
    if text.count(old)!=1:raise ValueError('Original quantity anchor changed')
    text=text.replace(old,'open_notional=portfolio.open_notional,')
    old='    qty = compute_entry_qty(\n'
    if text.count(old)!=1:raise ValueError('Quantity function anchor changed')
    text=text.replace(old,"""    # This registered research overlay owns only opening-route reservation.
    # Remove its switch from the original quantity calculation on every route.
    funding_limit = (float(config.risk_max_gross_equity_pct)
        if entry_meta.get('entry_route') == 'first30_open_anchor' else 0.0)
    if config.risk_max_gross_equity_pct:
        config = replace(config, risk_max_gross_equity_pct=0.0)
    qty = compute_entry_qty(
""")
    old="    if config.risk_daily_vol_target:\n        qty=int(qty*float(state.meta.get('portfolio_volatility_scale',config.risk_volatility_warmup_mult)))\n    if qty <= 0:"
    if text.count(old)!=1:raise ValueError('Post-scale anchor changed')
    return text.replace(old,"""    if config.risk_daily_vol_target:
        qty=int(qty*float(state.meta.get('portfolio_volatility_scale',config.risk_volatility_warmup_mult)))
    if funding_limit:
        original_qty = qty
        reserved = _governed_open_notional(state, portfolio)
        remaining = max(0.0, float(portfolio.equity or 0.0)*funding_limit-reserved)
        qty = min(qty, int(remaining/bar.close))
        entry_meta['pending_funding_reservation'] = {
            'original_qty':original_qty, 'capped_qty':qty,
            'equity':float(portfolio.equity or 0.0), 'reserved_gross':reserved,
            'remaining_notional':remaining, 'completed_close':bar.close,
            'limit':funding_limit, 'route':'first30_open_anchor'}
    if qty <= 0:""")


def freeze(root,name,period,old):
    from .late_entry_io_v5 import cached_link
    p=read(root/'protocol.json');recipe=p['recipes'][name]
    folder=Path(p['runtime_root'])/'cases'/name/period
    receipt=root/'cases'/name/period/'input_profile.json'
    if receipt.exists():
        item=read(receipt);verify([item[k] for k in ('manifest','plan','policy')]);return item
    verify([old[k] for k in ('manifest','plan','policy')]+list(p['overrides'].values()))
    pp=Path(old['manifest']['path']);parent=read(pp);sig=parent.pop('manifest_hash')
    if sig!=stable_signature(parent):raise ValueError('Changed parent manifest')
    folder.mkdir(parents=True,exist_ok=False)
    plan=read(old['plan']['path']);cfg=read(plan['config']);context=read(plan['audit_configuration'])
    if cfg.get('risk_max_gross_equity_pct',0):raise ValueError('Parent overlay already active')
    if recipe['reservation'] not in (0,2):raise ValueError('Undeclared funding limit')
    if recipe['reservation']:
        cfg['risk_max_gross_equity_pct']=2.0;write(folder/'config.json',cfg)
    else:(folder/'config.json').write_bytes(Path(plan['config']).read_bytes())
    context.update(output_root=str(folder),promotion_status='pending_funding_screening_only_no_promotion')
    write(folder/'audit_configuration.json',context)
    for key in ('config','audit_configuration'):
        plan['input_sha256'].pop(plan[key],None);plan[key]=str(folder/(key+'.json'));plan['input_sha256'][plan[key]]=digest(plan[key])
    plan['pending_funding_research']={'registration':bound(root/'protocol.json'),'recipe':recipe,
        'parent_profile':old,'label_review':p['label_review'],'scope':'Opening quantity only; checkpoints retain original quantity rules.'}
    write(folder/'plan.json',plan)
    replacements={rel:Path(b['path']) for rel,b in p['overrides'].items()}
    replacements.update({'research/kalcb_executed_value/plan.json':folder/'plan.json',
        'research/pending_funding/protocol.json':root/'protocol.json',
        'research/pending_funding/config.json':folder/'config.json',
        'research/pending_funding/audit_configuration.json':folder/'audit_configuration.json'})
    files={};cache=Path(p['runtime_root'])/'content_cache'
    for rel,sha in parent['files'].items():
        if rel in replacements:continue
        cached_link(pp.parent/'files'/rel,folder/'inputs/files'/rel,sha,cache);files[rel]=sha
    for rel,src in replacements.items():
        sha=digest(src);cached_link(src,folder/'inputs/files'/rel,sha,cache);files[rel]=sha
    payload=deepcopy(parent);payload['files']=files
    payload['evidence']['pending_funding_research']={'protocol':bound(root/'protocol.json'),
        'parent_manifest':bound(pp),'parent_policy':old['policy'],'recipe':recipe,'labels':p['label_review']}
    code=folder/'inputs/files';actual=source_hash(code);payload['source_generation_hash']=actual
    write(folder/'inputs/manifest.json',{**payload,'manifest_hash':stable_signature(payload)})
    policy=code/'research/kalcb_executed_value/policies/policies_manifest.json'
    if digest(policy)!=old['policy']['sha256']:raise ValueError('Changed learned policy')
    item={'name':name,'period':period,'folder':str(folder),'source_generation_hash':actual,
        'manifest':bound(folder/'inputs/manifest.json'),'plan':bound(folder/'plan.json'),'policy':bound(policy)}
    write(receipt,item);return item

"""Authenticate a full-horizon producer runtime and completed native controls."""
from datetime import date
from pathlib import Path
import json
import yaml
from backtests.strategies.kalcb.retention_research_io_v1 import read, bound, verify, digest, validate_native, source_hash


def calendar_extension(old_code, new_code, first, last):
    old_code,new_code=Path(old_code),Path(new_code)
    old=yaml.safe_load((old_code/'kis_core/data/krx_sessions.yaml').read_text())
    new=yaml.safe_load((new_code/'kis_core/data/krx_sessions.yaml').read_text())
    first,last=date.fromisoformat(first),date.fromisoformat(last)
    if not date.fromisoformat(new['coverage_start'])<=first<=last<=date.fromisoformat(new['coverage_end']):
        raise ValueError('Producer calendar does not cover the full research horizon')
    for key in ('regular','markets','exceptions','symbol_suspensions'):
        if old.get(key)!=new.get(key):raise ValueError('Changed historical calendar rule: '+key)
    if any(new['sources'].get(k)!=v for k,v in old['sources'].items()):
        raise ValueError('Changed historical session source')
    if digest(old_code/'kis_core/data/krx_holidays.yaml')!=digest(new_code/'kis_core/data/krx_holidays.yaml'):
        raise ValueError('Changed holiday history')
    a=yaml.safe_load((old_code/'kis_core/data/krx_intraday_halts.yaml').read_text())
    b=yaml.safe_load((new_code/'kis_core/data/krx_intraday_halts.yaml').read_text())
    through=old['coverage_end']
    for key in ('markets','membership_source','membership_captured_on','membership_latest_listing_date','scope','periodic_auctions','closing_auctions'):
        if a.get(key)!=b.get(key):raise ValueError('Changed historical matching rule: '+key)
    past=lambda rows:sorted(json.dumps(row,sort_keys=True) for row in rows if row['date']<=through)
    if past(a['halts'])!=past(b['halts']):raise ValueError('Changed historical halt history')
    if any(b['evidence_files'].get(k)!=v for k,v in a['evidence_files'].items()):
        raise ValueError('Changed retained matching evidence')
    verify([{'path':str(new_code/p),'sha256':sha} for p,sha in b['evidence_files'].items()])
    return {'passed':True,'old_through':through,'producer_through':new['coverage_end'],
            'historical_rules_unchanged':True,'new_halts':len(b['halts'])-len(a['halts']),
            'calendar':bound(new_code/'kis_core/data/krx_sessions.yaml'),
            'matching_regimes':bound(new_code/'kis_core/data/krx_intraday_halts.yaml')}


def authenticate_predecessor(root):
    root=Path(root);protocol=read(root/'protocol.json');gate=read(root/'preflight/completion.json')
    verify(protocol['programs']+protocol['bindings']+protocol['data_bindings'])
    if not gate['passed'] or gate['native_identity_bridges']!=4 or any(gate['fresh_filled_labels'][s].get('filled',0)<1 for s in ('baseline','combined')):
        raise ValueError('Incomplete predecessor readiness gates')
    bindings=[bound(root/p) for p in ('protocol.json','preflight/completion.json','failed_stage.json','prepare.log','progress.json')]
    results=[];profiles={}
    from backtests.strategies.kalcb.qualified_artifacts import REPLAY_FILES
    for prior in (100,125):
        for period in ('is','oos'):
            folder=root/f'cases/identity_{prior}/{period}';item=read(folder/'input_profile.json');out=folder/'baseline'
            checked=validate_native(out,item,'baseline',save=False)
            bridge=read(out/'identity_bridge.json')
            if not bridge['passed']:raise ValueError('Unverified parent identity bridge')
            bindings+=[bound(folder/'input_profile.json')]+[bound(out/f) for f in (*REPLAY_FILES,'independent_account_verification.json','identity_bridge.json')]
            bindings+=[item[k] for k in ('plan','manifest','policy')]
            results.append({'prior':prior,'period':period,'metrics':checked,'bridge':bridge,'profile':item})
            if prior==100:profiles[period]=item
    old_code=Path(profiles['is']['folder'])/'inputs/files';code=Path(profiles['oos']['folder'])/'inputs/files'
    old_manifest=read(profiles['is']['manifest']['path']);manifest=read(profiles['oos']['manifest']['path'])
    execution=lambda files:{p:sha for p,sha in files.items() if p.endswith('.py') and p.split('/')[0] in ('backtests','strategy_common','strategy_kalcb','strategy_olr','strategy_pcim','deployment','instrumentation','oms','oms_client','kis_core','scripts')}
    a,b=execution(old_manifest['files']),execution(manifest['files'])
    changed={p for p in set(a)|set(b) if a.get(p)!=b.get(p)}
    if changed!={'scripts/update_kis_intraday.py'}:raise ValueError('Unexpected execution-code correspondence')
    # The one differing script is a data updater, never imported by a producer.
    verify([{'path':str(code/p),'sha256':sha} for p,sha in b.items()])
    if source_hash(code)!=profiles['oos']['source_generation_hash']:raise ValueError('Changed producer source generation')
    calendar=calendar_extension(old_code,code,'2025-05-12','2026-07-10')
    return {'passed':True,'root':str(root),'results':results,'bindings':bindings,
            'producer_profile':profiles['oos'],'producer_code':str(code),'calendar':calendar,
            'code_correspondence':{'changed':sorted(changed),'scope':'Only the nonexecuted data updater differs; strategy, broker and producer modules are byte-identical.'}}

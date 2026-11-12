"""Finite, outcome-free pending-funding intervention census in a frozen core.

Registration and publication occur before any new preview. This is not a
portfolio replay, label producer, or fitted model. Changed quantities are
exported for fresh native full-cost labels before any economic trial.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
BASE = Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,default=str,allow_nan=False),encoding='utf-8')


def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def bound(p):
    return {'path':str(Path(p).resolve()),'sha256':digest(p)}


def verify(items):
    for x in items:
        if digest(x['path'])!=x['sha256']:raise ValueError('Changed binding: '+x['path'])


def patch_core(text):
    old='open_notional=(_governed_open_notional(state,portfolio)\n            if config.risk_max_gross_equity_pct else portfolio.open_notional),'
    if text.count(old)!=1:raise ValueError('Nonunique original quantity anchor')
    # Do not shrink the learned action twice by reducing its pre-scale base.
    text=text.replace(old,'open_notional=portfolio.open_notional,')
    old="    if config.risk_daily_vol_target:\n        qty=int(qty*float(state.meta.get('portfolio_volatility_scale',config.risk_volatility_warmup_mult)))\n    if qty <= 0:"
    if text.count(old)!=1:raise ValueError('Nonunique post-scale anchor')
    new="""    if config.risk_daily_vol_target:
        qty=int(qty*float(state.meta.get('portfolio_volatility_scale',config.risk_volatility_warmup_mult)))
    if config.risk_max_gross_equity_pct:
        # Research-only frozen post-scale reservation. The actual broker still
        # checks next-bar prices, costs and available funding independently.
        reserved_gross = _governed_open_notional(state, portfolio)
        remaining = max(0.0, float(portfolio.equity or 0.0) * config.risk_max_gross_equity_pct - reserved_gross)
        qty = min(qty, int(remaining / bar.close))
    if qty <= 0:"""
    return text.replace(old,new)


def register_original(root, tests):
    sys.path.insert(0,str(REPO))
    from backtests.auto.shared.cache_keys import stable_signature
    from backtests.strategies.kalcb.late_entry_io_v5 import cached_link
    from backtests.strategies.kalcb.research_process_v1 import exact_process
    parent=BASE/'v21_parent_observers'
    for path in (parent/'supervisor_start.json',parent/'assessment_launch_v1/supervisor_start.json'):
        if exact_process(read(path)):raise ValueError('Parent work is still active')
    completion=read(parent/'assessment_v1/completion.json');verify([completion['assessment']])
    if not completion['complete'] or completion['exact_bridges']!=8:raise ValueError('Incomplete checkpoint')
    receipt=read(tests)
    if receipt['exit_code']!=0:raise ValueError('Tests must pass')
    verify(receipt['programs']+[receipt['log']])
    programs=[bound(__file__),bound(REPO/'tests/backtests/strategies/test_kalcb_pending_funding_preflight_v1.py')]
    cases=[];bindings=[bound(parent/'assessment_v1/completion.json'),completion['assessment'],
        bound(parent/'economic_checkpoint_v1/review.json'),bound(tests)]
    for name in ('control','control_prior125'):
        for period in ('is','oos'):
            profile=read(parent/'cases'/name/period/'input_profile.json')
            plan=read(profile['plan']['path'])
            snapshots=Path(profile['policy']['path']).parent/'interaction_cost_margin.snapshots.json'
            bindings += [profile[k] for k in ('manifest','plan','policy')]+[bound(plan['config']),bound(snapshots)]
            for scenario in ('baseline','combined'):
                case=parent/'cases'/name/period/scenario
                if not read(case/'identity_bridge.json')['passed'] or not read(case/'independent_account_verification.json')['passed']:
                    raise ValueError('Unaudited parent case')
                item={'name':name,'period':period,'scenario':scenario,'profile':profile,
                    'snapshot':bound(snapshots),'evidence':bound(case/'observer_evidence.json'),
                    'account':bound(parent/'assessment_v1/accounts'/name/period/(scenario+'.json'))}
                cases.append(item);bindings += [item['evidence'],item['account'],bound(case/'reclaim_source_proof.json')]
    data=read(BASE/'v18_reclaim_labels/protocol.json')['databases']
    bindings += [bound(p) for p in data]
    runtime=cases[2]['profile'];original=read(runtime['manifest']['path']);signature=original.pop('manifest_hash')
    if signature!=stable_signature(original):raise ValueError('Invalid source runtime')
    root.mkdir(parents=True,exist_ok=False)
    code=root/'runtime/inputs/files';files={};cache=root/'runtime/content_cache'
    source=Path(runtime['manifest']['path']).parent/'files'
    override=root/'overrides/strategy_kalcb/core/logic.py';override.parent.mkdir(parents=True)
    override.write_text(patch_core((source/'strategy_kalcb/core/logic.py').read_text(encoding='utf-8')),encoding='utf-8')
    for rel,sha in original['files'].items():
        src=override if rel=='strategy_kalcb/core/logic.py' else source/rel
        if src==override:sha=digest(src)
        cached_link(src,code/rel,sha,cache);files[rel]=sha
    manifest={'files':files,'parent_manifest':runtime['manifest'],'source_override':bound(override)}
    write(root/'runtime/inputs/manifest.json',manifest)
    batches=sum(len({r['decision_available_at'] for r in read(c['evidence']['path'])['entry_exposures']
        if r['action']['metadata']['entry_route']=='first30_open_anchor'}) for c in cases)
    p={'version':'pending-funding-postscale-preflight-v1','registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'programs':programs,'bindings':bindings,'cases':cases,'databases':data,'runtime':str(code),
        'runtime_manifest':bound(root/'runtime/inputs/manifest.json'),'tests':bound(tests),
        'control_config_change':{},'candidate_config_change':{'risk_max_gross_equity_pct':2.0},
        'formula':'Compute the unchanged parent quantity first, with actual portfolio.open_notional, then cap at floor(max(0,2*equity-governed_open_notional(state,portfolio))/current_completed_close). governed_open_notional includes actual marked positions and unfilled pending-entry reservations, net of confirmed partial fills. No pro-rata scaling or new rank.',
        'scope':'All observed opening submission batches in eight audited accounts; outcome-free copied prefix previews. Native quantity/selection/funding support only, not fills, profitable activity, account returns or labels.',
        'population_selection':'Every first30_open_anchor submission batch; no date/issuer/outcome or overbooking filter.',
        'counts':{'parent_accounts':8,'opening_batches':batches,'core_previews':2*batches,'portfolio_cases':0},
        'runtime_estimate':'1-5 minutes for outcome-free prefixes after publication, then inspect support. No additional portfolio grid is registered.',
        'next_gate':'Exact disabled-control order/symbol/quantity/clock reconstruction on every batch; retained zero-quantity blocks and all changed positive-quantity requests. Fresh independent full-cost action labels, matching native source/config contracts and eight exact whole-account disabled bridges required before finite portfolio replay.',
        'invariants':'Unchanged physical .001, 48bps forecast reserve, checkpoint count, earlier snapshots, costs, next-bar/gap broker funding and all mature history/clocks. No fitting or return forecast changes. April-July reused development. No live or canonical changes.'}
    write(root/'protocol.json',p);write(root/'study.json',{'study_complete':False,'data_only':True,'counts':p['counts']})
    print(json.dumps({'registered':str(root),'counts':p['counts']}),flush=True)


def register(root, tests):
    # The only producer fix is path construction for origin receipts. Reuse the
    # completed immutable publication, not any failed/missing preview output.
    import psutil
    parent=BASE/'v22_pending_funding_preflight';p=read(parent/'protocol.json')
    identity=read(parent/'launch_receipt.json')
    try:
        proc=psutil.Process(identity['pid'])
        if abs(proc.create_time()-identity['process_created'])<.01 and proc.cmdline()==identity['command']:
            raise ValueError('Predecessor still alive')
    except psutil.NoSuchProcess:pass
    if read(parent/'status.json')['stage']!='attention_required':raise ValueError('Unexpected predecessor')
    if (parent/'prepared').exists():raise ValueError('Do not reuse failed preview outputs')
    verify(p['programs']+p['bindings']+[p['runtime_manifest']])
    test=read(tests)
    if test['exit_code']!=0:raise ValueError('Tests must pass')
    verify(test['programs']+[test['log']])
    code=Path(p['runtime']);manifest=read(p['runtime_manifest']['path'])
    for rel,sha in manifest['files'].items():
        if digest(code/rel)!=sha:raise ValueError('Changed frozen publication: '+rel)
    root.mkdir(parents=True,exist_ok=False)
    p['version']='pending-funding-postscale-preflight-v2'
    p['registered_at_utc']=datetime.now(timezone.utc).isoformat()
    p['programs']=[bound(__file__),bound(REPO/'tests/backtests/strategies/test_kalcb_pending_funding_preflight_v2.py')]
    p['bindings']+=test['programs']+[test['log'],bound(tests),bound(parent/'protocol.json'),bound(parent/'failed_stage.json'),bound(parent/'prepare.log')]
    p['tests']=bound(tests)
    p['authenticated_reuse']={'parent':str(parent),'runtime_files':len(manifest['files']),
        'runtime_manifest':p['runtime_manifest'],'unchanged_economic_source':manifest['source_override'],
        'failed_output_reused':False,'repair':'Parenthesize receipt filename before WindowsPath division. No economic/formula/population change.'}
    write(root/'protocol.json',p);write(root/'study.json',{'study_complete':False,'data_only':True,'counts':p['counts']})
    write(root/'reuse_receipt.json',p['authenticated_reuse'])
    print(json.dumps({'registered':str(root),'counts':p['counts'],'authenticated_files':len(manifest['files'])}),flush=True)


def bootstrap(root):
    p=read(root/'protocol.json');verify(p['programs']+p['bindings']+[p['runtime_manifest']])
    prefixes=('backtests','scripts','kis_core','strategy_kalcb','strategy_common')
    if any(k.split('.')[0] in prefixes for k in sys.modules):raise ValueError('Application imported before isolation')
    code=Path(p['runtime']).resolve()
    sys.path[:]=[str(code)]+[x for x in sys.path if x and not Path(x).resolve().is_relative_to(REPO)]
    manifest=read(p['runtime_manifest']['path'])
    def origins():
        rows={}
        for name,module in list(sys.modules.items()):
            path=getattr(module,'__file__',None)
            if name.split('.')[0] not in prefixes or not path:continue
            path=Path(path).resolve()
            if not path.is_relative_to(code):raise ValueError('Frozen import escaped: '+name)
            rel=path.relative_to(code).as_posix();sha=manifest['files'].get(rel)
            if not sha or digest(path)!=sha:raise ValueError('Frozen module hash mismatch: '+name)
            rows[name]={'path':str(path),'sha256':sha}
        return {'checked_modules':len(rows),'origins':rows}
    return p,origins


def prepare(root, smoke=False):
    p,origins=bootstrap(root)
    from dataclasses import replace
    import sqlite3
    from strategy_kalcb.models import KALCBDailySnapshot
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    from strategy_kalcb.core.preview import preview_entry
    from strategy_common.market import MarketBar
    from scripts.kalcb_executed_value_study import config_for
    write(root/(('smoke' if smoke else 'prepare')+'_imports_before.json'),origins())
    results=[];requests=[];number=0;began=time.monotonic()
    for case in (p['cases'][:1] if smoke else p['cases']):
        profile=case['profile'];cfg=config_for(read(profile['plan']['path']),case['scenario'])
        if cfg.risk_max_gross_equity_pct or cfg.risk_max_single_equity_pct or cfg.risk_daily_vol_target:
            raise ValueError('Unexpected parent active exposure overlay')
        snapshots={r['trade_date']:r for r in read(case['snapshot']['path'])}
        evidence=read(case['evidence']['path']);groups=defaultdict(list)
        for row in evidence['entry_exposures']:
            if row['action']['metadata']['entry_route']=='first30_open_anchor':groups[row['decision_available_at']].append(row)
        db=p['databases'][int(case['period']=='oos')]
        with sqlite3.connect('file:'+Path(db).as_posix()+'?mode=ro',uri=True) as conn:
            pairs=list(sorted(groups.items()))
            if smoke:pairs=pairs[:1]
            for stamp,entries in pairs:
                available=datetime.fromisoformat(stamp);day=available.date().isoformat();raw=snapshots[day]
                snap=KALCBDailySnapshot.from_json_dict(raw)
                symbols={c.symbol for c in snap.candidates}
                bars=[MarketBar(s,datetime.fromisoformat(t),'5m',o,h,l,c,v) for s,t,o,h,l,c,v in conn.execute(
                    'SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day=? ORDER BY ts,symbol',(day,))
                    if s in symbols and datetime.fromisoformat(t)+timedelta(minutes=5)<=available]
                account=KALCBPortfolioView(**entries[0]['account_before']['portfolio'])
                if account.positions or account.open_positions or account.open_notional:
                    raise ValueError('Nonflat opening requires full-state reconstruction')
                # Compare raw completed-prefix summaries to the actual parent
                # decision. Source proof/whole-account audits remain separately bound.
                for row in entries:
                    seq=[b for b in bars if b.symbol==row['action']['symbol']];m=row['action']['metadata']
                    values={'first30_open':seq[0].open,'first30_high':max(b.high for b in seq),
                        'first30_low':min(b.low for b in seq),'first30_close':seq[-1].close,'first30_volume':sum(b.volume for b in seq)}
                    if len(seq)!=6 or any(abs(m[k]-v)>1e-6 for k,v in values.items()):raise ValueError('Raw opening prefix mismatch')
                baseline=preview_entry(bars,snap,cfg,account,available_at=available)
                modified=preview_entry(bars,snap,replace(cfg,risk_max_gross_equity_pct=2.0),account,available_at=available)
                original={(r['action']['symbol'],r['action']['qty']) for r in entries}
                current={(r['symbol'],r['quantity']) for r in baseline['intents']}
                if len(original)!=len(entries) or current!=original or len(current)!=len(baseline['intents']):
                    raise ValueError('Native control order reconstruction mismatch: '+str((case['name'],case['period'],stamp,original,current)))
                changed={r['symbol']:r for r in modified['intents']}
                if set(changed)-{s for s,q in original}:raise ValueError('Funding cap introduced unadmitted issuer')
                diffs=[]
                for row in entries:
                    symbol=row['action']['symbol'];old=row['action']['qty'];new=changed.get(symbol,{}).get('quantity',0)
                    if new>old:raise ValueError('Funding cap increased quantity')
                    if old!=new:
                        req={'name':case['name'],'period':case['period'],'scenario':case['scenario'],'day':day,
                            'symbol':symbol,'available_at':available.isoformat(),'parent_qty':old,'capped_qty':new,
                            'parent_order_id':row['order_id'],'initial_equity':account.equity,'snapshot':case['snapshot'],
                            'reference_price':row['action']['metadata']['entry_price_ref'],
                            'parent_scale':next(c for c in raw['candidates'] if c['symbol']==symbol)['metadata']['action_quantity_scale'],
                            'native_cap_intent':changed.get(symbol),'source_evidence':case['evidence'],
                            'requires_fresh_full_cost_label':new>0,'zero_is_abstention_not_exposure':new==0}
                        requests.append(req);diffs.append({'symbol':symbol,'old_qty':old,'new_qty':new})
                result={'name':case['name'],'period':case['period'],'scenario':case['scenario'],'day':day,
                    'control_exact':True,'parent_orders':len(entries),'candidate_orders':len(changed),
                    'changes':diffs,'candidate_rejections':[r for r in modified['decisions'] if r['reason']=='qty_zero']}
                results.append(result);number+=1
                if number%20==0:print(json.dumps({'opening_batches':number,'expected':p['counts']['opening_batches'],'changed_requests':len(requests)}),flush=True)
    stage='smoke' if smoke else 'prepared';target=root/stage;target.mkdir(exist_ok=False)
    write(target/'batches.json',results);write(target/'label_requests.json',requests)
    summary={'complete':True,'study_complete':False,'data_only':True,'control_exact_batches':number,
        'changed_batches':sum(bool(r['changes']) for r in results),'changed_requests':len(requests),
        'positive_changed_quantity_requests':sum(r['capped_qty']>0 for r in requests),
        'zero_quantity_abstentions':sum(r['capped_qty']==0 for r in requests),'seconds':time.monotonic()-began,
        'batches':bound(target/'batches.json'),'label_requests':bound(target/'label_requests.json'),
        'native_fills_generated':0,'performance_improvement_demonstrated':False,'next':p['next_gate']}
    write(target/'completion.json',summary);write(root/(('smoke' if smoke else 'prepare')+'_imports_after.json'),origins())
    verify(p['programs']+p['bindings']);print(json.dumps(summary),flush=True)


def supervise(root):
    import psutil,subprocess
    sys.path.insert(0,str(REPO))
    from backtests.strategies.kalcb.research_process_v1 import execute
    p=read(root/'protocol.json');verify(p['programs']+p['bindings'])
    if (root/'supervisor_start.json').exists():raise ValueError('Preserve existing attempt')
    for proc in psutil.process_iter(['pid','cmdline']):
        if proc.pid==os.getpid():continue
        args=proc.info['cmdline'] or []
        if any(x.startswith('scripts.kalcb_') and ('worker' in x or 'replay' in x) for x in args):raise ValueError('Native worker already active')
    (root/'temp').mkdir(exist_ok=True);os.environ['TEMP']=os.environ['TMP']=str(root/'temp')
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':psutil.Process().cmdline()}
    write(root/'supervisor_start.json',state)
    try:
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'prepare','--root',str(root)]
        execute(root,command,'prepare',p['runtime'],state)
        summary=read(root/'prepared/completion.json')
        if summary['control_exact_batches']!=p['counts']['opening_batches']:raise ValueError('Incomplete mechanical census')
        write(root/'completion.json',{'complete':True,'study_complete':False,'preflight':bound(root/'prepared/completion.json')})
        write(root/'status.json',{**state,'stage':'awaiting_fresh_quantity_labels_and_economic_registration','child':None})
    except Exception as e:
        write(root/'status.json',{**state,'stage':'attention_required','error':repr(e)});raise


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=('register','run','prepare','smoke'))
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--tests',type=Path)
    a=ap.parse_args();root=a.root.resolve()
    if a.stage=='register':register(root,a.tests)
    elif a.stage=='run':supervise(root)
    else:prepare(root,a.stage=='smoke')

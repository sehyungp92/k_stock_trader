"""Registered finite fresh-quantity data pipeline; no portfolio optimizer."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
MODULE='backtests.strategies.kalcb.pending_funding_labels_v1'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,default=str,allow_nan=False),encoding='utf-8')
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(p):return {'path':str(Path(p).resolve()),'sha256':digest(p)}
def verify(items):
    for v in items:
        if digest(v['path'])!=v['sha256']:raise ValueError('Changed registered input: '+v['path'])


def episode_specs(parent):
    p=read(parent/'protocol.json');requests=read(parent/'prepared/label_requests.json')
    profiles={(c['name'],c['period'],c['scenario']):c['profile'] for c in p['cases']}
    unique={};zeros=[]
    for req in requests:
        if req['capped_qty']==0:zeros.append(req);continue
        if not 0<req['capped_qty']<req['parent_qty']:raise ValueError('Not a positive reduction')
        profile=profiles[req['name'],req['period'],req['scenario']]
        snapshot=next(s for s in read(req['snapshot']['path']) if s['trade_date']==req['day'])
        config=read(profile['plan']['path'])['config']
        for kind,quantity in (('parent',req['parent_qty']),('capped',req['capped_qty'])):
            identity={'scenario':req['scenario'],'day':req['day'],'symbol':req['symbol'],
                'initial_equity':req['initial_equity'],'target_qty':quantity,'snapshot':snapshot,
                'config_sha256':digest(config),'available_at':req['available_at']}
            key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            alias={'name':req['name'],'period':req['period'],'scenario':req['scenario'],
                'day':req['day'],'symbol':req['symbol'],'parent_order_id':req['parent_order_id'],
                'action_kind':kind,'source_evidence':req['source_evidence']}
            if key in unique:
                unique[key]['aliases'].append(alias);continue
            unique[key]={'episode_id':key[:24],**{k:req[k] for k in ('day','symbol','scenario','available_at',
                'parent_qty','initial_equity','reference_price','snapshot')},'target_qty':quantity,
                'action_kind':kind,'profile':profile,'aliases':[alias]}
    return list(unique.values()),zeros


def register(root,tests):
    sys.path.insert(0,str(REPO))
    from backtests.strategies.kalcb.research_process_v1 import exact_process
    parent=BASE/'v23_pending_funding_preflight';p=read(parent/'protocol.json')
    if exact_process(read(parent/'supervisor_start.json')):raise ValueError('Preflight is still active')
    summary=read(parent/'prepared/completion.json')
    if not read(parent/'completion.json')['complete'] or summary['control_exact_batches']!=544:
        raise ValueError('Incomplete mechanical preflight')
    verify(p['programs']+p['bindings']+[p['runtime_manifest'],summary['batches'],summary['label_requests']])
    t=read(tests)
    if t['exit_code']!=0:raise ValueError('Tests must pass')
    verify(t['programs']+[t['log']])
    specs,zeros=episode_specs(parent)
    if len(specs)>124 or len(zeros)!=40:raise ValueError('Undeclared expanded label population')
    root.mkdir(parents=True,exist_ok=False);write(root/'episodes.json',specs);write(root/'zero_abstentions.json',zeros)
    programs=[bound(__file__),bound(REPO/'backtests/strategies/kalcb/pending_funding_labels_v1.py'),
        bound(REPO/'tests/backtests/strategies/test_kalcb_pending_funding_labels_v1.py'),
        bound(REPO/'backtests/strategies/kalcb/research_process_v1.py')]
    bindings=p['bindings']+[bound(parent/'protocol.json'),bound(parent/'completion.json'),
        bound(parent/'prepared/completion.json'),summary['batches'],summary['label_requests'],
        bound(parent/'prepare_imports_before.json'),bound(parent/'prepare_imports_after.json'),bound(tests)]
    protocol={'version':'pending-funding-quantity-labels-v1','registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'programs':programs,'bindings':bindings,'runtime':p['runtime'],'runtime_manifest':p['runtime_manifest'],
        'databases':p['databases'],'episodes':bound(root/'episodes.json'),'abstentions':bound(root/'zero_abstentions.json'),
        'tests':t,'episodes_planned':len(specs),'logical_episode_aliases':sum(len(v['aliases']) for v in specs),
        'two_historical_smoke_episodes_authenticated_into_full_output_without_replay':True,
        'source_preflight':str(parent),'mechanism':p['formula'],'portfolio_trials':0,
        'label_contract':'Independently replay each positive capped quantity and its exact original quantity through the unchanged native episode/core/broker. Initial isolated capital matches the actual parent decision basis; full parent snapshot membership/rank retained. Derive an exact reduction-only native quantity scale, verify original and reduced intent quantities before replay. This is not a daily portfolio reset or compounding rescale.',
        'non_selection':'Every positive changed quantity from the frozen544batch census, no future-outcome/issuer/date choice. Zero quantities are retained abstentions and cannot inflate physical exposure support. Exact episode aliases retained, not independent evidence.',
        'scope':'DATA ONLY. Fresh full-cost labels and broker/source/account execution checks, not funded portfolio results, payoff refit or new policies. Whole-portfolio audit_status remains unverified without its manifest.',
        'runtime_estimate':'5-10 minutes for at most124 short native episodes plus source authentication; one worker,384MiB physical/1280MiB Windowscommit headroom.',
        'next':'After exact supervisor/child exit authenticate all labels/journals/origins; preserve losses/no-fills/failures, assess whether reservation only creates tiny orders or useful funded support. Any portfolio screening needs a finite registered design/runtime estimate, shared PhaseRunner,8 fresh disabled-overlay native whole-account bridges, strict source/account lineage. No broad grid or promotion.'}
    protocol['bindings'] += [protocol['episodes'],protocol['abstentions']]
    write(root/'protocol.json',protocol);write(root/'study.json',{'data_only':True,'episodes':len(specs),'study_complete':False})
    print(json.dumps({'registered':str(root),'episodes':len(specs),'logical_episode_aliases':protocol['logical_episode_aliases'],'zero_abstentions':len(zeros)}),flush=True)


def produce(root,smoke):
    p=read(root/'protocol.json');verify(p['programs']+p['bindings']+[p['runtime_manifest']])
    prefixes=('backtests','scripts','kis_core','strategy_kalcb','strategy_common')
    if any(k.split('.')[0] in prefixes for k in sys.modules):raise ValueError('Application imported before isolation')
    code=Path(p['runtime']).resolve();sys.path[:]=[str(code)]+[x for x in sys.path if x and not Path(x).resolve().is_relative_to(REPO)]
    source=next(v for v in p['programs'] if Path(v['path']).name=='pending_funding_labels_v1.py' and 'backtests' in Path(v['path']).parts)
    spec=importlib.util.spec_from_file_location(MODULE,source['path']);module=importlib.util.module_from_spec(spec)
    sys.modules[MODULE]=module;spec.loader.exec_module(module)
    manifest=read(p['runtime_manifest']['path'])
    def origins():
        rows={}
        for name,obj in list(sys.modules.items()):
            file=getattr(obj,'__file__',None)
            if not file or name.split('.')[0] not in prefixes:continue
            path=Path(file).resolve()
            if name==MODULE:
                if path!=Path(source['path']).resolve() or digest(path)!=source['sha256']:raise ValueError('Changed label module')
                rows[name]={**source,'registered_external':True};continue
            if not path.is_relative_to(code):raise ValueError('Escaped frozen import: '+name)
            sha=manifest['files'].get(path.relative_to(code).as_posix())
            if not sha or digest(path)!=sha:raise ValueError('Changed frozen import: '+name)
            rows[name]={'path':str(path),'sha256':sha}
        return {'checked_modules':len(rows),'origins':rows}
    stage='smoke' if smoke else 'labels'
    write(root/(stage+'_imports_before.json'),origins());module.run(root,smoke)
    write(root/(stage+'_imports_after.json'),origins());verify(p['programs']+p['bindings'])


def run(root):
    import psutil
    sys.path.insert(0,str(REPO))
    from backtests.strategies.kalcb.research_process_v1 import execute
    p=read(root/'protocol.json');verify(p['programs']+p['bindings'])
    if (root/'supervisor_start.json').exists():raise ValueError('Preserve prior attempt')
    for proc in psutil.process_iter(['pid','cmdline','name']):
        args=proc.info['cmdline'] or []
        module_worker=any(x.startswith('scripts.kalcb_') and ('worker' in x or 'replay' in x) for x in args)
        data_worker=any(Path(x).name.startswith('kalcb_') and x.endswith('.py') for x in args) and any(x in ('prepare','produce','smoke','label','labels_baseline','labels_combined') for x in args)
        if proc.pid!=os.getpid() and 'python' in (proc.info['name'] or '').lower() and (module_worker or data_worker):raise ValueError('Existing native/data worker')
    (root/'temp').mkdir(exist_ok=True);os.environ['TEMP']=os.environ['TMP']=str(root/'temp')
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':psutil.Process().cmdline()}
    write(root/'supervisor_start.json',state)
    try:
        for stage in ('smoke','produce'):
            execute(root,[sys.executable,'-B','-u',str(Path(__file__).resolve()),stage,'--root',str(root)],stage,p['runtime'],state)
            if stage=='smoke':
                receipt=read(root/'smoke/completion.json')
                if receipt['episodes']!=2 or receipt['counts'].get('filled')!=2 or any(receipt['counts'].get(k,0) for k in ('physical_failures','deadline_violations','independent_execution_failures')):
                    raise ValueError('Historical filled smoke gate failed; preserve outcomes')
        done=read(root/'labels/completion.json')
        if done['episodes']!=p['episodes_planned']:raise ValueError('Incomplete label set')
        write(root/'completion.json',{'complete':True,'study_complete':False,'labels':bound(root/'labels/completion.json')})
        write(root/'status.json',{**state,'stage':'awaiting_label_review_before_any_portfolio_registration','child':None})
    except Exception as e:
        write(root/'status.json',{**state,'stage':'attention_required','error':repr(e)});raise


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=('register','run','smoke','produce'))
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--tests',type=Path)
    a=ap.parse_args();root=a.root.resolve()
    if a.stage=='register':register(root,a.tests)
    elif a.stage=='run':run(root)
    else:produce(root,a.stage=='smoke')

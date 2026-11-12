"""Finite paired native label producer. Bootstrap stdlib first, frozen imports."""
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
MODULE='backtests.strategies.kalcb.checkpoint_contrast_labels_v1'
HELPER=REPO/'backtests/strategies/kalcb/checkpoint_contrast_labels_v1.py'
TEST=REPO/'tests/backtests/strategies/test_kalcb_checkpoint_contrast_labels_v1.py'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,default=str,allow_nan=False),encoding='utf-8')
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(p):return {'path':str(Path(p).resolve()),'sha256':digest(p)}
def verify(items):
    for v in items:
        if digest(v['path'])!=v['sha256']:raise ValueError('Changed registered input: '+v['path'])


def register(root,tests):
    import psutil
    parent=BASE/'v31_checkpoint_context_support';p=read(parent/'protocol.json')
    identity=read(parent/'supervisor_start.json')
    try:
        proc=psutil.Process(identity['pid'])
        if abs(proc.create_time()-identity['creation_epoch'])<.01 and proc.cmdline()==identity['command']:
            raise ValueError('Source census still running')
    except psutil.NoSuchProcess:pass
    if not read(parent/'completion.json')['complete'] or not read(parent/'review_v1/completion.json')['complete']:
        raise ValueError('Source census/review incomplete')
    t=read(tests)
    if t['exit_code']!=0:raise ValueError('Tests must pass')
    verify(t['programs']+[t['log']]);verify(p['programs']+p['bindings'])
    old=read(BASE/'v18_reclaim_labels/protocol.json');runtime=old['producer_runtime']
    with gzip.open(parent/'contexts.jsonl.gz','rt',encoding='utf-8') as f:
        rows=[json.loads(line) for line in f]
    supported=[r for r in rows if r['context_supported']]
    if len(rows)!=7676 or len(supported)!=6418:raise ValueError('Changed finite source population')
    programs=[bound(__file__),bound(HELPER),bound(TEST),bound(REPO/'backtests/strategies/kalcb/research_process_v1.py'),
        bound(REPO/'backtests/strategies/kalcb/retention_research_io_v1.py')]
    bindings=old['data_bindings']+[bound(parent/'protocol.json'),bound(parent/'completion.json'),
        bound(parent/'review_v1/review.json'),bound(parent/'review_v1/completion.json'),bound(parent/'contexts.jsonl.gz'),
        bound(tests),runtime['manifest'],runtime['plan'],bound(read(runtime['plan']['path'])['config'])]
    bindings=list({b['path']:b for b in bindings}.values());verify(bindings)
    root.mkdir(parents=True,exist_ok=False)
    protocol={'version':'completed45-native-hold-liquidate-labels-v1','registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'programs':programs,'bindings':bindings,'runtime':runtime['code'],'runtime_manifest':runtime['manifest'],
        'plan':runtime['plan'],'contexts':bound(parent/'contexts.jsonl.gz'),'opening_prepared':p['opening_prepared'],
        'databases':p['databases'],'quantities':[.25,.5,1.0],'scenarios':['baseline','combined'],
        'issuer_days':len(supported),'planned_pairs':len(supported)*6,'planned_arm_episodes':len(supported)*12,
        'action':'Exactly one contrast at completed opening+45 minutes, AFTER unchanged parent bar and timer decisions. HOLD unchanged native management; LIQUIDATE only an actually held first30_open_anchor position with no existing complete-parent exit. Cancel target pending orders, retain entry late-fill reconciliation, then shared-core SubmitExit MARKET for actual held quantity. Native next-bar/gap/capacity/funding/cost execution unchanged.',
        'population':'Every context-supported original causal opening issuer-day, all three EXISTING native quantity doses and both existing scenarios. No outcomes, future membership, selected winning dates or issuer exclusions. Unknown pre-IS parent selection remains unknown; these labels do not assert funded selected-account support or exchangeability.',
        'target':'Exact (liquidate total native net KRW minus hold total native net KRW)/(actual held shares at checkpoint * completed checkpoint close)*10000. Same complete pre-action state and fills/orders/cash identity required. Mature only after BOTH arms finish with reconciled valid labels; no-fills, old exits, losses, failures and dependent quantity/scenario pairs retained.',
        'source_context':p['formulas'],'history':old['history'],
        'smoke':'First full source session, every supported issuer and every declared quantity/scenario. Additional unhooked original HOLD replay per smoke pair must exactly bridge fills/orders/cash/equity. All smoke pair artifacts hash-authenticated and hardlinked into production, not replayed. Synthetic fixture tests are not economic evidence.',
        'gates':'Before full labels, both scenarios must contain a genuinely new filled exit, all smoke exact parent/prestate bridges and independent source/execution/cash/fee/inventory/rawMTM checks pass. Measured throughput published. Exactly one native worker;384MiB physical/1280MiB Windows commit headroom.',
        'post_label':'After exact exits authenticate all artifacts/origins, chronological mature held-state support and conditional comparability to both unchanged parents. Only then consider a separately preregistered own-state versus new-context economic model contrast. Original60session minimum,20session clock,1session embargo. No parameter sweep, automatic fit, threshold relaxation or portfolio trials.',
        'runtime_estimate':'Measured first-source-session smoke precedes full production; extrapolation and caveats written to startup_health.json. One fixed clock/action contrast, not a timing or quantity search.',
        'tests':t,'models':0,'policies':0,'portfolio_cases':0,'study_complete':False,'production_enabled':False}
    write(root/'protocol.json',protocol);write(root/'study.json',{'data_only':True,'paired_contrast':True,'planned_pairs':protocol['planned_pairs'],'study_complete':False})
    print(json.dumps({'registered':str(root),'planned_pairs':protocol['planned_pairs']}),flush=True)


def produce(root,smoke):
    p=read(root/'protocol.json');verify(p['programs']+p['bindings'])
    prefixes=('backtests','scripts','kis_core','strategy_kalcb','strategy_common')
    if any(k.split('.')[0] in prefixes for k in sys.modules):raise ValueError('Application imported before isolation')
    code=Path(p['runtime']).resolve();sys.path[:]=[str(code)]+[x for x in sys.path if x and not Path(x).resolve().is_relative_to(REPO)]
    source=next(v for v in p['programs'] if Path(v['path']).resolve()==HELPER.resolve())
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
    # Force all application imports required by the episode before the first replay.
    import scripts.kalcb_action_quantity_labels
    import scripts.kalcb_executed_value_replay
    import backtests.engine.integrity
    import backtests.strategies.kalcb.runner
    import strategy_kalcb.core.serializers
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
        native=any(x.startswith('scripts.kalcb_') and ('worker' in x or 'replay' in x) for x in args)
        data=any(Path(x).name.startswith('kalcb_') and x.endswith('.py') for x in args) and any(x in ('produce','smoke','label','labels_baseline','labels_combined') for x in args)
        if proc.pid!=os.getpid() and 'python' in (proc.info['name'] or '').lower() and (native or data):raise ValueError('Existing native/data worker')
    (root/'temp').mkdir(exist_ok=True);os.environ['TEMP']=os.environ['TMP']=str(root/'temp')
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':psutil.Process().cmdline()}
    write(root/'supervisor_start.json',state)
    try:
        for stage in ('smoke','produce'):
            execute(root,[sys.executable,'-I','-B','-u',str(Path(__file__).resolve()),stage,'--root',str(root)],stage,p['runtime'],state)
            if stage=='smoke':
                done=read(root/'smoke/completion.json');counts=done['counts']
                if not counts.get('valid_pairs') or any(counts.get(k,0) for k in ('execution_failures','account_failures','deadline_violations')):
                    raise ValueError('Smoke native/account gate failed; preserve outputs')
                exits={s:0 for s in p['scenarios']}
                for a in done['artifacts']:
                    if '.events.' in a['path']:continue
                    with gzip.open(a['path'],'rt',encoding='utf-8') as f:
                        for line in f:
                            v=json.loads(line)
                            if not v['exact_predecision_identity'] or not v['original_unhooked_hold_bridge']:raise ValueError('Missing smoke bridges')
                            exits[v['scenario']]+=bool(v['liquidate']['actual_changed_exit_fills'])
                if not all(exits.values()):raise ValueError('No actual new exit fills in a smoke scenario')
                rate=done['pairs']/done['elapsed_seconds']
                write(root/'startup_health.json',{'passed':True,'supervisor':state,'smoke':bound(root/'smoke/completion.json'),
                    'origin_before':bound(root/'smoke_imports_before.json'),'origin_after':bound(root/'smoke_imports_after.json'),
                    'new_exit_pair_counts':exits,'measured_smoke_pairs_per_second':rate,
                    'conservative_full_minutes_at_smoke_rate':p['planned_pairs']/rate/60,
                    'estimate_note':'Smoke includes three arms per pair rather than two; source verification/publication overhead and machine load vary. Not a whole-study ETA.',
                    'registered_programs':len(p['programs']),'registered_bindings':len(p['bindings'])})
        done=read(root/'labels/completion.json')
        if done['pairs']!=p['planned_pairs']:raise ValueError('Incomplete finite pair population')
        write(root/'completion.json',{'complete':True,'study_complete':False,'labels':bound(root/'labels/completion.json')})
        write(root/'status.json',{**state,'stage':'awaiting_independent_pair_review_before_any_fit','child':None})
    except Exception as e:
        write(root/'status.json',{**state,'stage':'attention_required','error':repr(e)});raise


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=('register','run','smoke','produce'))
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--tests',type=Path)
    a=ap.parse_args();root=a.root.resolve()
    if a.stage=='register':register(root,a.tests)
    elif a.stage=='run':run(root)
    else:produce(root,a.stage=='smoke')

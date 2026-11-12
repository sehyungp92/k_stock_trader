"""Versioned collection-gate/source-reader successor; native economics unchanged."""
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


def smoke_gate(done,review):
    if not review['passed'] or review['pairs']!=222 or review['new_action_deadline_failures']!=0:
        raise ValueError('Unreconciled or new-action startup failure')
    if any(done['counts'].get(k,0) for k in ('execution_failures','account_failures')):
        raise ValueError('Native execution/account failure')
    if review['inherited_parent_deadline_failures']!=done['counts'].get('deadline_violations',0):
        raise ValueError('Unexplained deadline failure')


def register(root,tests):
    import psutil
    oldroot=BASE/'v32_checkpoint_contrast_labels';old=read(oldroot/'protocol.json')
    for name in ('supervisor_start.json','launch_receipt.json'):
        identity=read(oldroot/name)
        try:
            proc=psutil.Process(identity['pid'])
            if abs(proc.create_time()-identity['process_created'])<.01 and proc.cmdline()==identity['command']:
                raise ValueError('Predecessor still running')
        except psutil.NoSuchProcess:pass
    if read(oldroot/'status.json')['stage']!='attention_required':raise ValueError('Expected preserved collection-gate failure')
    review=read(oldroot/'failure_review_v1/review.json');done=read(oldroot/'smoke/completion.json')
    smoke_gate(done,review)
    verify(old['programs']+old['bindings']+done['artifacts'])
    t=read(tests);verify(t['programs']+[t['log']])
    if t['exit_code']!=0:raise ValueError('Successor tests failed')
    programs=list({v['path']:v for v in old['programs']+t['programs']+[bound(__file__)]}.values())
    bindings=old['bindings']+[bound(oldroot/'protocol.json'),bound(oldroot/'status.json'),
        bound(oldroot/'failure_review_v1/registration.json'),bound(oldroot/'failure_review_v1/completion.json'),
        bound(oldroot/'failure_review_v1/review.json'),bound(oldroot/'smoke/completion.json'),
        bound(oldroot/'smoke_imports_before.json'),bound(oldroot/'smoke_imports_after.json'),
        bound(BASE/'v32_checkpoint_contrast_preflight/frozen_source_reader_benchmark_v2.json')]
    verify(bindings)
    root.mkdir(parents=True,exist_ok=False);(root/'smoke').mkdir()
    artifacts=[]
    for binding in done['artifacts']:
        target=root/'smoke'/Path(binding['path']).name
        os.link(binding['path'],target);artifacts.append(bound(target))
    write(root/'smoke/completion.json',{**done,'artifacts':artifacts,'authenticated_from':bound(oldroot/'smoke/completion.json'),'no_native_rerun':True})
    protocol={**old,'version':'completed45-native-contrast-labels-v2','registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'programs':programs,'bindings':bindings,'tests':t,'predecessor':str(oldroot),
        'data_collection_gate_correction':'V32 stopped on one inherited unchanged-HOLD one-share deadline breach, despite an exact unhooked HOLD bridge and safe new liquidation. Preserve all222pairs including failure and negative action value. Independent review34253rawMTMmarks passed. Distinguish an inherited risk observation from an introduced execution defect. Candidate physical/deadline admission thresholds remain unchanged; all risk outcomes remain available separately from mature economic values. No outcome row is deleted or reclassified as safe.',
        'source_reader':'Same native helper/core/broker and fresh per-audit source_generation_hash. Re-enumerate membership and SHA256 EVERY source byte EVERY audit through OS read-only mappings; Windows read handles deny writes/deletion/replacement until close. Expected full digest must match original function. No digest caching or manifest substitution. Before/after import origins and hashes retained. Added/deleted/mutated-source tests fail closed.',
        'economic_change_from_v32':False,'authenticated_pairs_reused':222,
        'runtime_estimate':'Source-only full frozen runtime benchmark bound; production throughput measured fresh after first new full day. V32 smoke was dominated by repeated source filesystem traversal. Do not claim a whole-study ETA.'}
    protocol['bindings'] += [bound(root/'smoke/completion.json')]+artifacts
    write(root/'protocol.json',protocol);write(root/'study.json',{'data_only':True,'planned_pairs':38508,'authenticated_reuse':222,'economic_changes':0,'study_complete':False})
    print(json.dumps({'registered':str(root),'planned_pairs':38508,'authenticated_pairs_reused':222}),flush=True)


def produce(root,smoke):
    p=read(root/'protocol.json');verify(p['programs']+p['bindings'])
    prefixes=('backtests','scripts','kis_core','strategy_kalcb','strategy_common')
    if any(k.split('.')[0] in prefixes for k in sys.modules):raise ValueError('Application imported before isolation')
    code=Path(p['runtime']).resolve();sys.path[:]=[str(code)]+[x for x in sys.path if x and not Path(x).resolve().is_relative_to(REPO)]
    source=next(v for v in p['programs'] if Path(v['path']).resolve()==HELPER.resolve())
    reader_source=next(v for v in p['programs'] if Path(v['path']).name=='exact_source_reader_v1.py')
    spec=importlib.util.spec_from_file_location(MODULE,source['path']);module=importlib.util.module_from_spec(spec)
    sys.modules[MODULE]=module;spec.loader.exec_module(module)
    reader_name='backtests.strategies.kalcb.exact_source_reader_v1'
    rs=importlib.util.spec_from_file_location(reader_name,reader_source['path']);reader_module=importlib.util.module_from_spec(rs)
    sys.modules[reader_name]=reader_module;rs.loader.exec_module(reader_module)
    external={MODULE:source,reader_name:reader_source}
    manifest=read(p['runtime_manifest']['path'])
    def origins():
        rows={}
        for name,obj in list(sys.modules.items()):
            file=getattr(obj,'__file__',None)
            if not file or name.split('.')[0] not in prefixes:continue
            path=Path(file).resolve()
            if name in external:
                registered=external[name]
                if path!=Path(registered['path']).resolve() or digest(path)!=registered['sha256']:raise ValueError('Changed registered external module')
                rows[name]={**registered,'registered_external':True};continue
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
    write(root/(stage+'_imports_before.json'),origins())
    from unittest.mock import patch
    from backtests.engine import integrity
    expected=integrity.source_generation_hash()
    with reader_module.ExactSourceReader(code,expected) as reader:
        with patch.object(integrity,'source_generation_hash',reader):module.run(root,smoke)
        write(root/'source_reader_receipt.json',{'source_hash':reader(),'fresh_all_byte_hash_calls':reader.calls,'source_files':len(reader.paths),'hash_cache':False,'windows_write_deny_handles':os.name=='nt'})
    if integrity.source_generation_hash()!=expected:raise ValueError('Source changed after mapped-reader release')
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
        write(root/'startup_health.json',{'passed':True,'supervisor':state,'authenticated_smoke':bound(root/'smoke/completion.json'),
            'predecessor_review':bound(Path(p['predecessor'])/'failure_review_v1/review.json'),
            'inherited_parent_deadline_failure_retained':1,'new_action_deadline_failure_in_smoke':0,
            'tests':p['tests']['passed'],'economic_changes':0,'study_complete':False})
        execute(root,[sys.executable,'-I','-B','-u',str(Path(__file__).resolve()),'produce','--root',str(root)],'produce',p['runtime'],state)
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

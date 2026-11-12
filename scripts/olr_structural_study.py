"""Immutable source preparation and single funded structural mechanism run."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,json,os,shutil,subprocess,sys
import psutil

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.olr_economic_study import sha,read,write,sig,source_hash,CHANGED
from scripts.olr_structural_service_patches import patch,VERSION
from scripts.olr_minute_exit_patches_v5 import once
from backtests.engine.olr_interval_service import IntervalParticipationScenario

PARENT=Path('E:/CodexResearch/k_stock_trader/olr_economic_policy_20261013')
OOS_PARENT=Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010/structural_v6/drawdown_reserve_v1')
COPIED=tuple(dict.fromkeys((*CHANGED,
    'backtests/engine/olr_minute_v5.py','backtests/engine/olr_interval_service.py',
    'backtests/strategies/olr/funded_checkpoint.py','backtests/strategies/olr/structural_driver.py',
    'strategy_olr/core/signal_episodes.py','strategy_olr/core/marked_book_paths.py','strategy_olr/core/precision_v3.py',
    'scripts/olr_structural_service_patches.py','scripts/olr_structural_study.py','scripts/olr_structural_controller.py',
    'scripts/olr_minute_exit_patches_v5.py','scripts/olr_economic_study.py','scripts/olr_economic_assessment.py',
    'backtests/strategies/olr/economic_evidence.py','backtests/strategies/olr/decision_response.py',
    'tests/strategy_olr/test_capacity_allocation.py','tests/strategy_olr/test_decision_response.py',
    'tests/strategy_olr/test_structural_controller.py','tests/strategy_olr/test_precision_v3.py','tests/strategy_olr/test_opening_risk_v2.py',
    'tests/strategy_olr/test_structural_service.py','tests/strategy_olr/test_marked_book_paths.py',
    'tests/strategy_olr/test_signal_episodes.py','tests/strategy_olr/test_decision_account.py',
    'tests/strategy_olr/test_minute_exit_v5.py','tests/strategy_olr/test_economic_evidence.py')))
PATCHED=('backtests/engine/sim_broker.py','backtests/engine/integrity.py',
    'backtests/engine/capacity_audit.py','backtests/engine/qualification_evidence.py',
    'backtests/engine/olr_minute_audit_v5.py','backtests/engine/olr_minute_source_v5.py',
    'strategy_olr/core/logic.py')


def private_write(path,text):
    # Never truncate a hard-linked parent. Replace the link with a private file.
    path=Path(path)
    if path.exists():path.unlink()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:f.write(text)


def patch_runner(text):
    text=once(text,"    minute_clock=study.get('minute_source_definition')", "    structural=bound('structural_settings') if study.get('structural_settings') else None\n    minute_clock=study.get('minute_source_definition')")
    text=once(text,'        capacity_model=MinuteParticipationScenario(fraction,scenario_name)',
        "        from backtests.engine.olr_interval_service import IntervalParticipationScenario\n        capacity_model=(IntervalParticipationScenario if structural and structural['interval_service'] else MinuteParticipationScenario)(fraction,scenario_name)")
    text=once(text,'        minute_driver=MinuteOLRDriver(broker,minute_adapter,fast_exit=ocfg.research_minute_exit)',
        "        from backtests.strategies.olr.structural_driver import StructuralDriver\n        minute_driver=StructuralDriver(broker,minute_adapter,mode=structural['mode'],output=args.output,interval_service=structural['interval_service']) if structural else MinuteOLRDriver(broker,minute_adapter,fast_exit=ocfg.research_minute_exit)")
    anchor="    if olr_adapter is not None:\n        write_json(args.output/'capacity_terminal_state.json'"
    text=once(text,anchor,"""    if structural:
        if not minute_driver.checkpoint_receipt or not minute_driver.checkpoint_receipt['no_action_continuation_equal']:
            raise ValueError('No verified complete-account checkpoint')
        write_json(args.output/'funded_checkpoint_receipt.json',minute_driver.checkpoint_receipt)
        write_json(args.output/'marked_book_observations.json',minute_driver.structural_observer.trace)
        write_json(args.output/'signal_episodes.json',olr_adapter.state.meta.get('structural_signal_episodes_v1',{}))
    if olr_adapter is not None:
        write_json(args.output/'capacity_terminal_state.json'""")
    old="                if row['event'] in ('minute_day_scope_v5','minute_exit_signal_v5','fill','submit','cancel','expire','reject'):"
    old_marks="                if row['event'] in ('minute_day_scope_v5','minute_exit_signal_v5','opening_marks','closing_marks','fill','submit','cancel','expire','reject'):"
    if old_marks in text:old=old_marks
    text=once(text,old,'                if True:  # Complete audit journal, including marks/service/capacity/clocks.')
    text=once(text,"'source_definition':minute_clock,'fast_exit':bool(ocfg.research_minute_exit)}",
        "'source_definition':minute_clock,'fast_exit':broker.minute_fast_exit}\n        if structural:identity['configuration']['structural_decision_v1']={'settings_path':study['structural_settings']}")
    return text


def prepare(root,window='train'):
    parent=PARENT if window=='train' else OOS_PARENT
    old=read(parent/'native_engines'/window/'ready.json');src=Path(old['files'])
    dest=root/'native_engines'/window/'files';dest.mkdir(parents=True,exist_ok=False)
    m=copy.deepcopy(old['manifest']);m.pop('manifest_hash',None)
    for i,name in enumerate(m['files']):
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(src/name,target)
        if i%2000==0:print(json.dumps(dict(phase='immutable_links',files=i)),flush=True)
    for name in COPIED:
        private_write(dest/name,(REPO/name).read_text(encoding='utf-8'))
        m['files'][name]=sha(dest/name)
    for name in PATCHED:
        text=patch(name,(dest/name).read_text(encoding='utf-8'));compile(text,name,'exec')
        private_write(dest/name,text);m['files'][name]=sha(dest/name)
    name='scripts/portfolio_synergy_study.py';text=patch_runner((dest/name).read_text(encoding='utf-8'))
    compile(text,name,'exec');private_write(dest/name,text);m['files'][name]=sha(dest/name)
    # Registered capacity evidence is changed only for interval cases; legacy
    # parity uses the original declaration, avoiding a self-selected comparator.
    qpath=m['evidence']['qualification_path'];e=read(dest/qpath)
    e['capacity']['model_version']=VERSION
    for row in e['capacity']['registered_execution_policies']:
        c=row['policy'].get('capacity')
        if c:row['policy']['capacity']=IntervalParticipationScenario(c['participation'],c['name']).contract()
    qnew='research/structural_v1/qualification_interval.json';write(dest/qnew,e);m['files'][qnew]=sha(dest/qnew)
    study=copy.deepcopy(old['study']);study['code']=str(dest);study['source_generation_hash']=source_hash(dest)
    m['source_generation_hash']=study['source_generation_hash']
    write(root/'native_engines'/window/'ready.json',dict(files=str(dest),study=study,manifest=m,interval_qualification=qnew))
    write(root/'native_engines'/window/'freeze_receipt.json',dict(parent=str(parent),parent_ready_sha256=sha(parent/'native_engines'/window/'ready.json'),
        source=study['source_generation_hash'],private_files={n:h for n,h in m['files'].items() if h!=old['manifest']['files'].get(n)},
        retained_inputs_unchanged=True,legacy_model_preserved=True))
    print(json.dumps(dict(prepared=window,source=study['source_generation_hash'])),flush=True)


def launch(root,window,mode,scenario='baseline',capped=False,legacy=False):
    r=read(root/'native_engines'/window/'ready.json');files=Path(r['files']);s=copy.deepcopy(r['study']);m=copy.deepcopy(r['manifest'])
    tag=('legacy_' if legacy else '')+mode+'_'+scenario+('_capped' if capped else '')
    out=root/'cases'/window/tag
    if out.exists():raise FileExistsError('Preserve existing case '+str(out))
    settings=dict(mode=mode,interval_service=not legacy)
    prefix='research/structural_v1/'+window+'/'+tag+'/'
    s['structural_settings']=prefix+'settings.json';write(files/s['structural_settings'],settings)
    m['files'][s['structural_settings']]=sha(files/s['structural_settings'])
    s.pop('economic_policy_settings',None);s.pop('drawdown_reserve_contract',None)
    if mode=='structural_partial':
        s['candidates']['candidate']['olr_capacity'].update(capacity_structural_partial=True,capacity_weak_partial_fraction=.5)
    if not legacy:m['evidence']['qualification_path']=r['interval_qualification']
    if window=='oos':
        # Replace old parent's market-history continuation with THIS policy's
        # authenticated flat IS terminal state; the native loader verifies it.
        case=root/'cases/train'/tag
        from backtests.strategies.olr.economic_evidence import completed_case
        if not completed_case(case)['valid']:raise ValueError('Invalid IS parent for OOS')
        for key,name in [('capacity_checkpoint','capacity_terminal_state.json'),('capacity_checkpoint_result','result.json'),('capacity_checkpoint_completion','completion.json')]:
            rel=prefix+'parent_'+name;target=files/rel;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(case/name,target);m['files'][rel]=sha(target);s[key]=rel
    s['manifest']=str(files.parent/(tag+'.manifest.json'));s['configuration']['input_manifest_path']=s['manifest']
    s['spec_path']=prefix+'spec.json';write(files/s['spec_path'],{k:v for k,v in s.items() if k not in ('code','manifest','spec_path','source_generation_hash')})
    m['files'][s['spec_path']]=sha(files/s['spec_path']);write(s['manifest'],{**m,'manifest_hash':sig(m)})
    sp=root/'runtime'/window/tag/'study.json';write(sp,s)
    command=[sys.executable,'-B','-u',str(files/'scripts/portfolio_synergy_study.py'),'replay','--study',str(sp),
        '--recipe','candidate','--scenario',scenario,'--standalone','OLR','--output',str(out),'--full-audit']
    if capped:command.append('--no-upward-compounding')
    tmp=root/'tmp'/tag;tmp.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(TEMP=str(tmp),TMP=str(tmp),PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    log=root/'logs'/window/(tag+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x',encoding='utf-8') as f:
        process=subprocess.Popen(command,cwd=files,env=env,stdout=f,stderr=subprocess.STDOUT,creationflags=0x08000000)
    identity=dict(pid=process.pid,created_at=psutil.Process(process.pid).create_time(),command=command,
        source=s['source_generation_hash'],settings=settings,log=str(log),case=str(out),started_at=datetime.now(timezone.utc).isoformat())
    write(sp.parent/'process.json',identity)
    return process,identity


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','run'));p.add_argument('--root',type=Path,required=True)
    p.add_argument('--window',default='train');p.add_argument('--mode',default='control');p.add_argument('--scenario',default='baseline')
    p.add_argument('--capped',action='store_true');p.add_argument('--legacy',action='store_true');a=p.parse_args()
    if a.phase=='prepare':prepare(a.root,a.window)
    else:
        process,identity=launch(a.root,a.window,a.mode,a.scenario,a.capped,a.legacy)
        print(json.dumps(identity),flush=True);rc=process.wait()
        if rc:raise SystemExit(rc)

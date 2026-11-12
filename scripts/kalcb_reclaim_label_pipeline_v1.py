"""Serialized registered data production only; no optimizer or portfolio runner."""
from datetime import datetime,timezone
from pathlib import Path
import argparse,os,sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify
from backtests.strategies.kalcb.research_process_v1 import execute


def native_workers():
    import psutil
    result=[]
    for process in psutil.process_iter(['pid','cmdline','create_time']):
        if process.pid==os.getpid():continue
        args=process.info['cmdline'] or []
        if any(a=='scripts.kalcb_executed_value_replay' or
            (Path(a).name.startswith('kalcb_') and ('_worker_v' in Path(a).name or 'reclaim_producer_' in Path(a).name))
            for a in args):result.append(process.info)
    return result


def run(root):
    import psutil,msvcrt
    p=read(root/'protocol.json');verify(p['programs']+p['bindings']+p['data_bindings'])
    if (root/'supervisor_start.json').exists():raise ValueError('Preserve previous attempt')
    if native_workers():raise ValueError('Another native data/portfolio worker exists')
    lock=(root/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    temp=root/'temporary';temp.mkdir(exist_ok=True);os.environ.update(TEMP=str(temp),TMP=str(temp),PYTHONDONTWRITEBYTECODE='1')
    own={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':psutil.Process().cmdline()}
    write(root/'supervisor_start.json',own);progress=[]
    try:
        for stage in ('prepare','smoke_baseline','smoke_combined','labels_baseline','labels_combined'):
            if native_workers():raise ValueError('Second native worker detected')
            if stage.startswith('labels'):
                for scenario in ('baseline','combined'):
                    receipt=read(root/'smoke'/scenario/'completion.json')
                    if not receipt['complete'] or receipt['counts'].get('filled',0)<1 or receipt['counts'].get('independent_execution_failures',0):
                        raise ValueError('Filled audited native smoke missing')
            code=Path(p['producer_runtime']['code'])
            command=[sys.executable,'-B','-u',str(REPO/'scripts/kalcb_reclaim_producer_v1.py'),
                '--root',str(root),'--code',str(code),'--stage',stage]
            log=execute(root,command,stage,code,own)
            before=read(root/(stage+'_imports_before.json'));after=read(root/(stage+'_imports_after.json'))
            progress.append({'stage':stage,'log':log,'imports_before':bound(root/(stage+'_imports_before.json')),
                'imports_after':bound(root/(stage+'_imports_after.json'))})
            write(root/'progress.json',progress)
        for scenario in ('baseline','combined'):
            done=read(root/'labels'/scenario/'completion.json')
            if not done['complete'] or done['smoke']:raise ValueError('Incomplete full label production')
            verify([{'path':str(root/'labels'/scenario/name),'sha256':h} for name,h in done['files'].items()])
        verify(p['programs']+p['bindings']+p['data_bindings'])
        write(root/'completion.json',{'complete':True,'study_complete':False,'portfolio_replays':0,'models_fitted':0,
            'prepared':bound(root/'prepared/prepared.json'),
            'labels':{s:bound(root/'labels'/s/'completion.json') for s in ('baseline','combined')},
            'next':'Review native readiness, failures/no-fills, mature support and fresh exact-quarter costs/targets. Freeze finite model/calibration/support recipes before any fit or economic replay. Require native unchanged parent bridges and full minute45 plus minute90 source/account qualification.'})
        write(root/'status.json',{**own,'stage':'fresh_labels_complete_pending_review','child':None,'completed_stages':len(progress)})
    except Exception as error:
        write(root/'status.json',{**own,'stage':'attention_required','child':None,'error':repr(error),'completed_stages':len(progress)})
        raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root.resolve())

"""One-shot handoff after the immutable v3 verifier-context failure.

Never terminates or duplicates a healthy research process. No replay criteria
are relaxed; the versioned successor authenticates and checks prior evidence.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.kalcb_full_oos_recovery import bound, digest, env, read, write

ROOT = Path('E:/CodexResearch/k_stock_trader/kalcb_recovery_qualification_20261010/successor_watch_v1')
PRIOR = ROOT.with_name('v3')
NEXT = ROOT.with_name('v4')
PROGRAM = REPO/'scripts/kalcb_recovery_qualification_round_v4.py'


def main():
    import msvcrt
    import psutil
    protocol = read(ROOT/'protocol.json')
    for item in protocol['bindings']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Changed successor dependency '+item['path'])
    lock = (ROOT/'run.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1): lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    state = {'pid': os.getpid(), 'process_created': psutil.Process().create_time()}

    def status(stage, **extra):
        state.update(stage=stage, updated_at_utc=datetime.now(timezone.utc).isoformat(), **extra)
        write(ROOT/'status.tmp', state); os.replace(ROOT/'status.tmp', ROOT/'status.json')

    def same_process(identity):
        try:
            process = psutil.Process(identity['pid'])
            if abs(process.create_time()-identity['process_created']) > .1: return False
            if process.cmdline() != identity['command']:
                raise ValueError('Process identity changed')
            return True
        except psutil.NoSuchProcess:
            return False

    def run(stage):
        if digest(PROGRAM) != protocol['successor_sha256']:
            raise ValueError('Successor changed after registration')
        for process in psutil.process_iter(['pid', 'cmdline']):
            try:
                command = process.info.get('cmdline') or []
                if any(Path(x).name == PROGRAM.name for x in command):
                    raise ValueError('A successor process already exists; never duplicate it')
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        with (ROOT/(stage+'.log')).open('x', encoding='utf-8') as stream:
            command = [sys.executable, '-B', '-u', str(PROGRAM), stage]
            process = subprocess.Popen(command, cwd=REPO, env=env(REPO), stdout=stream,
                                       stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            status('successor_'+stage, child={'pid': process.pid,
                'process_created': psutil.Process(process.pid).create_time(), 'command': command})
            while process.poll() is None: time.sleep(15)
        if process.returncode: raise ValueError('Successor '+stage+' failed; preserve evidence')

    try:
        status('waiting_for_immutable_prior', prior=protocol['prior_process'])
        while same_process(protocol['prior_process']): time.sleep(15)
        prior = read(PRIOR/'status.json')
        if prior['stage'] != 'attention_required' or prior.get('error') != "KeyError('audit_manifest')":
            raise ValueError('Prior state is not the narrowly authorized metadata repair: '+str(prior))
        child = prior.get('child')
        if child and same_process(child): raise ValueError('Prior child still alive')
        if NEXT.exists(): raise ValueError('Preserve existing successor; no implicit restart')
        run('register')
        run('run')
        if not read(NEXT/'completion.json')['complete']: raise ValueError('Incomplete qualification')
        status('ready_for_final_research_review', child=None)
    except Exception as error:
        status('attention_required', error=repr(error)); raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__':
    main()

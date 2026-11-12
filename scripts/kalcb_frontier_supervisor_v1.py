"""Continue one identified source check into its registered offline replay.

Operational supervision only: no economic parameters, retries, or new search.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def main(root, producer_pid):
    root = root.resolve()
    runner = Path(__file__).with_name('kalcb_frontier_completion_v1.py').resolve()
    producer = psutil.Process(producer_pid)
    command = producer.cmdline()
    expected_tail = ['--phase', 'register', '--output', str(root).replace('\\', '/')]
    normalized = [arg.replace('\\', '/') for arg in command]
    if normalized[-4:] != expected_tail or not any(arg.endswith('/kalcb_frontier_completion_v1.py') for arg in normalized):
        raise ValueError('Not the intended registration process')
    identity = dict(pid=producer_pid, created_at=producer.create_time(), command=command)
    receipt = dict(at_utc=datetime.now(timezone.utc).isoformat(), producer=identity,
                   runner=dict(path=str(runner), sha256=digest(runner)),
                   supervisor=dict(path=str(Path(__file__).resolve()), sha256=digest(__file__)),
                   scope='Wait for this producer, then execute its existing registered eight-case design once. No retries.')
    with (root / 'supervisor_registration.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)

    def status(phase, **extra):
        payload = dict(at_utc=datetime.now(timezone.utc).isoformat(), phase=phase,
                       pid=os.getpid(), created_at=psutil.Process().create_time(), **extra)
        save(root / 'supervisor_status.json', payload)
        print(json.dumps(payload), flush=True)

    try:
        while True:
            try:
                if not producer.is_running() or producer.create_time() != identity['created_at']:
                    break
                if producer.cmdline() != command:
                    raise ValueError('Registration process identity changed')
                counters = producer.io_counters()._asdict()
            except psutil.NoSuchProcess:
                break
            status('waiting_for_source_authentication', producer=identity, producer_io=counters)
            time.sleep(30)
        if (root / 'failure_register.json').exists():
            raise RuntimeError('Registration failed; preserve generation and inspect failure receipt')
        registration = json.loads((root / 'registration.json').read_text(encoding='utf-8'))
        if digest(runner) != receipt['runner']['sha256']:
            raise ValueError('Runner changed while registration was running')
        bindings = [b for b in registration['programs'] if Path(b['path']).resolve() == runner]
        if len(bindings) != 1 or bindings[0]['sha256'] != receipt['runner']['sha256']:
            raise ValueError('Registration does not bind the waiting runner')
        if registration['fresh_cases'] != 8 or registration['workers'] != 1:
            raise ValueError('Unexpected registered execution allocation')
        if (root / 'cases').exists() or (root / 'completion.json').exists():
            raise ValueError('Native generation already started; do not duplicate or overwrite')
        run_command = [sys.executable, '-B', '-u', str(runner), '--phase', 'run', '--output', str(root)]
        run_env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                   'TEMP': str(root / 'temp'), 'TMP': str(root / 'temp'),
                   **{k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS')}}
        with (root / 'native_orchestrator.log').open('x', encoding='utf-8') as stream:
            child = subprocess.Popen(run_command, cwd=runner.parents[1], env=run_env,
                                     stdout=stream, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            child_identity = dict(pid=child.pid, created_at=psutil.Process(child.pid).create_time(), command=run_command)
            status('native_orchestrator_running', child=child_identity)
            code = child.wait()
        if code:
            raise RuntimeError('Native orchestration failed; inspect log and preserve cases')
        completion = json.loads((root / 'completion.json').read_text(encoding='utf-8'))
        if not completion['complete']:
            raise RuntimeError('Native process exited without completed assessment')
        status('assessment_complete_economic_decision_pending', completion_sha256=digest(root / 'completion.json'))
    except BaseException as exc:
        status('attention_required', error=repr(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--producer-pid', type=int, required=True)
    args = parser.parse_args()
    main(args.output, args.producer_pid)

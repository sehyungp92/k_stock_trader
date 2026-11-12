"""Run the already specified 0035 review once all three accounts are accepted."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def run(root):
    protocol = read(root / 'postreview_dispatch_protocol.json')
    review = Path(protocol['review_script'])
    assert digest(Path(__file__)) == protocol['dispatcher_sha256']
    assert digest(review) == protocol['review_sha256']
    assert not (root / 'causal_review_v1').exists(), 'Preserve existing review; inspect instead of duplicating'
    status_path = root / 'postreview_dispatch_status.json'
    deadline = time.monotonic() + 45 * 60
    write(status_path, dict(at=stamp(), phase='waiting_for_three_accepted_accounts'))
    while True:
        state = read(root / 'status.json')
        if state['phase'] == 'awaiting_economic_and_causal_assessment':
            assert set(state['accepted']) == {'control', 'half', 'full'} and not state['queued']
            assessment = read(root / 'assessment.json')
            assert set(assessment['summaries']) == {'control', 'half', 'full'}
            break
        assert state['phase'] != 'needs_repair', state
        identity = read(root / 'controller_process.json')
        controller = psutil.Process(identity['pid'])
        assert controller.create_time() == identity['created_at']
        assert controller.cmdline() == identity['command']
        assert time.monotonic() < deadline, 'Review dispatch wait bound reached; inspect healthy work before extending'
        time.sleep(30)
    assert digest(review) == protocol['review_sha256']
    assert not (root / 'causal_review_v1').exists()
    command = [sys.executable, '-B', str(review), str(root)]
    worker = subprocess.Popen(command, cwd=protocol['repo'], creationflags=0x08000000)
    write(root / 'postreview_worker_process.json', dict(at=stamp(), pid=worker.pid,
          created_at=psutil.Process(worker.pid).create_time(), command=command))
    write(status_path, dict(at=stamp(), phase='review_running', pid=worker.pid))
    code = worker.wait()
    assert code == 0, f'Review failed with exit {code}; preserve its generation before repair'
    output = root / 'causal_review_v1' / 'assessment.json'
    assert output.exists()
    write(status_path, dict(at=stamp(), phase='awaiting_economic_interpretation',
          output=str(output), output_sha256=digest(output), economic_objective_achieved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    arguments = parser.parse_args()
    try:
        run(arguments.root)
    except Exception as exc:
        write(arguments.root / 'postreview_dispatch_status.json',
              dict(at=stamp(), phase='needs_attention', error=repr(exc)))
        raise

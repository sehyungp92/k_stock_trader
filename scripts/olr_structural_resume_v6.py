"""Resume completed source preparation after the preserved test-import failure."""
from pathlib import Path
import argparse
import sys
import traceback
import xml.etree.ElementTree as ET
from scripts import olr_structural_native_v6 as e
from scripts.olr_structural_v6 import REPO, read, sha, write, utc

PRESERVATION = 'cae22b25d2773d0dc01615004da5ebc2515c198e0b94d36fdcea2d8556e08e6b'
LAUNCH = 'd6a33306fcc65cb14057706511a03d5996be3d39596fb00561a7c71762a2c6af'


def authenticate():
    e.configure()
    assert sha(e.ROOT / 'launch_harness.json') == LAUNCH
    e.verify_launch()
    receipt = e.ROOT / 'test_isolation_repair.json'
    repair = read(receipt)
    assert sha(e.ROOT / 'freeze_failed_attempt_0/preservation.json') == PRESERVATION
    for name, digest in repair['files'].items():
        assert sha(REPO / name) == digest, name
    for name, digest in repair['artifacts'].items():
        assert sha(name) == digest, name
    assert read(e.ROOT / 'preparation_complete.json')['no_performance_executed']
    return repair


def freeze():
    repair = authenticate()
    assert not (e.ROOT / 'frozen_harness.json').exists(), 'Never re-freeze'
    command = [sys.executable, '-B', str(REPO / 'scripts/olr_structural_native_test_isolated_v6.py'), str(e.ROOT)]
    e.child(command, e.ROOT / 'isolated_tests.log', e.ROOT / 'isolated_tests_process.json',
            cwd=e.t.ENGINES / 'train/inputs/files')
    suites = list(ET.parse(e.ROOT / 'isolated_tests.xml').getroot().iter('testsuite'))
    counts = {k: sum(int(v.get(k, '0')) for v in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    assert counts['tests'] == 64 and not any(counts[k] for k in ('failures', 'errors', 'skipped'))
    imports = read(e.ROOT / 'isolated_test_native_imports.json')
    assert imports['exit_code'] == 0 and imports['import_guard_before_execution']
    launch = e.verify_launch()
    bound_files = {**launch['files'], **repair['files'], **imports['test_files']}
    for name, digest in bound_files.items():
        assert sha(REPO / name) == digest, name
    for name, digest in imports['test_files'].items():
        assert repair['files'].get(name) == digest, 'Test helper was not sealed before execution: ' + name
    artifacts = dict(repair['artifacts'])
    for name in ('registration.json', 'candidate_plan.json', 'protocol.json', 'mechanism_plan.json',
                 'preparation_complete.json', 'source_reuse_train.json', 'source_reuse_oos.json',
                 'source_projection_contract.json', 'oos_source_worker.py', 'test_isolation_repair.json',
                 'isolated_tests.xml', 'isolated_tests.log', 'isolated_test_native_imports.json'):
        artifacts[str(e.ROOT / name)] = sha(e.ROOT / name)
    # Completed generations are authenticated in place. No preparation or
    # source extraction is repeated, and there are no parser exemptions.
    for window in ('train', 'oos'):
        ready_path = e.t.ENGINES / window / 'ready.json'
        ready = read(ready_path)
        files = Path(ready['files'])
        assert e.n.source_hash(files) == ready['study']['source_generation_hash']
        for name, digest in ready['manifest']['files'].items():
            assert sha(files / name) == digest, name
        if window == 'train':
            for name, digest in imports['imports'].items():
                assert ready['manifest']['files'][name] == digest
        artifacts[str(ready_path)] = sha(ready_path)
    authenticate()
    write(e.ROOT / 'frozen_harness.json', dict(utc=utc(), files=bound_files,
        artifacts=artifacts, tests=counts, before_performance=True,
        test_import_provenance_sha256=sha(e.ROOT / 'isolated_test_native_imports.json'),
        additive_test_isolation_repair_sha256=sha(e.ROOT / 'test_isolation_repair.json')))


def qualify():
    authenticate()
    e.qualify()


def run():
    authenticate()
    for phase in ('freeze', 'qualify'):
        e.child([sys.executable, '-B', '-u', '-m', 'scripts.olr_structural_resume_v6', phase],
                e.ROOT / ('resume_' + phase + '.controller.log'),
                e.ROOT / ('resume_' + phase + '_process.json'))
    write(e.ROOT / 'resume_controller_complete.json', dict(utc=utc(),
        qualification_complete_sha256=sha(e.ROOT / 'qualification_complete.json'), report_review_remaining=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('freeze', 'qualify', 'run'))
    args = parser.parse_args()
    e.configure()
    try:
        {'freeze': freeze, 'qualify': qualify, 'run': run}[args.phase]()
    except Exception:
        path = e.ROOT / ('resume_' + args.phase + '_failure.json')
        if not path.exists():
            write(path, dict(utc=utc(), traceback=traceback.format_exc()))
        raise

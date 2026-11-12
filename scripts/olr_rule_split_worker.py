"""Separate native candidate preparation from replay to release the entire heap.

The prepared object is the original native EventReplayBundle. Both phases invoke
the frozen evaluator; no candidate, bar, fill, sizing or audit rule is replaced.
"""
import argparse
import copyreg
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading
from types import MappingProxyType


class BundlePrepared(BaseException):
    """Stop the preparation process at the native replay call boundary."""


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def candidate_sha(path):
    with gzip.open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def restore_mapping(values):
    return MappingProxyType(values)


def reduce_mapping(value):
    return restore_mapping, (dict(value),)


def save_bundle(bundle, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.preparing')
    with temporary.open('wb') as stream:
        pickler = pickle.Pickler(stream, protocol=pickle.HIGHEST_PROTOCOL)
        pickler.dispatch_table = copyreg.dispatch_table.copy()
        pickler.dispatch_table[MappingProxyType] = reduce_mapping
        pickler.dump(bundle)
    temporary.replace(path)
    return sha(path)


def load_bundle(path, expected):
    # Hash and deserialize the same file handle; never load unverified bytes.
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise ValueError('Prepared native replay bundle changed')
        stream.seek(0)
        return pickle.load(stream)


@contextmanager
def memory_profile(root, name, mode, phase, write, runtime_hash):
    import psutil
    process = psutil.Process()
    done, lock = threading.Event(), threading.Lock()
    peak, stages = {}, []

    def observe(stage=None):
        with lock:
            values = process.memory_info()._asdict()
            for key in ('rss', 'vms', 'peak_pagefile', 'peak_wset'):
                peak[key] = max(peak.get(key, 0), values.get(key, 0))
            if stage:
                stages.append({'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(), 'memory': values})

    def sample():
        while not done.wait(1):
            observe()

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        yield observe
    finally:
        observe('phase_finished_or_failed')
        done.set()
        monitor.join(timeout=2)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        write(root / 'memory_profiles' / name / mode / (stamp + '_' + phase + '.json'), {
            'phase': phase, 'runtime_manifest_sha256': runtime_hash, 'peak': peak, 'stages': stages,
            'policy': 'Separate preparation and replay processes; verified native bundle; original audit'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['worker', 'prepare', 'replay'])
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--mutation-file', type=Path)
    parser.add_argument('--runtime-parity', action='store_true')
    args = parser.parse_args()
    root = args.study.resolve().parent
    manifest_path = root / 'analysis_pipeline_v5_manifest.json'
    registration = json.loads((root / 'runtime_v5_registration.json').read_text())
    if sha(manifest_path) != registration['pipeline_manifest_sha256']:
        raise ValueError('Split runtime manifest changed')
    protocol = json.loads(manifest_path.read_text())
    for name, expected in protocol['files'].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError('Split runtime dependency changed: ' + name)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from olr_rule_study import load_study, read, write, signature
    spec = load_study(args.study)
    mutations = read(args.mutation_file) if args.mutation_file else spec['cases'][args.case]['mutations']
    if args.runtime_parity:
        if args.case != 'control' or args.mode != 'sizing_cap' or args.mutation_file:
            raise ValueError('Resource parity is restricted to the published capped control')
        spec['output'] = str(root / 'resource_validation/v5')
    output = Path(spec['output'])
    output.mkdir(parents=True, exist_ok=True)
    key = signature({'study': spec['study_hash'], 'case': args.case, 'mode': args.mode,
                     'mutations': mutations, 'runtime_parity': args.runtime_parity})
    cache = root / 'prepared_native_bundles' / key
    bundle_path, receipt_path = cache / 'bundle.pkl', cache / 'prepared.json'

    def prepared():
        record = read(receipt_path)
        if (record['registration_key'] != key or record['source_hash'] != spec['source_generation_hash']
                or record['runtime_manifest_sha256'] != sha(manifest_path)
                or record['input_manifest_sha256'] != sha(spec['input_manifest'])
                or sha(bundle_path) != record['bundle_sha256']):
            raise ValueError('Prepared native bundle identity differs')
        return record

    if args.command == 'worker':
        common = ['--study', str(args.study), '--case', args.case, '--mode', args.mode]
        if args.mutation_file:
            common += ['--mutation-file', str(args.mutation_file)]
        if args.runtime_parity:
            common += ['--runtime-parity']
        if receipt_path.exists():
            prepared()
            print('REUSE verified native prepared bundle', args.case, args.mode, flush=True)
        else:
            subprocess.run([sys.executable, '-B', '-u', __file__, 'prepare', *common], check=True)
        prepared()
        # The preparation child has exited before replay imports the engine.
        subprocess.run([sys.executable, '-B', '-u', __file__, 'replay', *common], check=True)
        return

    sys.path.insert(0, spec['engine_root'])
    os.chdir(spec['engine_root'])
    temp = root / 'temp'
    temp.mkdir(exist_ok=True)
    os.environ['TEMP'] = os.environ['TMP'] = str(temp)
    import tempfile
    tempfile.tempdir = str(temp)
    with memory_profile(root, args.case, args.mode, args.command, write, sha(manifest_path)) as observe:
        from olr_rule_memory import install_memory_controls
        install_memory_controls(observe)
        from backtests.strategies.olr import runner, replay_cache
        from backtests.auto.shared.cache_keys import source_generation_hash
        from olr_rule_resolved_pipeline import worker
        case_dir = output / 'cases' / args.case / args.mode
        original_runner = runner.run_olr_backtest

        if args.command == 'prepare':
            def capture(config, effective_mutations, *, replay_bundle):
                observe('native_bundle_prepared')
                job = read(case_dir / 'job.json')
                request_hash = signature({'config': config, 'mutations': effective_mutations})
                if request_hash != signature({'config': job['config'], 'mutations': job['mutations']}):
                    raise ValueError('Prepared request differs from native job')
                bundle_hash = save_bundle(replay_bundle, bundle_path)
                if source_generation_hash() != spec['source_generation_hash']:
                    raise ValueError('Native source changed during preparation')
                write(receipt_path, {'registration_key': key, 'source_hash': spec['source_generation_hash'],
                    'input_manifest_sha256': sha(spec['input_manifest']), 'bundle_sha256': bundle_hash,
                    'request_hash': request_hash, 'job_hash': signature(job),
                    'candidate_rows_sha256': candidate_sha(case_dir / 'candidate_selection.jsonl.gz'),
                    'runtime_manifest_sha256': sha(manifest_path), 'events': len(replay_bundle.events),
                    'prepared_at_utc': datetime.now(timezone.utc).isoformat()})
                raise BundlePrepared()
            runner.run_olr_backtest = capture
            try:
                worker(spec, args.case, args.mode, args.mutation_file)
            except BundlePrepared:
                print('PREPARED original native bundle; preparation process exits before replay', flush=True)
            else:
                raise RuntimeError('Native preparation did not reach the expected replay boundary')
        else:
            record = prepared()

            def restore(config, effective_mutations):
                if signature({'config': config, 'mutations': effective_mutations}) != record['request_hash']:
                    raise ValueError('Replay request differs from prepared native request')
                result = load_bundle(bundle_path, record['bundle_sha256'])
                if len(result.events) != record['events']:
                    raise ValueError('Prepared event count differs')
                observe('native_bundle_restored')
                return result

            def checked_replay(config, effective_mutations, *, replay_bundle):
                if (signature(read(case_dir / 'job.json')) != record['job_hash']
                        or candidate_sha(case_dir / 'candidate_selection.jsonl.gz') != record['candidate_rows_sha256']):
                    raise ValueError('Restored native candidates or job differ from preparation')
                observe('before_original_native_replay')
                return original_runner(config, effective_mutations, replay_bundle=replay_bundle)

            replay_cache.load_olr_real_replay_bundle = restore
            replay_cache.load_olr_declared_window_bundle = restore
            runner.run_olr_backtest = checked_replay
            worker(spec, args.case, args.mode, args.mutation_file)


if __name__ == '__main__':
    main()

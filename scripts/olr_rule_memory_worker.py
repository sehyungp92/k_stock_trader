"""Memory-managed invocation of the unchanged frozen OLR study evaluator."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['worker'])
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--mutation-file', type=Path)
    args = parser.parse_args()
    root = args.study.resolve().parent
    protocol = json.loads((root / 'memory_runtime_manifest.json').read_text())
    for name, expected in protocol['files'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Memory runtime changed')
    sys.path.insert(0, str(root / 'harness_v3'))
    from olr_rule_study import load_study, write
    from olr_rule_resolved_pipeline import worker
    spec = load_study(args.study)
    sys.path.insert(0, spec['engine_root'])
    os.chdir(spec['engine_root'])
    os.environ['TEMP'] = os.environ['TMP'] = str(root / 'temp')
    import tempfile
    tempfile.tempdir = str(root / 'temp')
    import psutil
    from olr_rule_memory import install_memory_controls
    process = psutil.Process()
    done = threading.Event()
    memory_lock = threading.Lock()
    peak, stages = {}, []

    def sample():
        with memory_lock:
            values = process.memory_info()._asdict()
            for key, value in values.items():
                if key in {'rss', 'vms', 'pagefile', 'peak_pagefile', 'peak_wset'}:
                    peak[key] = max(peak.get(key, 0), value)
            return values

    def observe(stage):
        stages.append({'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(), 'memory': sample()})

    def sampler():
        while not done.wait(1):
            sample()

    monitor = threading.Thread(target=sampler, daemon=True)
    monitor.start()
    cache = install_memory_controls(observe)
    observe('before_original_worker')
    try:
        worker(spec, args.case, args.mode, args.mutation_file)
    finally:
        observe('worker_finished_or_failed')
        done.set()
        monitor.join(timeout=2)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        write(root / 'memory_profiles' / args.case / args.mode / (stamp + '.json'), {
            'study_hash': spec['study_hash'], 'runtime_manifest_sha256': hashlib.sha256((root / 'memory_runtime_manifest.json').read_bytes()).hexdigest(),
            'peak': peak, 'stages': stages, 'cache': {name: getattr(cache, name) for name in (
                'maximum_bytes', 'bytes', 'peak_bytes', 'evictions', 'hits', 'misses')},
            'policy': '64MiB rebuildable verified-byte cache; single Arrow worker; release unused buffers around unchanged native audit'})


if __name__ == '__main__':
    main()

"""Execute registered diagnostic replays; never select or change a recipe."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if digest(plan['wrapper']) != plan['wrapper_sha256']:
        raise ValueError('Frozen observer changed')

    def run(item):
        name, case = item
        if digest(case['study']) != case['study_sha256']:
            raise ValueError('Frozen study changed: ' + name)
        path = Path(case['path'])
        if case['reuse']:
            if not (path/'completion.json').exists():
                raise ValueError('Missing registered audited reference: ' + name)
            return name, 'reused'
        if (path/'completion.json').exists():
            return name, 'exists; must be validated by final reporter'
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-B', plan['wrapper'], '--study', case['study'],
                   '--recipe', case['recipe'], '--scenario', case['scenario'],
                   '--output', str(path), '--full-audit']
        if case['no_upward_compounding']:
            command.append('--no-upward-compounding')
        print('Starting ' + name, flush=True)
        with path.with_suffix('.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        return name, 'completed'

    # Independent isolated processes, each reading the same immutable source.
    # Final publication separately authenticates every native receipt and output.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, item) for item in plan['cases'].items()]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)


if __name__ == '__main__':
    main()

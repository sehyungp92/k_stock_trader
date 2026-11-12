"""Read-only selected-policy diagnosis. Never changes policy or frozen inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010/remaining_drawdown_v2')
OUT = REPO / 'data/backtests/analysis/olr_remaining_drawdown_20261010'
PRIOR = ROOT.parent / 'targeted_risk_v1'
SELECTED = 'cc9a4ad98fc93351e'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def observe():
    assert not ROOT.exists(), 'Preserve existing diagnostic runs; inspect before resuming.'
    ROOT.mkdir(parents=True)
    OUT.mkdir(parents=True, exist_ok=True)
    observer = REPO / 'scripts/olr_drawdown_trace_v1.py'
    write(ROOT/'registration.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'pid': os.getpid(),
        'command': 'python -B -u -m scripts.olr_remaining_drawdown_v2 observe',
        'selected': SELECTED,
        'purpose': 'Exact-ledger attribution of remaining selected-policy drawdown and decision-time winner/loser overlap. Observational only; no policy search or changes.',
        'observer_sha256': sha(observer),
        'windows_serial': ['oos', 'train'],
        'constraints': ['Original native audits', 'Exact trades/daily/equity/terminal-history parity', 'Frozen inputs untouched', 'No live activation', 'April-July is inspected development evidence'],
    })
    for window in ('oos', 'train'):
        reference = PRIOR/'cases'/SELECTED/window/'100000000_baseline'
        with (ROOT/(window+'.log')).open('w', encoding='utf-8') as stream:
            subprocess.run([sys.executable, '-B', '-u', str(observer), '--reference', str(reference), '--output', str(ROOT/window)],
                           cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(window+' exact-parity observation complete', flush=True)
    write(ROOT/'observations_complete.json', {'completed_utc': datetime.now(timezone.utc).isoformat(),
          'proofs': {w: read(ROOT/(w+'_exact_parity.json')) for w in ('oos','train')}})

def reconstruct():
    from scripts import olr_drawdown_diagnosis_v1 as d
    d.ROOT = ROOT
    d.OUT = OUT
    for window in ('oos', 'train'):
        d.trace(window)

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('action', choices=['observe','reconstruct'])
    args=p.parse_args()
    {'observe': observe, 'reconstruct': reconstruct}[args.action]()

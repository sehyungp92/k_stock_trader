"""Archive a superseded round and replay the sealed candidate with the existing engine.

This is explicit research round assignment, never an OOS acceptance override.
The existing report_native_portfolio.py writes/finalizes the round after replay.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from backtests.auto.shared.phase_state import _atomic_write_json
from scripts.portfolio_native_budget_study import checked, read, verified_spec
from scripts.replay_selected_portfolio import binding, bound, digest


def archive_round(strategy_dir, stamp):
    strategy_dir = Path(strategy_dir).resolve()
    source = (strategy_dir / 'round_1').resolve()
    archive = (strategy_dir / 'archive' / stamp).resolve()
    # Check resolved targets before any recursive move, including junctions.
    if not source.is_relative_to(strategy_dir) or not archive.is_relative_to(strategy_dir):
        raise ValueError('Archive target escapes the strategy directory')
    if archive.exists() or not source.is_dir():
        raise FileExistsError('Require an existing round and a new archive directory')
    hashes = {str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file()}
    archive.mkdir(parents=True)
    shutil.copy2(strategy_dir / 'rounds_manifest.json', archive / 'rounds_manifest.json')
    source.rename(archive / 'round_1')
    for name, sha in hashes.items():
        if digest(archive / 'round_1' / name) != sha:
            raise ValueError('Archived round bytes changed')
    _atomic_write_json({'archived_utc': datetime.now(timezone.utc).isoformat(),
        'original_directory': str(source), 'archive_directory': str(archive / 'round_1'),
        'files': hashes, 'file_count': len(hashes), 'verified_exact': True,
        'embedded_paths_preserved': True,
        'path_note': 'Historical embedded round_1 paths refer to this archived generation; files were preserved byte-for-byte.'},
        archive / 'archive_receipt.json')
    return archive


def prepare(parent, oos, strategy_dir, pointer):
    parent, oos, strategy_dir, pointer = map(lambda p: Path(p).resolve(), (parent, oos, strategy_dir, pointer))
    selection_path = parent / 'development_selection.json'
    seal = read(selection_path)
    study_path = parent / 'battery/study.json'
    study = read(study_path)
    verified_spec(study)
    from scripts.portfolio_allocation_profile import verify_frozen_bytes
    verify_frozen_bytes(study, {})
    if (seal['finalist'] != 'k_marginal' or study['candidates']['k_marginal'] != seal['recipe']
            or study['initial_equity'] != 25_000_000
            or study['configuration']['date_range'] != {'start': '2025-08-07', 'end': '2026-03-31'}):
        raise ValueError('Candidate, capital or IS boundary differs from the sealed experiment')
    for item in seal['candidate_batteries']['k_marginal'].values():
        bound(item['completion'])
        checked(Path(item['completion']['path']).parent, True)
    acceptance = read(oos / 'acceptance.json')
    if acceptance['finalist'] != seal['finalist'] or acceptance['acceptance']['pass'] is not False:
        raise ValueError('Expected the retained failed OOS acceptance evidence')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_before_k_marginal_round1')
    archive = archive_round(strategy_dir, stamp)
    shutil.copy2(pointer, archive / 'portfolio_synergy_selected.json')
    output = strategy_dir / 'round_1'
    output.mkdir()
    context = {'version': 'portfolio-native-round-registration-v1', 'round': 1,
        'registered_utc': datetime.now(timezone.utc).isoformat(),
        'assignment': 'Explicit user assignment of the sealed candidate as research round_1',
        'recipe': seal['finalist'], 'settings': seal['recipe'], 'initial_equity_krw': 25_000_000,
        'date_range': study['configuration']['date_range'],
        'selection': binding(selection_path), 'study': binding(study_path),
        'manifest': binding(study['manifest']), 'protocol': binding(parent / 'protocol.json'),
        'oos_acceptance': binding(oos / 'acceptance.json'),
        'archive_receipt': binding(archive / 'archive_receipt.json'),
        'archived_manifest': binding(archive / 'rounds_manifest.json'),
        'previous_pointer': binding(archive / 'portfolio_synergy_selected.json'),
        'selected_pointer_path': str(pointer),
        'replay_scenarios': ['baseline', 'combined'], 'new_search_trials': 0,
        'live_deployment_enabled': False, 'oos_acceptance_pass': False}
    _atomic_write_json(context, output / 'round_registration.json')
    _atomic_write_json({'complete': False, 'status': 'registered_awaiting_full_native_replays',
                        'date_range': context['date_range']}, output / 'round_final_diagnostics_status.json')
    # Remove the stale active entry immediately; final report registers completed metrics.
    manifest = read(strategy_dir / 'rounds_manifest.json')
    former = [r for r in manifest['rounds'] if r['round'] == 1]
    manifest['rounds'] = [r for r in manifest['rounds'] if r['round'] != 1]
    manifest.setdefault('archived_rounds', []).append({'round': 1, 'archived': True,
        'path': str(archive / 'round_1'), 'receipt': context['archive_receipt'], 'original_entries': former})
    manifest['pending_round'] = {'round': 1, 'registration': binding(output / 'round_registration.json')}
    _atomic_write_json(manifest, strategy_dir / 'rounds_manifest.json')
    return output


def replay(output):
    output = Path(output).resolve()
    context = read(output / 'round_registration.json')
    study = bound(context['study'])
    (output / 'logs').mkdir(exist_ok=True)

    def one(scenario):
        case = output / 'diagnostics' / scenario
        if case.exists():
            return checked(case, True)  # Incomplete or altered outputs fail closed.
        command = [sys.executable, '-B', '-u', str(Path(study['code']) / 'scripts/portfolio_synergy_study.py'),
            'replay', '--study', context['study']['path'], '--recipe', context['recipe'],
            '--scenario', scenario, '--output', str(case), '--full-audit']
        print(json.dumps({'replay_started': scenario, 'output': str(case)}), flush=True)
        with (output / 'logs' / (scenario + '.log')).open('x', encoding='utf-8') as log:
            done = subprocess.run(command, cwd=study['code'], stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                     'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        _atomic_write_json({'command': command, 'returncode': done.returncode},
                           output / 'logs' / (scenario + '.receipt.json'))
        if done.returncode:
            raise RuntimeError('Native replay failed: ' + scenario)
        value = checked(case, True)
        print(json.dumps({'replay_completed': scenario,
                          **{k: value[k] for k in ('return_pct', 'drawdown_pct', 'trades', 'audit_status')}}), flush=True)
        return value

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(one, context['replay_scenarios']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'replay'))
    p.add_argument('--parent', type=Path)
    p.add_argument('--oos', type=Path)
    p.add_argument('--strategy-dir', type=Path, default=REPO / 'data/backtests/output/portfolio_synergy')
    p.add_argument('--pointer', type=Path, default=REPO / 'config/research/portfolio_synergy_selected.json')
    p.add_argument('--round-dir', type=Path)
    args = p.parse_args()
    if args.command == 'prepare':
        if not args.parent or not args.oos:
            p.error('prepare requires --parent and --oos')
        print(prepare(args.parent, args.oos, args.strategy_dir, args.pointer))
    else:
        if not args.round_dir:
            p.error('replay requires --round-dir')
        replay(args.round_dir)


if __name__ == '__main__':
    main()

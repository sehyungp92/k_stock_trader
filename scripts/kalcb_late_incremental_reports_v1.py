"""Registered, read-only full screening reports; no native study execution."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import os
import subprocess
import sys
import psutil

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound, verify, env
from backtests.strategies.kalcb.late_incremental_full_diagnostics_v1 import authenticated_protocol


def exact_alive(identity):
    try:
        p = psutil.Process(identity['pid'])
        if abs(p.create_time() - identity['process_created']) >= .01:
            return False
        if p.cmdline() != identity['command']:
            raise ValueError('Process identity command mismatch')
        return True
    except psutil.NoSuchProcess:
        return False


def render(output, name, period, scenario):
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    from backtests.strategies.kalcb.late_incremental_full_diagnostics_v1 import LateIncrementalScreeningReportPlugin
    plugin = LateIncrementalScreeningReportPlugin(output, name, period, scenario)
    manager = RoundManager('stock', 'kalcb', base_dir=output / 'framework' / name / period / scenario)
    folder = manager.round_path(1)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Preserve existing report; do not restart')
    PhaseRunner(plugin, manager.get_round_dir(1), round_name=name + '_' + period + '_' + scenario,
        round_manager=manager, round_num=1, max_rounds=1, max_retries=0, max_diagnostic_retries=0).run_all_phases()
    check = read(folder / 'diagnostic_integrity.json')
    if not check['passed'] or not check['native_evidence_unchanged']:
        raise ValueError('Full diagnostic integrity failure')
    write(output / 'reports' / name / period / scenario / 'completion.json', {'complete': True,
        'report': bound(folder / 'round_final_diagnostics.txt'),
        'full_diagnostics': bound(folder / 'round_final_full_diagnostics.json'),
        'integrity': bound(folder / 'diagnostic_integrity.json'),
        'framework_status': bound(folder / 'round_final_diagnostics_status.json'), 'economic_promotion': False})


def run(output):
    p = authenticated_protocol(output)
    for identity in p['closed_supervisors']:
        if exact_alive(identity):
            raise ValueError('Prerequisite supervisor still alive')
    if (output / 'supervisor_start.json').exists():
        raise ValueError('Preserve prior report supervisor')
    own = {'pid': os.getpid(), 'process_created': psutil.Process().create_time(), 'command': psutil.Process().cmdline()}
    write(output / 'supervisor_start.json', own)
    temporary = output / 'temporary'; temporary.mkdir(exist_ok=True)
    child_env = env(REPO); child_env.update(TEMP=str(temporary), TMP=str(temporary))
    progress = []
    try:
        for name in p['recipes']:
            for period in ('is', 'oos'):
                for scenario in ('baseline', 'combined'):
                    log = output / (name + '_' + period + '_' + scenario + '.log')
                    command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), 'render', '--output', str(output),
                               '--name', name, '--period', period, '--scenario', scenario]
                    with log.open('x', encoding='utf-8') as stream:
                        child = subprocess.Popen(command, cwd=REPO, env=child_env, stdout=stream,
                            stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                        record = {'pid': child.pid, 'process_created': psutil.Process(child.pid).create_time(), 'command': command}
                        write(output / 'status.json', {**own, 'stage': 'render', 'completed': len(progress),
                            'name': name, 'period': period, 'scenario': scenario, 'child': record, 'log': str(log)})
                        code = child.wait()
                    if code:
                        raise RuntimeError('Registered report failed; preserve evidence and repair in successor: ' + str(log))
                    receipt = output / 'reports' / name / period / scenario / 'completion.json'
                    if not read(receipt)['complete']:
                        raise ValueError('Incomplete full diagnostic report')
                    progress.append({'name': name, 'period': period, 'scenario': scenario, 'completion': bound(receipt)})
                    write(output / 'progress.json', progress)
                    print(f'completed {len(progress)}/32 {name} {period} {scenario}', flush=True)
        authenticated_protocol(output)
        write(output / 'completion.json', {'complete': True, 'reports': progress, 'study_complete': False,
            'final_report_review_required': True, 'canonical_replaced': False, 'production_enabled': False})
        write(output / 'status.json', {**own, 'stage': 'full_reports_complete_pending_final_review', 'completed': 32, 'child': None})
    except Exception as error:
        write(output / 'status.json', {**own, 'stage': 'attention_required', 'error': repr(error), 'completed': len(progress)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('run', 'render'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--name'); parser.add_argument('--period'); parser.add_argument('--scenario')
    args = parser.parse_args()
    if args.stage == 'run': run(args.output)
    else: render(args.output, args.name, args.period, args.scenario)

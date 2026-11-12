"""Register eight new full stress reports and eight authenticated V25 reports."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, sys
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound, verify
from backtests.strategies.kalcb.pending_funding_qualification_assessment_v1 import completion_gate, require
from backtests.strategies.kalcb.research_process_v1 import exact_process
from backtests.strategies.kalcb.pending_funding_qualification_report_checks_v1 import check_report


def register(root, output, tests):
    rows = completion_gate(root)
    ar = root/'assessment_v1'
    require(not exact_process(read(ar/'supervisor_start.json')), 'Assessment still running')
    done = read(ar/'completion.json')
    require(done['complete'], 'Assessment incomplete')
    verify([done['assessment']])
    a = read(done['assessment']['path'])
    require(a['accounted_cases'] == 16 and a['fresh_native_cases'] == 8 and
        a['authenticated_reuse'] == 8 and a['exact_parent_bridges'] == 8, 'Wrong assessment scope')
    verify(a['bindings'])
    t = read(tests)
    require(t['exit_code'] == 0, 'Report tests failed')
    verify(t['programs']+[t['log']])
    p = read(root/'protocol.json')
    verify(p['programs']+p['bindings'])
    old = root.parent/'v25_pending_funding_screen'/'full_reports_v1'
    old_verification = old/'verification_v1/full_report_verification.json'
    require(read(old_verification)['passed'], 'Original V25 reports not verified')
    require(not exact_process(read(old/'supervisor_start.json')), 'Old reports still running')
    profiles = {n: {period: {} for period in ('is', 'oos')} for n in p['recipes']}
    bindings = p['bindings']+a['bindings']+[bound(f) for f in (root/'protocol.json', root/'stage_completion.json',
        root/'progress.json', ar/'protocol.json', ar/'assessment.json', ar/'completion.json', tests, old_verification)]
    reused = {}
    for row in rows:
        name, period, scenario = (row[k] for k in ('name','period','scenario'))
        key = '/'.join((name, period, scenario))
        profiles[name][period][scenario] = {'folder': row['folder'], 'input_profile': row['input_profile'],
            'reused_native': row['reused']}
        if row['reused']:
            receipt = old/'reports'/name/period/scenario/'completion.json'
            check = check_report(receipt, key, a['accounts'][key], p['recipes'][name], row['folder'], True)
            bindings += check['artifacts']
            reused[key] = {'completion': bound(receipt), 'verification': check}
    require(len(reused) == 8, 'Wrong reused report count')
    paths = {Path(x['path']) for x in p['programs']}
    for directory in ('backtests/auto/shared','backtests/strategies/common','backtests/strategies/kalcb',
                      'backtests/engine','strategy_common','strategy_kalcb'):
        paths.update(f for f in (REPO/directory).rglob('*.py') if not f.name.startswith('olr_'))
    paths.update(REPO/f for f in ('scripts/kalcb_pending_funding_qualification_reports_v1.py',
        'scripts/kalcb_pending_funding_qualification_reports_register_v1.py',
        'scripts/kalcb_pending_funding_qualification_report_verify_v1.py', 'scripts/kalcb_executed_value_replay.py',
        'tests/backtests/strategies/test_kalcb_pending_funding_qualification_reports_v1.py'))
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    imported = []
    for module in list(sys.modules.values()):
        file = getattr(module,'__file__',None)
        if file and Path(file).suffix == '.py' and Path(file).resolve().is_relative_to(REPO):
            require(not Path(file).name.startswith('olr_'), 'Unexpected unrelated dependency')
            imported.append(bound(Path(file).resolve()))
            paths.add(Path(file).resolve())
    programs = [bound(f) for f in sorted(paths)]
    bindings = list({x['path']:x for x in bindings}.values())
    verify(programs+bindings)
    output.mkdir(parents=True, exist_ok=False)
    write(output/'protocol.json', {'version':'pending-funding-qualification-full-reports-v1',
        'registered_at_utc':datetime.now(timezone.utc).isoformat(), 'native_case_count':16, 'report_count':16,
        'fresh_report_count':8, 'reused_report_count':8, 'reused_reports':reused,
        'recipes':p['recipes'], 'profiles':profiles, 'programs':programs, 'bindings':bindings,
        'closed_supervisors':[read(root/'supervisor_start.json'),read(ar/'supervisor_start.json')],
        'native_protocol':bound(root/'protocol.json'), 'assessment':bound(ar/'assessment.json'),
        'assessment_completion':bound(ar/'completion.json'), 'tests':bound(tests),
        'imported_application_modules':imported,
        'scope':'Eight fresh stress full reports through locked shared PhaseRunner and eight original unchanged V25 reports. Four-stress evidence only; funded/independent/live qualification pending.'})


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('root','output','tests'): ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args(); register(a.root,a.output,a.tests)

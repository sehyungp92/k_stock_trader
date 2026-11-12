"""Freeze sixteen read-only reports after completed V25 native and assessment."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify
from backtests.strategies.kalcb.pending_funding_assessment_v1 import completion_gate,exact_process,require


def register(root,output,tests):
    completion_gate(root);ar=root/'assessment_v1'
    require(not exact_process(read(ar/'supervisor_start.json')),'Assessment still running')
    done=read(ar/'completion.json');require(done['complete'] and done['native_cases']==16 and done['exact_parent_bridges']==8,'Incomplete assessment')
    verify([done['assessment']]);a=read(ar/'assessment.json');verify(a['bindings'])
    t=read(tests);require(t['exit_code']==0,'Failed report tests');verify(t['programs']+[t['log']])
    preflight=root/'full_report_preflight_v1/completion.json';pre=read(preflight)
    require(pre['complete'] and set(pre['statuses'].values())=={'audited_pending_funding_screening_only_no_promotion'},'Actual report preflight missing')
    verify([pre['program'],pre['report'],pre['integrity']])
    p=read(root/'protocol.json');verify(p['programs']+p['bindings'])
    profiles={n:{period:{s:{'folder':str(root/'cases'/n/period/s),'input_profile':bound(root/'cases'/n/period/'input_profile.json')}
        for s in ('baseline','combined')} for period in ('is','oos')} for n in p['recipes']}
    bindings=p['bindings']+a['bindings']+[bound(f) for f in (root/'protocol.json',root/'stage_completion.json',root/'progress.json',
        ar/'protocol.json',ar/'assessment.json',ar/'completion.json',tests,preflight)]
    paths={Path(x['path']) for x in p['programs']}
    for directory in ('backtests/auto/shared','backtests/strategies/common','backtests/strategies/kalcb','backtests/engine','strategy_common','strategy_kalcb'):
        paths.update(f for f in (REPO/directory).rglob('*.py') if not f.name.startswith('olr_'))
    paths.update(REPO/f for f in ('scripts/kalcb_pending_funding_reports_v1.py','scripts/kalcb_pending_funding_reports_register_v1.py',
        'scripts/kalcb_executed_value_replay.py','tests/backtests/strategies/test_kalcb_pending_funding_reports_v1.py'))
    from backtests.strategies.kalcb import pending_funding_full_diagnostics_v1
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    imported=[]
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).suffix=='.py' and Path(file).resolve().is_relative_to(REPO):
            require(not Path(file).name.startswith('olr_'),'Unexpected unrelated report dependency')
            imported.append(bound(Path(file).resolve()));paths.add(Path(file).resolve())
    programs=[bound(f) for f in sorted(paths)];bindings=list({x['path']:x for x in bindings}.values());verify(programs+bindings)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'protocol.json',{'version':'pending-funding-full-reports-v1','registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'native_case_count':16,'report_count':16,'recipes':p['recipes'],'profiles':profiles,'programs':programs,'bindings':bindings,
        'closed_supervisors':[read(root/'supervisor_start.json'),read(ar/'supervisor_start.json')],
        'native_protocol':bound(root/'protocol.json'),'assessment':bound(ar/'assessment.json'),'assessment_completion':bound(ar/'completion.json'),
        'tests':bound(tests),'imported_application_modules':imported,'scope':'Read-only existing-engine reports through locked shared PhaseRunner. No native replay, promotion or next experiment.'})


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('root','output','tests'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();register(a.root,a.output,a.tests)

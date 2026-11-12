"""Freeze read-only full reports after completed native and joint assessment gates."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import sys

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify
from backtests.strategies.kalcb.late_signal_reuse_v1 import require_closed
from backtests.strategies.kalcb.exit_qualification_assessment_v1 import require_design


def profile_design(root,assessment):
    native=read(root/'protocol.json');base=root.parent
    competitor=read(Path('E:/CodexResearch/k_stock_trader/kalcb_selective_recovery_20261010/v7_funded_qualification/protocol.json'))
    recipes={};profiles={}
    for mode,names,scenarios in (
        ('compounding',('control','prior100_failure6','prior100_failure8'),('baseline','participation','cost','combined')),
        ('fixed',('control','prior100_failure8'),('baseline','combined')),
        ('competing',('prior125',),('baseline','participation','cost','combined')),
        ('rejected',('prior125_failure6','prior125_failure8'),('baseline','combined'))):
        for name in names:
            key=mode+'_'+name
            recipes[key]={'prior':125 if mode in ('competing','rejected') else 100,
                'hold':int(name[-1]) if 'failure' in name else 0,'mode':mode,
                'role':{'compounding':'All-four-stress native parent/backout/partial candidate evidence',
                    'fixed':'Funded nominal-allocation comparison, no replenishment',
                    'competing':'Unchanged prior125 competing parent',
                    'rejected':'Retained rejected prior125 exit from V12 screening'}[mode]}
            profiles[key]={}
            for period in ('is','oos'):
                profiles[key][period]={}
                for scenario in scenarios:
                    if mode=='competing':
                        folder=Path(native['competing_prior125_references'][period][scenario])
                        entry={'folder':str(folder),'item':competitor['profiles']['expanding_removal_prior125'][period],
                            'assessment_kind':'joint','assessment_key':'/'.join((mode,name,period,scenario))}
                    elif mode=='rejected':
                        folder=base/'v12_mechanisms/cases'/name/period/scenario
                        entry={'folder':str(folder),'input_profile':bound(folder.parent/'input_profile.json'),
                            'assessment_kind':'screening','assessment_key':'/'.join((name,period,scenario))}
                    else:
                        folder=root/mode/'cases'/name/period/scenario
                        entry={'folder':str(folder),'input_profile':bound(folder/'qualification_input_profile.json'),
                            'assessment_kind':'joint','assessment_key':'/'.join((mode,name,period,scenario))}
                    if entry['assessment_kind']=='joint' and entry['assessment_key'] not in assessment['accounts']:
                        raise ValueError('Unassessed report case')
                    profiles[key][period][scenario]=entry
    if sum(len(c) for ps in profiles.values() for c in ps.values())!=48:raise ValueError('Incomplete report design')
    return recipes,profiles


def register(output,root,tests,decision):
    if output.exists():raise ValueError('Preserve existing report registration')
    require_closed(read(root/'supervisor_start.json'));require_design(root)
    ar=root/'assessment_v1';require_closed(read(ar/'supervisor_start.json'))
    completion=read(ar/'completion.json');verify([completion['assessment']])
    if not completion['complete'] or completion['source_account_cases']!=48 or completion['exact_bridges']!=20:
        raise ValueError('Incomplete joint assessment')
    a=read(ar/'assessment.json');verify(a['bindings'])
    test=read(tests)
    if test['exit_code']!=0:raise ValueError('Failed tests')
    verify(test['programs']+[test['log']])
    recipes,profiles=profile_design(root,a)
    retained=read(root.parent/'v12_mechanisms/assessment_v1/assessment.json')
    bindings=[bound(f) for f in (root/'protocol.json',root/'progress.json',root/'stage_completion.json',
        ar/'protocol.json',ar/'assessment.json',ar/'completion.json',tests,decision,
        root.parent/'v12_mechanisms/assessment_v1/assessment.json',
        root.parent/'v15_incremental/full_reports_v1/verification_v2/full_report_verification.json')]
    bindings+=a['bindings']
    for periods in profiles.values():
        for cases in periods.values():
            for entry in cases.values():
                item=read(entry['input_profile']['path']) if 'input_profile' in entry else entry['item']
                if 'input_profile' in entry:bindings.append(entry['input_profile'])
                bindings += [item[k] for k in ('manifest','plan','policy')]
                if entry['assessment_kind']=='screening':
                    if entry['assessment_key'] not in retained['accounts']:raise ValueError('Missing rejected trial')
                    folder=Path(entry['folder'])
                    bindings += [bound(folder/f) for f in ('result.json','execution_audit.json','independent_account_verification.json',
                        'daily.json','equity.json','fills.json','trades.json','orders.jsonl.gz','decisions.jsonl.gz','mechanism_evidence.json')]
    native=read(root/'protocol.json');verify(native['programs']+native['bindings'])
    programs=list(native['programs'])
    # All shared engine/report and KALCB sources are bound. OLR modules are not
    # KALCB imports and remain outside this new report dependency contract.
    for directory in ('backtests/auto/shared','backtests/strategies/common','backtests/strategies/kalcb',
                      'backtests/engine','strategy_common','strategy_kalcb'):
        programs += [bound(f) for f in sorted((REPO/directory).rglob('*.py')) if not f.name.startswith('olr_')]
    programs += [bound(REPO/f) for f in ('scripts/kalcb_exit_joint_reports_v1.py','scripts/kalcb_exit_joint_reports_register_v1.py',
        'scripts/kalcb_executed_value_replay.py','tests/backtests/strategies/test_kalcb_exit_joint_reports_v1.py')]
    from backtests.strategies.kalcb import exit_joint_full_diagnostics_v1
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    imported=[]
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).suffix=='.py' and Path(file).resolve().is_relative_to(REPO):
            if Path(file).name.startswith('olr_'):raise ValueError('Unexpected OLR report dependency')
            imported.append(bound(Path(file).resolve()))
    programs+=imported
    programs=list({x['path']:x for x in programs}.values());bindings=list({x['path']:x for x in bindings}.values())
    verify(programs+bindings)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'protocol.json',{'version':'kalcb-exit-joint-full-reports-v1','registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'native_case_count':40,'report_count':48,'recipes':recipes,'profiles':profiles,'programs':programs,'bindings':bindings,
        'closed_supervisors':[read(root/'supervisor_start.json'),read(ar/'supervisor_start.json')],
        'native_protocol':bound(root/'protocol.json'),'assessment':bound(ar/'assessment.json'),
        'assessment_completion':bound(ar/'completion.json'),'retained_screening_assessment':bound(root.parent/'v12_mechanisms/assessment_v1/assessment.json'),
        'economic_decision':bound(decision),'tests':test,'imported_application_modules':imported,
        'scope':'Read-only existing full diagnostic engine through shared PhaseRunner; all48 scenario reports retained, no automatic economic promotion. No native replay or changed policy.',
        'dependency_scope':'Shared engine/report and KALCB programs plus imported application closure; unrelated OLR programs excluded and untouched.'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('output','root','tests','decision'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();register(args.output.resolve(),args.root.resolve(),args.tests.resolve(),args.decision.resolve())

"""Register and run one read-only assessment of completed exit qualification."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,os,sys
import psutil

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify
from backtests.strategies.kalcb.late_signal_reuse_v1 import require_closed
from backtests.strategies.kalcb.exit_qualification_assessment_v1 import require_design,assess


def register(output,root,tests):
    identity=read(root/'supervisor_start.json');require_closed(identity)
    rows=require_design(root);native=read(root/'protocol.json');verify(native['programs']+native['bindings'])
    test=read(tests)
    if test['exit_code']!=0:raise ValueError('Failed assessment tests')
    verify(test['programs']+[test['log']])
    if output.exists():raise ValueError('Preserve existing assessment')
    base=root.parent;competing=Path('E:/CodexResearch/k_stock_trader/kalcb_selective_recovery_20261010/v7_funded_qualification/protocol.json')
    files=[root/f for f in ('protocol.json','study.json','progress.json','stage_completion.json','supervisor_start.json','economic_decision.json')]
    files += [base/'v12_mechanisms/assessment_v1/assessment.json',competing,tests,
        base/'v15_incremental/full_reports_v1/verification_v2/full_report_verification.json']
    for row in rows:
        files +=[Path(row[k]['path']) for k in ('result','input_profile','funding','evidence')]
    programs=native['programs']+[bound(REPO/f) for f in (
        'scripts/kalcb_exit_joint_assessment_v1.py','backtests/strategies/kalcb/exit_qualification_assessment_v1.py',
        'backtests/strategies/kalcb/late_signal_reuse_v1.py','backtests/strategies/kalcb/fixed_allocation_funding_v1.py',
        'backtests/strategies/kalcb/late_artifact_validation_v2.py','backtests/strategies/kalcb/population_research_io_v1.py',
        'tests/backtests/strategies/test_kalcb_exit_joint_assessment_v1.py')]
    programs=list({p['path']:p for p in programs}.values());verify(programs)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'protocol.json',{'version':'kalcb-exit-joint-assessment-v1','registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'native_root':str(root),'native_supervisor':identity,'programs':programs,'bindings':[bound(f) for f in files],
        'competing_protocol':bound(competing),'screening_assessment':bound(base/'v12_mechanisms/assessment_v1/assessment.json'),
        'tests':test,'scope':'40qualified native cells plus8unchanged prior125 comparison cells; all4stresses, hold6/0 backouts, funded deltas, actual intervention/funding/tails/concentration/MTM and paired5/10session10000repseed20261011 sensitivity. Read-only; no automatic promotion or full-report completion claim.'})


def run(output):
    reg=read(output/'protocol.json');verify(reg['programs']+reg['bindings']);require_closed(reg['native_supervisor'])
    if (output/'supervisor_start.json').exists():raise ValueError('Preserve prior assessment attempt')
    own={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':psutil.Process().cmdline()}
    write(output/'supervisor_start.json',own);write(output/'status.json',{**own,'stage':'read_only_joint_assessment','child':None})
    try:
        assess(output)
        write(output/'status.json',{**own,'stage':'assessment_complete_economic_review_and_full_reports_pending','study_complete':False,'child':None})
    except Exception as error:
        write(output/'status.json',{**own,'stage':'attention_required','error':repr(error),'child':None});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('register','run'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--root',type=Path);parser.add_argument('--tests',type=Path)
    args=parser.parse_args()
    if args.stage=='register':register(args.output.resolve(),args.root.resolve(),args.tests.resolve())
    else:run(args.output.resolve())

"""Verify standard reporting/account analysis before any full treatment search."""
from pathlib import Path
import sys,time,traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.round3_native_io_v1 import ROOT,read,write,bound
from backtests.strategies.kalcb.round3_analysis_v1 import account,funding_chronology,compare,finer_marks
from backtests.strategies.kalcb.round3_contract_v1 import scalar
from backtests.strategies.kalcb.round3_reports_v1 import report
OUT=ROOT/'reporting_preflight_v1'
def main():
    OUT.mkdir(exist_ok=False);start=time.monotonic();bridge=read(ROOT/'preflight_v2/bridge_completion.json')
    cells={r['item']['period']+'/'+r['process']['scenario']:account(r) for r in bridge['receipts']}
    funding={r['item']['period']+'/'+r['process']['scenario']:funding_chronology(r) for r in bridge['receipts']}
    value=scalar(cells);assert abs(value['R']-37.714303)<.000001
    null=compare(cells,cells);assert all(v['planning_MDE_pp']==0 for v in null.values())
    r=next(r for r in bridge['receipts'] if r['item']['period']=='is' and r['process']['scenario']=='baseline')
    cfg=dict(initial_equity=1e8,date_range=r['profile']['date_range'],promotion_status='preflight_only',production_enabled=False)
    summary=report(r,'baseline',OUT/'diagnostics',cfg,read(r['profile']['configuration']))
    minute=finer_marks(r,OUT/'minute_marks.json')
    write(OUT/'completion.json',dict(passed=True,seconds=time.monotonic()-start,program=bound(__file__),score=value,
        native_reconstruction=funding,paired_identical_control=null,minute=minute,
        report=bound(OUT/'diagnostics/round_final_diagnostics.txt'),standard_report_metrics=summary.get('final_metrics')))
    print('Report, funding, score and finer-mark preflight passed',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise

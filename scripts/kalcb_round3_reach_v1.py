"""Native feasibility only; never computes or selects economic scores."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import sys,time,traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.round3_native_io_v1 import ROOT,execute,read,write,bound
from backtests.strategies.kalcb.round3_analysis_v1 import native_reach

OUT=ROOT/'reach_v1';RUNTIME=ROOT/'native_v5'

def main():
    bridge=read(ROOT/'preflight_v2/bridge_completion.json')
    if not bridge['passed'] or len(bridge['receipts'])!=8:raise ValueError('Baseline preflight incomplete')
    OUT.mkdir(exist_ok=False);profiles=read(RUNTIME/'preparation.json')['profiles'];arms=read(RUNTIME/'arms.json')
    write(OUT/'registration.json',dict(program=bound(__file__),analysis=bound(Path(__file__).resolve().parents[1]/'backtests/strategies/kalcb/round3_analysis_v1.py'),
        binding=bound(Path(__file__).resolve().parents[1]/'docs/optimisation/kalcb/experiments/kalcb-exp-0042-executable-binding.md'),
        max_workers=2,economic_metrics_inspected=False,scope='All finite arms, baseline scenario, fixed IS/OOS prefixes; decisions/actions/fills only'))
    receipts={};progress=0
    for names in [['v6_h0'],[n for n in arms if n!='v6_h0']]:
        jobs=[(p,n) for n in names for p in ('is','oos')]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(execute,profiles[p][n+'/100000000'],'baseline',OUT/'accounts'/p/n,smoke=True):(p,n) for p,n in jobs}
            for future in as_completed(futures):
                p,n=futures[future];receipts[p+'/'+n]=future.result();progress+=1
                write(OUT/f'progress_{progress:02d}.json',dict(completed=progress,planned=2*len(arms),period=p,arm=n,
                    receipt=bound(OUT/'accounts'/p/(n+'_acceptance.json'))))
                print('Native reach completed '+p+'/'+n,flush=True)
    delay=execute(profiles['oos']['v6_h0/100000000'],'baseline',OUT/'delay_oos',smoke=True,delay=5)
    delay_change=native_reach(OUT/'delay_oos',OUT/'accounts/oos/v6_h0',smoke=True)
    if not delay_change['changed_physical_fills']:raise ValueError('Native delay has no fill reach')
    results={}
    for name,arm in arms.items():
        reach={p:native_reach(OUT/'accounts'/p/name,OUT/'accounts'/p/'v6_h0',smoke=True) for p in ('is','oos')}
        effective=any(r['changed_action_records'] and r['changed_physical_fills'] for r in reach.values())
        results[name]=dict(eligible_for_full_evaluation=name=='v6_h0' or effective,periods=reach,
            verdict='native_action_and_fill_reached' if effective else ('control' if name=='v6_h0' else 'inactive_in_registered_prefix'),
            coverage_gap=not effective and name!='v6_h0',whole_window_inactivity_proven=False,
            effective_policy_hash=arm['effective_policy_hash'],alias_of=arm.get('alias_of'))
    write(OUT/'completion.json',dict(complete=True,arms=results,receipts=receipts,delay_receipt=delay,delay_reach=delay_change,
        economic_metrics_inspected=False,max_workers=2,registration=bound(OUT/'registration.json')))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise

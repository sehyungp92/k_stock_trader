"""Registered integrity preflight. No selectable policy search/adoption here."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import os
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.round3_native_io_v1 import ROOT,NATIVE,SOURCE,execute,read,write,bound
SCENARIOS=('baseline','participation','cost','combined')
OUT=ROOT/'preflight_v1'


def main():
    OUT.mkdir(exist_ok=False)
    write(OUT/'registration.json',dict(program=bound(__file__),runtime=bound(NATIVE/'preparation.json'),
        executor=bound(Path(__file__).resolve().parents[1]/'backtests/strategies/kalcb/round3_native_io_v1.py'),
        binding=bound(Path(__file__).resolve().parents[1]/'docs/optimisation/kalcb/experiments/kalcb-exp-0042-executable-binding.md'),
        bridge_cases=8,hydration_cases=['is/baseline','oos/combined'],max_workers=2,
        scope='Bridge first; no treatment or automatic full round launch'))
    write(OUT/'started.json',dict(pid=os.getpid(),at_utc=datetime.now(timezone.utc).isoformat()))
    profiles=read(NATIVE/'preparation.json')['profiles'];receipts=[]
    groups=[[('is','baseline'),('oos','combined')],
            [(p,s) for p in ('is','oos') for s in SCENARIOS if (p,s) not in [('is','baseline'),('oos','combined')]]]
    for index,cells in enumerate(groups):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(execute,profiles[p]['v6_h0/100000000'],s,OUT/'accounts'/f'{p}_{s}',
                hydrate=index==0,bridge=SOURCE/'cases'/f'combined_{p}_{s}'):(p,s) for p,s in cells}
            for future in as_completed(futures):
                receipt=future.result()
                if not receipt['passed']:raise ValueError('Baseline preflight failed')
                receipts.append(receipt)
                write(OUT/f'progress_{len(receipts):02d}.json',dict(accepted=len(receipts),planned=8,last=futures[future]))
                print('Accepted full baseline bridge: '+str(futures[future]),flush=True)
    write(OUT/'bridge_completion.json',dict(passed=True,receipts=receipts,accounts=8,full_ordered_path_equality=True,
        complete_journal_reconstruction=True,disk_hydration_two_windows=True,
        remaining=['native arm smoke/reach','standard diagnostics preflight','complete cost estimate','shared round launch']))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise

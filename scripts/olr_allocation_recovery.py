"""Run registered OLR allocation research through the shared PhaseRunner."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.olr.allocation_recovery import prepare, AllocationRecoveryPlugin
from backtests.auto.shared.phase_runner import PhaseRunner

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','screen','qualify','accept'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent',type=Path)
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    if a.command=='prepare':
        prepare(Path(__file__).resolve().parents[1],a.parent,a.root.resolve())
    else:
        plugin=AllocationRecoveryPlugin(a.root.resolve(),a.command,a.workers)
        PhaseRunner(plugin,a.root.resolve()/('phased_'+a.command),round_name='OLR allocation '+a.command,
            max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()

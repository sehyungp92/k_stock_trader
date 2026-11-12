"""Run the finite calibrated KALCB study through the shared optimization lifecycle."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.strategies.kalcb.calibrated_recovery_phase import KALCBCalibratedRecoveryPlugin

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);a=p.parse_args()
    plugin=KALCBCalibratedRecoveryPlugin(a.study)
    PhaseRunner(plugin,a.study.parent/'phased',round_name='Calibrated selection: profit retention and admission',
        max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()

if __name__=='__main__':main()

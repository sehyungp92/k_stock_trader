"""Continue the current authorized run at native replay boundaries."""
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import olr_round1_optimization_v1 as s


def main():
    while not (s.ROOT/'grid_completion.json').exists():
        if (s.ROOT/'STOP_FOLLOWUP.json').exists():raise RuntimeError('Follow-up dispatch stopped')
        time.sleep(15)
    for script,args in (
        ('olr_round1_optimization_results_v1.py',['center']),
        ('olr_continuous_panel_study_v1.py',['run']),
        ('olr_round1_optimization_qualify_v1.py',[])):
        if (s.ROOT/'STOP_FOLLOWUP.json').exists():raise RuntimeError('Follow-up dispatch stopped')
        print('Next registered stage',script,args,flush=True)
        subprocess.run([sys.executable,'-B','-u',str(REPO/'scripts'/script),*args],cwd=REPO,check=True)


if __name__=='__main__':main()

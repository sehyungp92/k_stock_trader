"""Run the frozen C17 plugin through shared RoundManager and PhaseRunner."""
from pathlib import Path
import argparse,json,os,sys,threading,time,traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.round3_native_io_v1 import read,write,bound
from backtests.strategies.kalcb.round3_service_v1 import RoundService
from backtests.strategies.kalcb.round3_plugin_v1 import C17RoundPlugin
from backtests.strategies.kalcb.round3_closeout_v1 import finish
from backtests.auto.shared.round_manager import RoundManager
from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.strategies.kalcb.round3_monitor_v1 import check

def main():
    p=argparse.ArgumentParser();p.add_argument('--registration',required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    reg=Path(a.registration);data=read(reg);out=reg.parent
    service=RoundService(reg);service.check_frozen();plugin=C17RoundPlugin(service)
    manager=RoundManager('stock','kalcb',base_dir=Path(data['repository'])/'data/backtests/output')
    canonical=manager.round_path(3)
    if a.preflight:
        runner=PhaseRunner(plugin,out/'framework_preflight',max_rounds=1,min_delta=0,max_retries=0,max_diagnostic_retries=0,round_manager=manager,round_num=3)
        initial=runner._initial_baseline_mutations();proof=runner._validate_initial_baseline(initial);metrics=plugin.compute_final_metrics(initial)
        plugin.validate_archived_execution(metrics)
        specs={str(n):[c.name for c in plugin.get_phase_spec(n,runner.load_state()).candidates] for n in (1,2,3,4)}
        result=dict(passed=True,baseline_integrity=proof,metrics=metrics,candidate_sets=specs,
            source=bound(__file__),registration=bound(reg),native_treatments_run=0,max_workers=2)
        write(out/'framework_preflight.json',result);print('Shared framework preflight passed',flush=True);return
    if read(out/'framework_preflight.json').get('passed') is not True:raise ValueError('No framework preflight')
    if canonical.exists():raise ValueError('Preserve existing round directory; explicit recovery required')
    canonical.mkdir(parents=True,exist_ok=False)
    import psutil
    proc=psutil.Process(os.getpid());write(out/'launch.json',dict(pid=proc.pid,created=proc.create_time(),registration=bound(reg),canonical=str(canonical),max_workers=2))
    stop=threading.Event()
    def monitor():
        while not stop.wait(1800):
            try:check(reg)
            except Exception as exc:
                with (out/'monitor_errors.jsonl').open('a') as f:f.write(json.dumps(dict(error=repr(exc),at=time.time()))+'\n')
    thread=threading.Thread(target=monitor,daemon=True);thread.start();check(reg)
    try:
        runner=PhaseRunner(plugin,canonical,round_name='C17 native funded round; frozen C16',max_rounds=1,min_delta=0,
            max_retries=0,max_diagnostic_retries=0,round_manager=manager,round_num=3)
        state=runner.run_all_phases();result=finish(service,state,canonical)
        print(json.dumps(result,default=str),flush=True)
    except Exception as exc:
        write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),no_automatic_retuning=True));raise
    finally:stop.set();check(reg)

if __name__=='__main__':main()

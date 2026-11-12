"""Registered observational native support checks; no new economic selection."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.kalcb_full_oos_recovery import ROOT,read,write,bound,env
from scripts.kalcb_alignment_walk_forward_v3 import verify_bindings
from scripts.kalcb_recovery_qualification_round import DEST as Q, PROFILES
DEST=Path('E:/CodexResearch/k_stock_trader/kalcb_atr_intervention_20261010/v2')
PROGRAM=REPO/'scripts/kalcb_atr_intervention_observer_v2.py'


def register(out):
    if out.exists():raise ValueError('Preserve prior intervention round')
    preparation=read(Q/'preparation.json')
    bindings=[bound(__file__),bound(PROGRAM),bound(Q/'preparation.json'),bound(ROOT/'stage5_review.json'),
              bound(ROOT/'atr_intervention_observer_tests_20261010.log'),
              bound(REPO/'tests/backtests/strategies/test_kalcb_atr_intervention_observer.py'),
              bound(ROOT/'inputs/manifest.json')]
    jobs=[]
    for item in preparation['profiles']:
        if item['name'] not in ('expanding_atr5','decay60_atr5'):continue
        bindings += [item['manifest'],item['plan'],item['policy']]
        for scenario in ('baseline','combined'):
            reference=Path(PROFILES[item['name']]['reference'])/item['period']/scenario
            for name in ('result.json','daily.json','equity.json','fills.json','trades.json'):
                bindings.append(bound(reference/name))
            jobs.append({'name':item['name'],'period':item['period'],'scenario':scenario,
                         'plan':item['plan']['path'],'policies':str(Path(item['policy']['path']).parent),
                         'reference':str(reference)})
    if len(jobs)!=8:raise ValueError('Incomplete support population')
    prior=DEST.with_name('v1')
    bindings += [bound(prior/p) for p in ('protocol.json','status.json','background.err.log','cases/expanding_atr5/is/baseline.log')]
    bindings += [bound(REPO/'scripts/kalcb_atr_intervention_observer.py'),bound(REPO/'scripts/kalcb_atr_intervention_round.py')]
    verify_bindings(bindings);out.mkdir(parents=True)
    write(out/'protocol.json',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),'jobs':jobs,'bindings':bindings,
        'mechanism':'Restore only diagnostic_original_quantity in the exact native sizing call; evaluate additional shares under same pre-fill cash, previous-bar capacity, native order priority and actual printed-volume feasibility. Return original decisions and fills unchanged.',
        'scope':'Eight observational account replays, not eight new strategy variants. No hypothetical fills/PnL; no extra training or parameter selection. Official source qualification remains separate.',
        'bridge':'Every daily balance/mark and physical fill/trade must reproduce the native reference. Forecast lineage metadata labels can differ across already equivalent exports and are not economic fills.',
        'support':'Prefer10 direct feasible issuer-days across8 sessions, both periods and stresses. Nominal quantity changes, unfilled orders and compounded descendants do not count as direct feasible interventions. Counts are conditional support, not independent observations.',
        'tests':'Seven tests passed, including unchanged real-broker fills/cash, insufficient capacity, buying power, realized volume and no future-volume influence on decision quantities.',
        'promotion':'Disabled. No canonical/live changes.'})
    print({'registered':str(out),'observational_replays':8},flush=True)


def run(out):
    import msvcrt,psutil
    p=read(out/'protocol.json');verify_bindings(p['bindings'])
    lock=(out/'run.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'child':None}
    def status(stage,**kwargs):
        state.update(stage=stage,updated_at_utc=datetime.now(timezone.utc).isoformat(),**kwargs)
        write(out/'status.tmp',state);os.replace(out/'status.tmp',out/'status.json')
    results=[]
    try:
        for job in p['jobs']:
            while psutil.virtual_memory().available<3*1024**3:
                status('waiting_for_memory');time.sleep(30)
            if (out/'hold.json').exists():raise ValueError('Explicit hold')
            verify_bindings(p['bindings'])
            target=out/'cases'/job['name']/job['period']/job['scenario'];target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():raise ValueError('Preserve previous observational case')
            command=[sys.executable,'-B','-u',str(PROGRAM),'--code',str(ROOT/'inputs/files'),
                     '--plan',job['plan'],'--policies',job['policies'],'--scenario',job['scenario'],
                     '--reference',job['reference'],'--output',str(target)]
            with target.with_suffix('.log').open('x',encoding='utf-8') as stream:
                process=subprocess.Popen(command,cwd=ROOT/'inputs/files',env=env(ROOT/'inputs/files'),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                status('observing',job=job,child={'pid':process.pid,'process_created':psutil.Process(process.pid).create_time(),'command':command})
                while process.poll() is None:time.sleep(15)
            if process.returncode:raise ValueError('Observer failure; preserve '+str(target))
            item=read(target/'intervention_support.json')
            if not item['complete'] or item['full_account_bridge']!='passed':raise ValueError('Incomplete intervention proof')
            results.append({**job,'summary':item['summary'],'evidence':bound(target/'intervention_support.json')})
            write(out/'progress.json',results);print(results[-1]['summary'],flush=True)
            status('case_complete',child=None)
        write(out/'completion.json',{'complete':True,'results':results,'protocol':bound(out/'protocol.json'),
              'scope':'Conditional mechanism support only. Four-scenario source qualification and final research acceptance remain separate.'})
        status('ready_for_support_review',child=None)
    except Exception as error:status('attention_required',error=repr(error));raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['register','run']);parser.add_argument('--root',type=Path,default=DEST)
    args=parser.parse_args();{'register':register,'run':run}[args.stage](args.root)

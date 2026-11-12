"""Register only after V21 exits; run one read-only observer assessment."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify
from backtests.strategies.kalcb.reclaim_observer_assessment_v1 import completion_gate, assess, require

PROGRAMS=('scripts/kalcb_reclaim_observer_assessment_v1.py',
    'backtests/strategies/kalcb/reclaim_observer_assessment_v1.py',
    'tests/backtests/strategies/test_kalcb_reclaim_observer_assessment_v1.py',
    'backtests/strategies/kalcb/retention_research_io_v1.py',
    'backtests/strategies/kalcb/late_artifact_validation_v2.py',
    'backtests/strategies/kalcb/qualified_artifacts.py',
    'backtests/strategies/kalcb/population_research_io_v1.py',
    'backtests/strategies/kalcb/exit_qualification_assessment_v1.py',
    'backtests/strategies/kalcb/fixed_allocation_funding_v1.py',
    'backtests/auto/shared/cache_keys.py')


def register(root,output,tests):
    rows=completion_gate(root)
    receipt=read(tests);require(receipt['exit_code']==0,'Focused tests did not pass')
    verify(receipt['programs']+[receipt['log']])
    protocol=read(root/'protocol.json');verify(protocol['programs']+protocol['bindings'])
    bindings=[bound(root/f) for f in ('protocol.json','study.json','opportunities.json','progress.json',
        'supervisor_start.json','stage_completion.json')]+[bound(tests)]
    for r in rows:
        folder=root/'cases'/r['name']/r['period']/r['scenario']
        bindings += [bound(folder/f) for f in ('completion.json','observer_evidence.json','reclaim_source_proof.json','identity_bridge.json')]
    output.mkdir(parents=True,exist_ok=False)
    write(output/'protocol.json',{'version':'kalcb-parent-observer-assessment-v1',
        'registered_at_utc':datetime.now(timezone.utc).isoformat(),'root':str(root),'native_supervisor':read(root/'supervisor_start.json'),
        'programs':[bound(REPO/p) for p in PROGRAMS],'bindings':bindings+protocol['bindings'],
        'tests':receipt,'cases':8,'new_variants':0,'study_complete':False,
        'units':'Actual unique position lifetimes; partial fills/exits collapsed with cash/fee/quantity reconciliation. Per-filled-notional bps are descriptive, never rescaled portfolio returns.',
        'exposure_bins':'All six declared exposure variables use IS-only route-specific 25/50/75 percentiles, ties deduplicated, applied unchanged OOS. No threshold optimization.',
        'source':'Original native audits and independent accounts, exact parent identities, additive90minute proofs, before/after origin hashes; every MTM mark additionally reconciled from raw closes and physical flows.',
        'scope':'Read-only observer attribution. No economic replay, alpha claim, promotion or whole-study completion.'})


def run(root,output):
    import psutil,msvcrt
    completion_gate(root)
    lock=(output/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    require(not (output/'supervisor_start.json').exists(),'Preserve previous assessment attempt')
    temp=output/'temp';temp.mkdir(exist_ok=True);os.environ['TEMP']=os.environ['TMP']=str(temp)
    identity={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':psutil.Process().cmdline()}
    write(output/'supervisor_start.json',identity)
    try:
        write(output/'status.json',{**identity,'stage':'read_only_assessment','child':None})
        assess(root,output)
        write(output/'status.json',{**identity,'stage':'complete','child':None,'study_complete':False})
    except Exception as error:
        write(output/'status.json',{**identity,'stage':'attention_required','error':repr(error)});raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('register','run'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tests',type=Path)
    a=p.parse_args();root=a.root.resolve();output=a.output.resolve()
    register(root,output,a.tests) if a.stage=='register' else run(root,output)

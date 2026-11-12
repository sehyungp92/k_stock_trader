"""Bounded monthly funding-state population, with matched native hold arms.

No branch return is a selection input. Empty cells stay empty; a failed pair
stops the queue. This estimates conditional complete-account responses, not
an automatically qualified trading rule or a parameter sweep.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import psutil

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.olr_economic_study import read,write,sha,source_hash,sig
from scripts.olr_structural_study import private_write
from scripts.olr_structural_controller import status,progress,verify_identity
from scripts.olr_funded_population_patches import patch_runner,patch_allocation
from scripts.olr_economic_assessment import contrast
from backtests.strategies.olr.economic_evidence import completed_case
from backtests.strategies.olr.funded_action_probe import VERSION as PROBE_VERSION
from backtests.strategies.olr.funded_population import VERSION,decision_evidence

PARENT=Path('E:/CodexResearch/k_stock_trader/olr_structural_execution_20261013/generation_v5')
UPDATED=('backtests/strategies/olr/funded_population.py','scripts/olr_funded_population_patches.py',
    'scripts/olr_funded_population_study.py','tests/strategy_olr/test_funded_population.py')


def prepare(root):
    ready=read(PARENT/'native_engines/train/ready.json');source=Path(ready['files'])
    if source_hash(source)!=read(PARENT/'preflight.json')['source']:raise ValueError('Changed parent source')
    parent=completed_case(PARENT/'cases/train/hold_baseline')
    assessment=read(PARENT/'funded_response_assessment.json')
    if not parent['valid'] or assessment['source']!=parent['source']:raise ValueError('Parent not authenticated')
    destination=root/'native_engines/train/files';destination.mkdir(parents=True,exist_ok=False)
    manifest=copy.deepcopy(ready['manifest']);manifest.pop('manifest_hash',None)
    for name in manifest['files']:
        p=destination/name;p.parent.mkdir(parents=True,exist_ok=True);os.link(source/name,p)
    for name in UPDATED:
        private_write(destination/name,(REPO/name).read_text(encoding='utf8'));manifest['files'][name]=sha(destination/name)
    for name,patcher in [('scripts/portfolio_synergy_study.py',patch_runner),('strategy_olr/core/allocation.py',patch_allocation)]:
        text=patcher((source/name).read_text(encoding='utf8'));compile(text,name,'exec')
        private_write(destination/name,text);manifest['files'][name]=sha(destination/name)
    generation=source_hash(destination);manifest['source_generation_hash']=generation
    study=copy.deepcopy(ready['study']);study.update(code=str(destination),source_generation_hash=generation)
    write(root/'native_engines/train/ready.json',dict(files=str(destination),study=study,
        manifest=manifest,interval_qualification=ready['interval_qualification']))
    write(root/'native_engines/train/freeze_receipt.json',dict(parent=str(PARENT),source=generation,
        parent_source=parent['source'],parent_completion=parent['receipt_sha256'],
        parent_assessment_sha256=sha(PARENT/'funded_response_assessment.json'),
        private_files={n:h for n,h in manifest['files'].items() if ready['manifest']['files'].get(n)!=h}))
    write(root/'protocol.json',dict(version=VERSION,inherited_protocol=str(PARENT/'protocol.json'),
        inherited_protocol_sha256=sha(PARENT/'protocol.json'),inherited_gates=read(PARENT/'protocol.json'),
        selector='First supported existing native mfe_fade boundary per calendar month and observed capital state. Sort simultaneous symbols lexically. Half sale marked notional >=1% initial NAV (inherited meaningful threshold).',
        capital_state='Constrained iff actual cash minus remaining acknowledged BUY reservations is below the current marked notional of the candidate holding. This coverage label alone is not proof of an entry funding shortfall.',
        catalog='Unchanged native control, exact V5 equity/daily/trades/full NAV required; first records per cell never overwritten. All observed eligible cells used, including unfavorable responses; no outcome-based subset.',
        actions=['hold','half','exit'],horizon='Full original IS native account; one intervention per independent arm; later scopes, observations, funding and entries endogenous.',
        procedure='Catalog first. Each cell gets a fresh source-audited hold with exact catalog selection and exact V5 economic parity, then two matched action arms. Pair failure stops the queue. Concurrency two, no adaptive extra cells or actions.',
        entry_evidence='Persist actual native hard-eligible allocation proposals, desired quantity/unit cost, approved quantity, binding limit, ledger cash, pending reserves and sizing envelope separately. Zero-capacity pre-proposal and raw signal episodes are not labeled fundable opportunities.',
        assessment='Full-event/common-grid DD and high-water; actual current quantity/service/cancellations; direct versus later quantity, observation and funded-entry effects. Group results by decision date/month and capital state, disclose unsupported/empty states; no independent-sample claim for overlapping full-horizon arms.',
        limitations='All response labels are retrospective and overlapping. They cannot choose an action at their own historical decision or qualify a general rule. No forecast model, profitable-entry assumption, new entry policy or risk cap is installed.',
        next='Use supported state/action and entry funding evidence to design a separately frozen causal current-book/meaningful-entry mechanism; require chronological qualification and all original gates. Do not stop at panel completion.',
        numerical_gates_relaxed=False,oos_authorized=False,economic_objective_achieved=False))
    print(json.dumps(dict(root=str(root),source=generation)),flush=True)


def launch(root,tag,action,cell,reference,catalog_reference):
    ready=read(root/'native_engines/train/ready.json');files=Path(ready['files'])
    s=copy.deepcopy(ready['study']);m=copy.deepcopy(ready['manifest']);out=root/'cases/train'/tag
    if out.exists():raise FileExistsError('Preserve existing output '+str(out))
    prefix='research/funded_population_v1/'+tag+'/'
    values=[('structural_settings','structural.json',dict(mode='control',interval_service=True)),
        ('funded_probe_settings','probe.json',dict(version=PROBE_VERSION,action=action,control_reference=reference)),
        ('funded_population_settings','population.json',dict(version=VERSION,cell=cell,catalog_reference=catalog_reference))]
    for key,name,value in values:
        s[key]=prefix+name;write(files/s[key],value);m['files'][s[key]]=sha(files/s[key])
    s.pop('economic_policy_settings',None);s.pop('drawdown_reserve_contract',None)
    m['evidence']['qualification_path']=ready['interval_qualification']
    s['manifest']=str(files.parent/(tag+'.manifest.json'));s['configuration']['input_manifest_path']=s['manifest']
    s['spec_path']=prefix+'spec.json'
    write(files/s['spec_path'],{k:v for k,v in s.items() if k not in ('code','manifest','spec_path','source_generation_hash')})
    m['files'][s['spec_path']]=sha(files/s['spec_path']);write(s['manifest'],{**m,'manifest_hash':sig(m)})
    sp=root/'runtime/train'/tag/'study.json';write(sp,s)
    command=[sys.executable,'-B','-u',str(files/'scripts/portfolio_synergy_study.py'),'replay','--study',str(sp),
        '--recipe','candidate','--scenario','baseline','--standalone','OLR','--output',str(out),'--full-audit']
    tmp=root/'tmp'/tag;tmp.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(TEMP=str(tmp),TMP=str(tmp),PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    log=root/'logs/train'/(tag+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x',encoding='utf8') as f:
        child=subprocess.Popen(command,cwd=files,env=env,stdout=f,stderr=subprocess.STDOUT,creationflags=0x08000000)
    write(sp.parent/'process.json',dict(pid=child.pid,created_at=psutil.Process(child.pid).create_time(),command=command,
        source=s['source_generation_hash'],settings=dict(action=action,cell=cell),log=str(log),case=str(out),started_at=datetime.now(timezone.utc).isoformat()))
    return child


def case(root,tag,action='hold',cell=None,reference=None,catalog_reference=None):
    output=root/'cases/train'/tag;identity=root/'runtime/train'/tag/'process.json'
    if not (output/'completion.json').exists():
        if identity.exists():
            import time
            process=read(identity)
            if not verify_identity(process):raise RuntimeError('Dead incomplete case; preserve and repair separately')
            while verify_identity(process):time.sleep(10)
        else:
            child=launch(root,tag,action,cell,reference,catalog_reference)
            if child.wait():raise RuntimeError('Native population case failed: '+tag)
    result=completed_case(output);cp=read(output/'funded_checkpoint_receipt.json')
    population=read(output/'funded_population_receipt.json')
    if not result['valid'] or not cp['no_action_continuation_equal'] or sha(output/'native_account_checkpoint.pkl')!=cp['sha256'] or not population['audit']['pass_']:
        raise ValueError('Unqualified population case: '+tag)
    return result


def parity(root,tag,parent,*,receipt_name=None):
    path=root/'cases/train'/tag
    checks={n:read(parent/n)==read(path/n) for n in ('equity.json','daily.json','trades.json','minute_nav_evidence.json')}
    receipt=dict(checks=checks,pass_=all(checks.values()),parent_completion=completed_case(parent)['receipt_sha256'],
        completion=completed_case(path)['receipt_sha256'])
    receipt_path=root/'parity'/((receipt_name or tag)+'.json')
    write(receipt_path,receipt)
    if not receipt['pass_']:raise ValueError('Population hold changed native economic path')
    return receipt_path


def run(root):
    pre=read(root/'preflight.json')
    if (not pre['tests_passed'] or sha(Path(pre['tests']))!=pre['tests_sha256'] or
            sha(root/'protocol.json')!=pre['protocol_sha256'] or source_hash(root/'native_engines/train/files')!=pre['source']):
        raise ValueError('Frozen preflight changed')
    status(root,'population_catalog_parity_running',economic_objective_achieved=False)
    catalog=case(root,'catalog_hold');parent=PARENT/'cases/train/hold_baseline';parity(root,'catalog_hold',parent)
    catalog_path=root/'cases/train/catalog_hold';cells=read(catalog_path/'funded_population_receipt.json')['cells']
    if not cells:raise ValueError('No supported meaningful population cells')
    write(root/'registered_cells.json',dict(catalog_completion=catalog['receipt_sha256'],cells=cells,
        selection_used_response_outcomes=False,total_cases=1+3*len(cells)))
    for index,(cell,selection) in enumerate(sorted(cells.items())):
        status(root,'population_cell_hold_running',cell=cell,completed_cells=index,total_cells=len(cells),economic_objective_achieved=False)
        hold=case(root,cell+'_hold',cell=cell,catalog_reference=selection);hp=Path(hold['path'])
        parity(root,cell+'_hold',parent)
        reference=read(hp/'funded_probe_receipt.json')
        if decision_evidence(reference['decision'])!=selection:raise ValueError('Changed cell selector')
        status(root,'population_cell_actions_running',cell=cell,completed_cells=index,total_cells=len(cells),economic_objective_achieved=False)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={a:pool.submit(case,root,cell+'_'+a,a,cell,reference,selection) for a in ('half','exit')}
            arms={a:f.result() for a,f in futures.items()}
        comparison={}
        for action,c in arms.items():
            path=Path(c['path']);probe=read(path/'funded_probe_receipt.json')
            if any(probe[k]!=reference[k] for k in ('decision','observed_account','start_signature','no_action_session_signature')):
                raise ValueError('Unmatched population action shadow')
            comparison[action]=dict(completion=c['receipt_sha256'],return_delta_pp=c['return_pct']-hold['return_pct'],
                full_dd_delta_pp=c['full_dd']-hold['full_dd'],common_dd_delta_pp=c['common_dd']-hold['common_dd'],
                acquisition_delta=c['trades']-hold['trades'],entry_day_delta=c['days']-hold['days'],
                funded_accounting=contrast(read(hp/'trades.json'),read(path/'trades.json')))
        write(root/'comparisons'/(cell+'.json'),dict(cell=cell,selection=selection,control_completion=hold['receipt_sha256'],
            comparisons=comparison,general_policy_qualified=False,economic_objective_achieved=False))
    status(root,'awaiting_population_response_review',completed_cells=len(cells),total_cells=len(cells),
        economic_objective_achieved=False,automatic_oos_authority=False,
        next='Audit complete DD, action and downstream meaningful funding mechanisms across all cells. Continue only supported new-generation structural decisions; this panel is not programme completion.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','run','progress'));p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if a.phase=='prepare':prepare(a.root)
    elif a.phase=='progress':progress(a.root)
    else:
        try:run(a.root)
        except BaseException as exc:
            status(a.root,'needs_engineering_review',error=repr(exc),economic_objective_achieved=False)
            raise

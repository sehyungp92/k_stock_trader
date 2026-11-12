"""V9 gives each parity comparison one exclusive receipt; economics stay fixed."""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.olr_economic_study import read,write,sha,source_hash
from scripts.olr_structural_study import private_write
from scripts.olr_structural_controller import status,progress,verify_identity
from scripts import olr_funded_population_study as population
from backtests.strategies.olr.economic_evidence import completed_case

PARENT=Path('E:/CodexResearch/k_stock_trader/olr_structural_execution_20261013/generation_v6')
FREEZE_PARENT=PARENT.parent/'generation_v8'
UPDATED=('scripts/olr_funded_population_study.py','scripts/olr_funded_population_repair_study.py',
    'tests/strategy_olr/test_population_controller_handoff.py')


def prepare(root):
    ready=read(FREEZE_PARENT/'native_engines/train/ready.json');source=Path(ready['files'])
    if source_hash(source)!=read(FREEZE_PARENT/'preflight.json')['source']:raise ValueError('Changed V8 source')
    if any(verify_identity(read(p)) for p in [FREEZE_PARENT/'controller_process.json',*sorted((FREEZE_PARENT/'runtime').glob('*/*/process.json'))]):
        raise ValueError('Do not duplicate active V8 processes')
    parent=completed_case(FREEZE_PARENT/'cases/train/catalog_hold');parity=read(FREEZE_PARENT/'parity/repair_catalog_vs_v6.json')
    if not parent['valid'] or not parity['pass_'] or parent['receipt_sha256']!=parity['completion']:
        raise ValueError('V8 catalog not authenticated')
    failure=FREEZE_PARENT/'catalog_completion_and_controller_incident.json';incident=read(failure)
    if not incident['catalog_valid'] or not incident['exclusive_write_failure_reproduced']:
        raise ValueError('Missing precise V8 regression evidence')
    destination=root/'native_engines/train/files';destination.mkdir(parents=True,exist_ok=False)
    manifest=copy.deepcopy(ready['manifest']);manifest.pop('manifest_hash',None)
    for name in manifest['files']:
        p=destination/name;p.parent.mkdir(parents=True,exist_ok=True);os.link(source/name,p)
    for name in UPDATED:
        private_write(destination/name,(REPO/name).read_text(encoding='utf8'));manifest['files'][name]=sha(destination/name)
    generation=source_hash(destination);manifest['source_generation_hash']=generation
    study=copy.deepcopy(ready['study']);study.update(code=str(destination),source_generation_hash=generation)
    write(root/'native_engines/train/ready.json',dict(files=str(destination),study=study,
        manifest=manifest,interval_qualification=ready['interval_qualification']))
    write(root/'native_engines/train/freeze_receipt.json',dict(parent=str(FREEZE_PARENT),source=generation,
        parent_source=parent['source'],parent_catalog_completion=parent['receipt_sha256'],
        failure_evidence_sha256=sha(failure),parent_protocol_sha256=sha(FREEZE_PARENT/'protocol.json'),
        registered_cells_path=str(PARENT/'registered_cells.json'),
        parent_registered_cells_sha256=sha(PARENT/'registered_cells.json'),
        private_files={n:h for n,h in manifest['files'].items() if ready['manifest']['files'].get(n)!=h}))
    protocol=copy.deepcopy(read(FREEZE_PARENT/'protocol.json'))
    protocol.update(engineering_generation='v9-exclusive-parity-receipts',
        parent_protocol=str(FREEZE_PARENT/'protocol.json'),parent_protocol_sha256=sha(FREEZE_PARENT/'protocol.json'),
        repair='Write repair-versus-V6 parity directly to its unique name; reserve catalog_hold.json for the original V5 comparison. Keep exclusive creation. No observations, decisions, quantities, gates, cell selection or action horizons change.',
        repair_qualification='Reproduce V8 failure with real exclusive writes, receipt validation and complete controller-to-controller handoff. Require the full synthetic two-cell queue and fail-closed catalog/hold/action/receipt cases, all frozen tests, then exact V6/V5 catalog parity and exact registered cell equality before fresh native population continuation.')
    write(root/'protocol.json',protocol)
    print(json.dumps(dict(root=str(root),source=generation)),flush=True)


def run(root):
    pre=read(root/'preflight.json')
    if (not pre['tests_passed'] or sha(Path(pre['tests']))!=pre['tests_sha256'] or
            sha(root/'protocol.json')!=pre['protocol_sha256'] or source_hash(root/'native_engines/train/files')!=pre['source'] or
            not pre.get('saved_failed_checkpoint_repaired')):
        raise ValueError('Repair preflight incomplete or changed')
    status(root,'repaired_catalog_parity_running',economic_objective_achieved=False)
    catalog=population.case(root,'catalog_hold')
    # Each comparison creates its own receipt exactly once. Copying a default
    # receipt leaves that exclusive path occupied for the inherited controller.
    repair_parity=population.parity(root,'catalog_hold',PARENT/'cases/train/catalog_hold',
        receipt_name='repair_catalog_vs_v6')
    cells=read(Path(catalog['path'])/'funded_population_receipt.json')['cells']
    inherited=read(PARENT/'registered_cells.json')['cells']
    if cells!=inherited:raise ValueError('Engineering repair changed the registered population')
    write(root/'repair_catalog_qualification.json',dict(completion=catalog['receipt_sha256'],
        parent_catalog_completion=completed_case(PARENT/'cases/train/catalog_hold')['receipt_sha256'],
        four_file_parity_path=str(repair_parity),four_file_parity_sha256=sha(repair_parity),registered_cells_equal=True,
        registered_cells_sha256=sha(PARENT/'registered_cells.json'),economic_objective_achieved=False))
    # V6's original controller still applies its own V5 baseline parity and
    # unchanged hold/action checks. No completed or failed V6 case is reused.
    population.run(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','run','progress'));p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if a.phase=='prepare':prepare(a.root)
    elif a.phase=='progress':progress(a.root)
    else:
        try:run(a.root)
        except BaseException as exc:
            status(a.root,'needs_engineering_review',error=repr(exc),economic_objective_achieved=False)
            raise

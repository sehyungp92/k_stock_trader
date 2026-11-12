"""Source-bound native parity stage for decision-level funded continuation.

V3 economic failures remain rejected. This stage verifies an engine boundary;
it cannot authorize OOS, activate a policy or declare the economic goal achieved.
"""
import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.olr_economic_study import read, write, sha, source_hash
from scripts.olr_structural_study import private_write
from scripts.olr_structural_controller import case, progress, status
from backtests.strategies.olr.economic_evidence import completed_case

PARENT = Path('E:/CodexResearch/k_stock_trader/olr_structural_execution_20261013/generation_v3')
UPDATED = ('backtests/strategies/olr/decision_continuation.py',
           'backtests/strategies/olr/funded_checkpoint.py',
           'tests/strategy_olr/test_decision_continuation.py',
           'scripts/olr_decision_continuation_study.py')


def prepare(root):
    parent = read(PARENT/'native_engines/train/ready.json')
    expected = read(PARENT/'preflight.json')['source']
    if source_hash(Path(parent['files'])) != expected:
        raise ValueError('Parent source changed')
    if read(PARENT/'retained_gate_assessment.json')['risk_component_eligible_for_oos_review']:
        raise ValueError('Unexpected V3 verdict; review programme before changing stage')
    source = Path(parent['files']); destination = root/'native_engines/train/files'
    destination.mkdir(parents=True, exist_ok=False)
    manifest = copy.deepcopy(parent['manifest']); manifest.pop('manifest_hash',None)
    for name in manifest['files']:
        p = destination/name; p.parent.mkdir(parents=True,exist_ok=True)
        os.link(source/name,p)
    for name in UPDATED:
        private_write(destination/name,(REPO/name).read_text(encoding='utf8'))
        manifest['files'][name] = sha(destination/name)
    name = 'scripts/portfolio_synergy_study.py'
    text = (source/name).read_text(encoding='utf8')
    old = 'from backtests.strategies.olr.structural_driver import StructuralDriver'
    if text.count(old) != 1:
        raise ValueError('Unexpected parent driver dispatch')
    text = text.replace(old,'from backtests.strategies.olr.decision_continuation import DecisionContinuationDriver as StructuralDriver')
    compile(text,name,'exec');private_write(destination/name,text)
    manifest['files'][name] = sha(destination/name)
    generation = source_hash(destination)
    manifest['source_generation_hash'] = generation
    study = copy.deepcopy(parent['study']);study.update(code=str(destination),source_generation_hash=generation)
    write(root/'native_engines/train/ready.json',dict(files=str(destination),study=study,
        manifest=manifest,interval_qualification=parent['interval_qualification']))
    delta = {n:h for n,h in manifest['files'].items() if parent['manifest']['files'].get(n)!=h}
    write(root/'native_engines/train/freeze_receipt.json',dict(
        at=datetime.now(timezone.utc).isoformat(),parent=str(PARENT),parent_source=expected,
        source=generation,private_files=delta,
        parent_ready_sha256=sha(PARENT/'native_engines/train/ready.json'),
        active_parent_inputs_unchanged=True))
    write(root/'protocol.json',dict(version='olr-decision-continuation-v4',
        inherited_protocol=str(PARENT/'protocol.json'),inherited_protocol_sha256=sha(PARENT/'protocol.json'),
        inherited_gates=read(PARENT/'protocol.json'),
        prior_verdict=str(PARENT/'retained_gate_assessment.json'),
        prior_verdict_sha256=sha(PARENT/'retained_gate_assessment.json'),
        matched_control=str(PARENT/'cases/train/control_baseline'),
        stage='Exact native no-action parity before decision-level funded alternatives.',
        trading_rule_changed=False,numerical_gates_relaxed=False,oos_authorized=False,
        next='After exact parity, implement and authenticate bounded complete-account holding/entry action comparisons at causally observed decision boundaries. No rule or forecast is authorized by engineering parity.'))
    print(json.dumps(dict(root=str(root),source=generation,changed=list(delta))),flush=True)


def run(root):
    preflight=read(root/'preflight.json');protocol=read(root/'protocol.json')
    if not preflight['tests_passed'] or sha(Path(preflight['tests']))!=preflight['tests_sha256']:
        raise ValueError('Frozen regression evidence failed')
    if sha(root/'protocol.json')!=preflight['protocol_sha256']:
        raise ValueError('Protocol changed')
    if source_hash(root/'native_engines/train/files')!=preflight['source']:
        raise ValueError('Frozen continuation source changed')
    status(root,'native_continuation_parity_running',economic_objective_achieved=False)
    candidate=case(root,'train','control')
    a=Path(protocol['matched_control']);b=Path(candidate['path'])
    parent=completed_case(a)
    checks={name:read(a/name)==read(b/name) for name in ('equity.json','daily.json','trades.json','minute_nav_evidence.json')}
    checks['physical_and_source_audits']=bool(parent['valid'] and candidate['valid'])
    checks['no_action_checkpoint']=read(b/'funded_checkpoint_receipt.json')['no_action_continuation_equal']
    write(root/'native_parity.json',dict(at=datetime.now(timezone.utc).isoformat(),checks=checks,
        pass_=all(checks.values()),parent_completion=parent['receipt_sha256'],
        new_completion=candidate['receipt_sha256'],economic_objective_achieved=False))
    if not all(checks.values()):
        raise ValueError('Native continuation changed the retained economic path')
    status(root,'awaiting_funded_action_design',economic_objective_achieved=False,
        next='Continue authorized native action-value and meaningful-entry implementation; parity is not programme completion.',
        automatic_oos_authority=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','run','progress'))
    p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if a.phase=='prepare':prepare(a.root)
    elif a.phase=='progress':progress(a.root)
    else:
        try:run(a.root)
        except BaseException as error:
            status(a.root,'needs_engineering_review',error=repr(error),economic_objective_achieved=False)
            raise

"""Controller integration using real exclusive writes and receipt validation.

Only the expensive native subprocess is replaced by a bound synthetic export.
These fixtures test orchestration contracts, not physical trading performance.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import olr_funded_population_repair_study as repair
from scripts import olr_funded_population_study as population
from scripts.olr_economic_study import read, write, sha, source_hash


@pytest.fixture
def controller_fixture(tmp_path, monkeypatch):
    root=tmp_path/'run';parent=tmp_path/'repair_parent';original=tmp_path/'original'
    root.mkdir();parent.mkdir();original.mkdir()
    source=root/'native_engines/train/files';source.mkdir(parents=True)
    generation=source_hash(source)
    cells={'2025-09_cash_rich':{'symbol':'A','quantity':20},
           '2025-09_constrained':{'symbol':'B','quantity':30}}
    trades=[dict(symbol='A',entry_decision_time='2025-09-04T09:01:00+09:00',
                 qty=20,entry_price=100,net_pnl=10,exit_reason='native')]
    faults={};launches=[]

    def export(path,tag,selection=None):
        path.mkdir(parents=True)
        popcells=deepcopy(cells)
        if faults.get(tag)=='cells':popcells.pop(next(iter(popcells)))
        nav=dict(all_event_drawdown_pct=1,common_original_drawdown_pct=1,original_grid_sha256='grid')
        if faults.get(tag)=='nav':nav['all_event_drawdown_pct']=2
        write(path/'equity.json',[100,110]);write(path/'daily.json',[{'nav':110}])
        write(path/'trades.json',trades);write(path/'minute_nav_evidence.json',nav)
        write(path/'result.json',dict(source_generation_hash=generation,execution_validity_pass=True,
              terminal_pending=0,terminal_inventory={},return_pct=10,drawdown_pct=1,trades=1,active_days=1))
        write(path/'execution_audit.json',dict(audit_receipt=dict(source_generation_hash=generation),
              audit_pass=faults.get(tag)!='audit',execution_validity_pass=True,audit_status='passed'))
        checkpoint=path/'native_account_checkpoint.pkl';checkpoint.write_bytes(b'synthetic-orchestration-checkpoint')
        write(path/'funded_checkpoint_receipt.json',dict(sha256=sha(checkpoint),no_action_continuation_equal=True))
        write(path/'funded_population_receipt.json',dict(cells=popcells,audit=dict(pass_=True)))
        decision=deepcopy(selection or next(iter(cells.values())))
        signature='altered' if faults.get(tag)=='shadow' else 'same-start'
        write(path/'funded_probe_receipt.json',dict(decision=decision,observed_account=dict(cash=100),
              start_signature=signature,no_action_session_signature='same-suffix'))
        receipt=dict(complete=True,source_generation_hash=generation,
                     result_sha256=sha(path/'result.json'),audit_sha256=sha(path/'execution_audit.json'),
                     artifacts={p.name:sha(p) for p in path.iterdir()})
        write(path/'completion.json',receipt)

    export(parent/'cases/train/catalog_hold','parent')
    export(original/'cases/train/hold_baseline','original')
    write(parent/'registered_cells.json',dict(cells=cells))
    tests=tmp_path/'tests.xml';tests.write_text('test-contract',encoding='utf8')
    write(root/'protocol.json',dict(numerical_gates_relaxed=False))
    write(root/'preflight.json',dict(tests_passed=True,tests=str(tests),tests_sha256=sha(tests),
          protocol_sha256=sha(root/'protocol.json'),source=generation,saved_failed_checkpoint_repaired=True))

    def launch(target,tag,action,cell,reference,catalog_reference):
        assert target==root
        launches.append(tag)
        export(root/'cases/train'/tag,tag,catalog_reference)
        return SimpleNamespace(wait=lambda:0)

    monkeypatch.setattr(repair,'PARENT',parent)
    monkeypatch.setattr(population,'PARENT',original)
    monkeypatch.setattr(population,'launch',launch)
    return SimpleNamespace(root=root,parent=parent,original=original,cells=cells,faults=faults,launches=launches)


def test_real_exclusive_writes_complete_both_parities_and_all_cells(controller_fixture):
    f=controller_fixture
    repair.run(f.root)
    status=read(f.root/'controller_status.json')
    assert status['phase']=='awaiting_population_response_review'
    assert not status['economic_objective_achieved']
    assert f.launches.count('catalog_hold')==1
    assert set(f.launches)=={'catalog_hold',*(cell+'_'+action for cell in f.cells for action in ('hold','half','exit'))}
    bound=read(f.root/'repair_catalog_qualification.json')
    proof=Path(bound['four_file_parity_path'])
    assert proof!=f.root/'parity/catalog_hold.json'
    assert sha(proof)==bound['four_file_parity_sha256']
    assert read(proof)['parent_completion']==sha(f.parent/'cases/train/catalog_hold/completion.json')
    assert read(f.root/'parity/catalog_hold.json')['parent_completion']==sha(f.original/'cases/train/hold_baseline/completion.json')
    assert read(f.root/'registered_cells.json')['cells']==f.cells
    assert len(list((f.root/'comparisons').glob('*.json')))==len(f.cells)
    assert all(not read(p)['general_policy_qualified'] for p in (f.root/'comparisons').glob('*.json'))


@pytest.mark.parametrize('fault',['audit','nav','cells'])
def test_catalog_failure_never_dispatches_cell_cases(controller_fixture,fault):
    f=controller_fixture;f.faults['catalog_hold']=fault
    with pytest.raises(ValueError):repair.run(f.root)
    assert f.launches==['catalog_hold']


def test_invalid_hold_never_dispatches_its_actions(controller_fixture):
    f=controller_fixture;first=sorted(f.cells)[0];f.faults[first+'_hold']='audit'
    with pytest.raises(ValueError,match='Unqualified population case'):repair.run(f.root)
    assert f.launches==['catalog_hold',first+'_hold']


@pytest.mark.parametrize('fault',['audit','shadow'])
def test_failed_action_pair_stops_before_next_cell(controller_fixture,fault):
    f=controller_fixture;first=sorted(f.cells)[0];f.faults[first+'_exit']=fault
    with pytest.raises(ValueError):repair.run(f.root)
    assert set(f.launches)=={'catalog_hold',first+'_hold',first+'_half',first+'_exit'}
    assert not list((f.root/'comparisons').glob('*.json'))


def test_existing_baseline_proof_is_preserved_and_queue_stops(controller_fixture):
    f=controller_fixture;p=f.root/'parity/catalog_hold.json'
    write(p,{'preexisting':'must remain byte-identical'});original=p.read_bytes()
    with pytest.raises(FileExistsError):repair.run(f.root)
    assert p.read_bytes()==original and f.launches==['catalog_hold']

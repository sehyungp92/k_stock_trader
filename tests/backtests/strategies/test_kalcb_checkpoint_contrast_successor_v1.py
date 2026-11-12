import ast
from pathlib import Path
import pytest
from scripts.kalcb_checkpoint_contrast_labels_v2 import smoke_gate


def fixture():
    return {'counts':{'execution_failures':0,'account_failures':0,'deadline_violations':1}}, {'passed':True,'pairs':222,'new_action_deadline_failures':0,'inherited_parent_deadline_failures':1}


def test_parent_risk_observation_is_retained_in_data_collection():
    d,r=fixture();smoke_gate(d,r)
    assert d['counts']['deadline_violations']==1


@pytest.mark.parametrize('poison',['execution','account','new_deadline','unaudited','unexplained'])
def test_new_action_or_unexplained_failures_cannot_pass(poison):
    d,r=fixture()
    if poison=='execution':d['counts']['execution_failures']=1
    elif poison=='account':d['counts']['account_failures']=1
    elif poison=='new_deadline':r['new_action_deadline_failures']=1
    elif poison=='unaudited':r['passed']=False
    else:d['counts']['deadline_violations']=2
    with pytest.raises(ValueError):smoke_gate(d,r)


def test_bootstrap_isolated_and_no_smoke_rerun():
    p=Path('scripts/kalcb_checkpoint_contrast_labels_v2.py');source=p.read_text();tree=ast.parse(source)
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[node.module] if isinstance(node,ast.ImportFrom) else [n.name for n in node.names]
            assert all(n.split('.')[0] not in ('backtests','scripts','strategy_kalcb','kis_core') for n in names)
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    calls=[n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='execute']
    assert len(calls)==1
    assert "'smoke'" not in ast.get_source_segment(source,calls[0])
    assert "'produce'" in ast.get_source_segment(source,calls[0])

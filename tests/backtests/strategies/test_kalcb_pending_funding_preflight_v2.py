import ast
from pathlib import Path
from scripts.kalcb_pending_funding_preflight_v2 import patch_core


def test_economic_patch_identical_to_registered_predecessor():
    from scripts.kalcb_pending_funding_preflight_v1 import patch_core as old
    source=Path('strategy_kalcb/core/logic.py').read_text(encoding='utf-8')
    assert patch_core(source)==old(source)


def test_actual_receipt_filename_expressions(tmp_path):
    tree=ast.parse(Path('scripts/kalcb_pending_funding_preflight_v2.py').read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
    paths=[n.args[0] for n in ast.walk(node) if isinstance(n,ast.Call) and
        isinstance(n.func,ast.Name) and n.func.id=='write' and isinstance(n.args[0],ast.BinOp) and
        '_imports_' in ast.unparse(n.args[0])]
    assert len(paths)==2
    for smoke in (False,True):
        for expr in paths:
            path=eval(compile(ast.Expression(expr),'<receipt>','eval'),{'root':tmp_path,'smoke':smoke})
            assert isinstance(path,Path) and path.parent==tmp_path


def test_no_application_import_before_runtime_isolation():
    tree=ast.parse(Path('scripts/kalcb_pending_funding_preflight_v2.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[node.module] if isinstance(node,ast.ImportFrom) else [x.name for x in node.names]
            assert all(n.split('.')[0] not in ('scripts','backtests','strategy_kalcb','kis_core') for n in names)

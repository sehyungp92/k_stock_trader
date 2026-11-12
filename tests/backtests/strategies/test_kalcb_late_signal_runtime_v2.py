import ast
import json
import os
from pathlib import Path
import subprocess
import sys


def test_bootstrap_top_level_has_only_standard_library_imports():
    path=Path('scripts/kalcb_late_signal_producer_v2.py')
    tree=ast.parse(path.read_text(encoding='utf-8'))
    imports=[]
    for node in tree.body:
        if isinstance(node,ast.Import):imports.extend(v.name.split('.')[0] for v in node.names)
        if isinstance(node,ast.ImportFrom):imports.append(node.module.split('.')[0])
    assert set(imports)<={'argparse','hashlib','importlib','json','pathlib','sys'}


def test_real_frozen_runtime_calendar_and_module_origins():
    root=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011/v8_signal')
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    code=Path(protocol['producer_runtime']['folder'])/'inputs/files'
    result=subprocess.run([sys.executable,'-B','scripts/kalcb_late_signal_producer_v2.py','--root',str(root),
        '--code',str(code),'--check-only'],capture_output=True,text=True,encoding='utf-8',
        env={**os.environ,'PYTHONUTF8':'1','OPENBLAS_NUM_THREADS':'1'},check=True)
    rows=[json.loads(x) for x in result.stdout.splitlines() if x.startswith('{')]
    assert rows[-1]['frozen_import_check'] and rows[-1]['modules']>50
    assert Path(rows[-1]['calendar']['path']).is_relative_to(code)

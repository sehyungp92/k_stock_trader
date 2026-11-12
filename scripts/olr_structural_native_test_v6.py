"""Test the actual freshly frozen modules, rejecting stray native imports."""
from pathlib import Path
import importlib.util,json,os,sys,hashlib

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    root=Path(sys.argv[1]);repo=Path(__file__).resolve().parents[1]
    ready=json.loads((root/'native_engines/train/ready.json').read_text(encoding='utf-8'));files=Path(ready['files'])
    sys.path.insert(0,str(repo));sys.path.insert(0,str(files));os.chdir(files)
    import strategy_olr,strategy_common,backtests,deployment,kis_core,scripts
    for name in ('olr_structural_source_v6','olr_structural_patches_v6'):
        qualified='scripts.'+name;path=repo/'scripts'/(name+'.py')
        spec=importlib.util.spec_from_file_location(qualified,path);module=importlib.util.module_from_spec(spec)
        sys.modules[qualified]=module;spec.loader.exec_module(module)
    import pytest
    tests=['test_liquidity_v6.py','test_liquidity_runtime_v6.py','test_minute_exit_v5.py','test_priority_loss_budget.py']
    rc=pytest.main([*[str(repo/'tests/strategy_olr'/name) for name in tests],'-q','--junitxml='+str(root/'tests.xml')])
    imports={}
    for name,module in list(sys.modules.items()):
        if not name.startswith(('strategy_olr.','strategy_common.','backtests.','deployment.','kis_core.','oms.','oms_client.')) or not getattr(module,'__file__',None):continue
        path=Path(module.__file__).resolve()
        if not path.is_relative_to(files.resolve()):raise ValueError('Non-native test import: '+name)
        relative=path.relative_to(files.resolve()).as_posix();digest=sha(path)
        if ready['manifest']['files'].get(relative)!=digest:raise ValueError('Unbound native test module: '+name)
        imports[relative]=digest
    (root/'test_native_imports.json').write_text(json.dumps(dict(imports=imports,test_files={str(repo/'tests/strategy_olr'/name):sha(repo/'tests/strategy_olr'/name) for name in tests},exit_code=rc),indent=2)+'\n',encoding='utf-8')
    raise SystemExit(rc)

if __name__=='__main__':main()

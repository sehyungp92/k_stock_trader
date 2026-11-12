"""Run the narrow clock contracts against authenticated frozen native imports."""
from pathlib import Path
import importlib.util
import json
import os
import sys


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def main():
    repo=Path(__file__).resolve().parents[1]
    contract=load('opportunity_test_contract',repo/'scripts/olr_opportunity_v7.py')
    isolation=load('opportunity_test_guard',repo/'scripts/olr_structural_native_test_isolated_v6.py')
    ready=contract.read(contract.PARENT/'native_engines/train/ready.json');files=Path(ready['files']).resolve()
    guard=isolation.NativeOnly(files,ready['manifest']['files']);sys.meta_path.insert(0,guard)
    sys.path.insert(0,str(files));os.chdir(files)
    load('opportunity_clock_contract',repo/'scripts/olr_opportunity_clock_v7.py')
    from backtests.core.input_manifest import InputManifest
    import pytest
    with InputManifest(ready['study']['manifest']).activate():
        rc=pytest.main([str(repo/'tests/strategy_olr/test_opportunity_clock_v7.py'),'-q',
                       '--junitxml='+str(contract.ROOT/'clock_tests.xml')])
    imports={}
    for name,module in list(sys.modules.items()):
        if name.split('.')[0] not in guard.roots or not getattr(module,'__file__',None):continue
        p=Path(module.__file__).resolve();assert p.is_relative_to(files)
        key=p.relative_to(files).as_posix();digest=contract.sha(p);assert ready['manifest']['files'][key]==digest
        imports[key]=digest
    contract.write(contract.ROOT/'clock_test_imports.json',dict(exit_code=rc,imports=imports,
        test_source_sha256=contract.sha(repo/'tests/strategy_olr/test_opportunity_clock_v7.py'),
        clock_source_sha256=contract.sha(repo/'scripts/olr_opportunity_clock_v7.py')))
    raise SystemExit(rc)


if __name__=='__main__':main()

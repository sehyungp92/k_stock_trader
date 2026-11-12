"""Pin every native package before pytest or its fixtures can alter sys.path."""
from pathlib import Path
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
import tempfile


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class NativeOnly(importlib.abc.MetaPathFinder):
    def __init__(self, files, manifest):
        self.files = files.resolve()
        self.manifest = manifest
        self.roots = {name.split('/')[0] for name in manifest
                      if name.count('/') == 1 and name.endswith('/__init__.py')} - {'tests'}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] not in self.roots:
            return None
        search = [str(self.files)] if path is None else path
        spec = importlib.machinery.PathFinder.find_spec(fullname, search)
        if spec is None or not spec.origin:
            raise ImportError('No frozen native module: ' + fullname)
        source = Path(spec.origin).resolve()
        if not source.is_relative_to(self.files):
            raise ImportError('Non-native import rejected before execution: ' + fullname)
        relative = source.relative_to(self.files).as_posix()
        if self.manifest.get(relative) != sha(source):
            raise ImportError('Unbound native module: ' + fullname)
        return spec


def guard_checks():
    with tempfile.TemporaryDirectory(prefix='structural-import-fixture-') as temporary:
        base = Path(temporary)
        native, foreign = base / 'native', base / 'foreign'
        for folder in (native, foreign):
            (folder / 'sample').mkdir(parents=True)
            (folder / 'sample/__init__.py').write_text('VALUE = 1\n')
        guard = NativeOnly(native, {'sample/__init__.py': sha(native / 'sample/__init__.py')})
        assert Path(guard.find_spec('sample').origin).resolve() == (native / 'sample/__init__.py').resolve()
        (native / 'sample/__init__.py').write_text('VALUE = 2\n')
        try:
            guard.find_spec('sample')
        except ImportError:
            pass
        else:
            raise AssertionError('Tampered source accepted')
        (foreign / 'sample/child.py').write_text('VALUE = 3\n')
        try:
            guard.find_spec('sample.child', [str(foreign / 'sample')])
        except ImportError:
            pass
        else:
            raise AssertionError('Foreign submodule accepted')
    return 3


def main():
    root = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parents[1]
    ready = json.loads((root / 'native_engines/train/ready.json').read_text())
    files = Path(ready['files']).resolve()
    guard_tests = guard_checks()
    guard = NativeOnly(files, ready['manifest']['files'])
    sys.meta_path.insert(0, guard)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(files))
    os.chdir(files)
    import scripts
    # These two pure source/patch helpers have no frozen predecessor. They are
    # explicit workspace dependencies in the original launch harness.
    helpers = {}
    for name in ('olr_structural_source_v6', 'olr_structural_patches_v6'):
        qualified = 'scripts.' + name
        source = repo / 'scripts' / (name + '.py')
        spec = importlib.util.spec_from_file_location(qualified, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        helpers[qualified] = sha(source)
    import pytest
    tests = ['test_liquidity_v6.py', 'test_liquidity_runtime_v6.py',
             'test_minute_exit_v5.py', 'test_priority_loss_budget.py']
    rc = pytest.main([*[str(repo / 'tests/strategy_olr' / name) for name in tests],
                      '-q', '--junitxml=' + str(root / 'isolated_tests.xml')])
    native, test_sources = {}, {}
    for name, module in list(sys.modules.items()):
        if not getattr(module, '__file__', None):
            continue
        source = Path(module.__file__).resolve()
        if name in helpers:
            assert sha(source) == helpers[name]
        elif name.split('.')[0] in guard.roots:
            if not source.is_relative_to(files):
                raise ValueError('Non-native test import: ' + name)
            relative = source.relative_to(files).as_posix()
            digest = sha(source)
            assert ready['manifest']['files'].get(relative) == digest, name
            native[relative] = digest
        elif source.is_relative_to(repo / 'tests'):
            test_sources[source.relative_to(repo).as_posix()] = sha(source)
    receipt = dict(imports=native, test_files=test_sources, explicit_helpers=helpers,
                   exit_code=rc, import_guard_before_execution=True, adversarial_import_checks=guard_tests)
    (root / 'isolated_test_native_imports.json').write_text(
        json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    raise SystemExit(rc)


if __name__ == '__main__':
    main()

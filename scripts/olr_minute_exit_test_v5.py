"""Test new minute modules with authenticated frozen native dependencies."""
import subprocess,sys
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts import olr_participation_study_v5 as t
from scripts.olr_minute_exit_patches_v5 import patch

TESTS=('tests/strategy_olr/test_minute_exit_v5.py','tests/strategy_olr/test_priority_loss_budget.py',
    'tests/strategy_olr/test_minute_materialize_v5.py','tests/strategy_olr/test_minute_source_v5.py',
    'tests/strategy_olr/test_minute_clock_source_v5.py')
MODULES={'backtests.engine.olr_minute_v5':'backtests/engine/olr_minute_v5.py',
    'backtests.engine.olr_minute_audit_v5':'backtests/engine/olr_minute_audit_v5.py',
    'backtests.engine.olr_minute_source_v5':'backtests/engine/olr_minute_source_v5.py'}
PATCHED=('strategy_olr/config.py','strategy_olr/research.py','backtests/engine/capacity_audit.py',
    'backtests/engine/integrity.py','backtests/engine/qualification_evidence.py','scripts/portfolio_synergy_study.py')


def main():
    files=Path.home()/'.codex/research/olr_repair_v4/train/inputs/files'
    root=t.OUT/'minute_engineering';root.mkdir(exist_ok=True)
    attempt=root/('attempt_'+str(len(list(root.glob('attempt_*')))))
    attempt.mkdir();log=attempt/'tests.log';xml=attempt/'tests.xml'
    modules=dict(MODULES)
    for name in PATCHED:
        target=attempt/'isolated_sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(patch(name,(files/name).read_text(encoding='utf-8')),encoding='utf-8')
        compile(target.read_text(encoding='utf-8'),str(target),'exec')
        if not name.startswith('scripts/'):
            modules[name[:-3].replace('/','.')]=str(target.relative_to(t.REPO))
    program='''import sys,importlib.util,json
from pathlib import Path
files=Path(sys.argv[1]);repo=Path(sys.argv[2]);modules=json.loads(sys.argv[3])
sys.path.insert(0,str(files))
import strategy_olr,strategy_common,backtests,deployment,kis_core
for name,path in modules.items():
    spec=importlib.util.spec_from_file_location(name,repo/path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
import pytest
rc=pytest.main(sys.argv[4:])
for name,module in list(sys.modules.items()):
    if name.startswith(('strategy_olr.','strategy_common.','backtests.','deployment.','kis_core.')) and getattr(module,'__file__',None):
        expected=repo/modules[name] if name in modules else None
        assert Path(module.__file__).resolve()==expected.resolve() if expected else Path(module.__file__).resolve().is_relative_to(files.resolve()),(name,module.__file__)
raise SystemExit(rc)
'''
    import json
    args=[sys.executable,'-B','-c',program,str(files),str(t.REPO),json.dumps(modules),
          *[str(t.REPO/p) for p in TESTS],'-q','--disable-warnings','--junitxml='+str(xml)]
    with log.open('w',encoding='utf-8') as f:r=subprocess.run(args,cwd=files,stdout=f,stderr=subprocess.STDOUT)
    counts=dict(tests=0,failures=0,errors=0,skipped=0)
    if xml.exists():
        tree=ET.parse(xml).getroot()
        for suite in tree.findall('testsuite') or [tree]:
            for k in counts:counts[k]+=int(suite.attrib.get(k,0))
    receipt=dict(utc=t.s.utc(),exit_code=r.returncode,**counts,
        passed=counts['tests']-counts['failures']-counts['errors']-counts['skipped'],
        source_hashes={p:t.n.sha(t.REPO/p) for p in (*TESTS,*modules.values(),
            'scripts/olr_minute_exit_patches_v5.py','scripts/olr_minute_exit_materialize_v5.py',
            'scripts/olr_minute_exit_clock_repair_v5.py',str(Path(__file__).relative_to(t.REPO)))},
        frozen_parent=str(files),log_sha256=t.n.sha(log),junit_sha256=t.n.sha(xml) if xml.exists() else None)
    t.n.write(attempt/'receipt.json',receipt);print(str(attempt));print(receipt)
    if r.returncode:print(log.read_text(encoding='utf-8')[-18000:])
    raise SystemExit(r.returncode)


if __name__=='__main__':main()

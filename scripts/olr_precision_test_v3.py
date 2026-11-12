"""Run focused tests against frozen strategy modules, binding exact sources."""
import os,subprocess,sys
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts import olr_precision_study_v3 as t
n=t.n
TESTS=('tests/strategy_olr/test_precision_v3.py','tests/strategy_olr/test_opening_risk_v2.py',
       'tests/strategy_olr/test_priority_loss_budget.py')

def main():
    files=t.ENGINES/'train/inputs/files';xml=t.OUT/'focused_tests.xml';log=t.OUT/'focused_tests.log'
    assert (files/'strategy_olr/core/precision_v3.py').exists()
    for name in t.PATCHED:
        assert n.sha(t.ENGINES/'train/inputs/files'/name)==n.sha(t.ENGINES/'oos/inputs/files'/name),name
    # Preload frozen package roots before pytest's repository conftest can add
    # the shared workspace. Every actual strategy dependency stays frozen.
    program='''import sys
from pathlib import Path
files=Path(sys.argv[1]);repo=Path(sys.argv[2])
sys.path.insert(0,str(files))
import strategy_olr, strategy_common, backtests, deployment
import strategy_olr.config
assert Path(strategy_olr.config.__file__).resolve().is_relative_to(files.resolve())
import pytest
rc=pytest.main(sys.argv[3:])
for name,module in list(sys.modules.items()):
    if name.startswith(('strategy_olr.','strategy_common.','backtests.','deployment.')) and getattr(module,'__file__',None):
        assert Path(module.__file__).resolve().is_relative_to(files.resolve()),(name,module.__file__)
raise SystemExit(rc)
'''
    args=[sys.executable,'-B','-c',program,str(files),str(t.REPO),*[str(t.REPO/p) for p in TESTS],'-q','--disable-warnings','--junitxml='+str(xml)]
    with log.open('w',encoding='utf-8') as f:result=subprocess.run(args,cwd=files,stdout=f,stderr=subprocess.STDOUT)
    tree=ET.parse(xml).getroot();suites=tree.findall('testsuite') or [tree]
    count=sum(int(x.attrib.get('tests',0)) for x in suites)
    failures=sum(int(x.attrib.get('failures',0))+int(x.attrib.get('errors',0)) for x in suites)
    skipped=sum(int(x.attrib.get('skipped',0)) for x in suites)
    receipt={'passed':count-failures-skipped,'failures':failures,'skipped':skipped,'exit_code':result.returncode,
             'test_hashes':{p:n.sha(t.REPO/p) for p in TESTS},
             'frozen_sources':{str(t.ENGINES/w/'inputs/files'/p):n.sha(t.ENGINES/w/'inputs/files'/p)
                for w in ('train','oos') for p in t.PATCHED},
             'log_sha256':n.sha(log),'junit_sha256':n.sha(xml)}
    n.write(t.OUT/'test_receipt.json',receipt);print(receipt)
    assert result.returncode==0 and failures==0 and skipped==0

if __name__=='__main__':main()

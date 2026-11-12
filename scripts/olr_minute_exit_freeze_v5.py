"""Freeze all qualification code and evidence only after engineering passes."""
from pathlib import Path
import sys,subprocess,xml.etree.ElementTree as ET
from scripts import olr_minute_exit_qualify_v5 as q
t=q.t;n,s=t.n,t.s


def main():
    q.check_preflight()
    if (t.ROOT/'cases').exists() or (t.OUT/'research_harness.json').exists():raise RuntimeError('Never rebind a frozen performance protocol')
    test='tests/strategy_olr/test_minute_qualification_v5.py'
    identity_test='tests/strategy_olr/test_minute_entry_identity_v5.py'
    log=t.OUT/'qualification_tests.log';xml=t.OUT/'qualification_tests.xml'
    with log.open('x',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,'-B','-m','pytest',test,identity_test,'--noconftest','-q','--junitxml='+str(xml)],
            cwd=t.REPO,stdout=stream,stderr=subprocess.STDOUT)
    root=ET.parse(xml).getroot();suites=root.findall('testsuite') or [root]
    counts={k:sum(int(x.attrib.get(k,0)) for x in suites) for k in ('tests','failures','errors','skipped')}
    assert result.returncode==counts['failures']==counts['errors']==counts['skipped']==0 and counts['tests']>=20
    trials=t.SOURCE_OUT/'minute_engineering'
    latest=max(trials.glob('attempt_*/receipt.json'),key=lambda p:int(p.parent.name.split('_')[-1]))
    receipt=n.read(latest)
    assert receipt['passed']>=65 and receipt['exit_code']==receipt['failures']==receipt['errors']==receipt['skipped']==0
    for name,digest in receipt['source_hashes'].items():assert n.sha(t.REPO/name)==digest,name
    assert n.sha(latest.parent/'tests.log')==receipt['log_sha256'] and n.sha(latest.parent/'tests.xml')==receipt['junit_sha256']
    sources=dict(n.read(t.ROOT/'engineering_controller_harness.json')['files'])
    for name in list(sources):assert n.sha(t.REPO/name)==sources[name],name
    # Include the transitive imported study, runner, history, metrics and native
    # construction helpers, not merely the new top-level controller.
    for module in list(sys.modules.values()):
        raw=getattr(module,'__file__',None)
        if not raw:continue
        path=Path(raw).resolve()
        if path.is_relative_to(t.REPO.resolve()) and path.suffix=='.py':
            sources[path.relative_to(t.REPO).as_posix()]=n.sha(path)
    for name in (*receipt['source_hashes'],test,identity_test,'scripts/olr_minute_exit_freeze_v5.py'):
        sources[name]=n.sha(t.REPO/name)
    files=[t.ROOT/name for name in ('engineering_complete.json','preflight_complete.json','preparation_complete.json',
        'registration.json','candidate_plan.json','mechanism_plan.json','protocol.json','source_preparation_complete.json',
        'engineering_controller_harness.json','native_source_preflight_train.json','native_source_preflight_oos.json',
        'entry_identity_train.json','entry_identity_oos.json','entry_identity_repair.json','temporary_storage_repair.json')]
    files += [latest,latest.parent/'tests.log',latest.parent/'tests.xml',log,xml,t.SOURCE_OUT/'results.json',
        t.previous.OUT/'selected_research_policy.json',t.previous.OUT/'results.json']
    for w in ('train','oos'):
        ready=n.read(t.ENGINES/w/'ready.json');files.extend([t.ENGINES/w/'ready.json',Path(ready['study']['manifest'])])
        assert n.source_hash(Path(ready['files']))==ready['study']['source_generation_hash']
        for name,digest in ready['manifest']['files'].items():assert n.sha(Path(ready['files'])/name)==digest,name
    n.write(t.OUT/'research_harness.json',dict(utc=s.utc(),files=sources,artifacts={str(p):n.sha(p) for p in files},
        tests_receipt=str(latest),tests_sha256=n.sha(latest),qualification_tests=counts,performance_started=False,
        scope='Frozen before control/E outcomes. Native manifests bind every source and engine file; no broad search or E+F union.'))
    print(dict(frozen=True,files=len(sources),tests=receipt['passed'],qualification_tests=counts),flush=True)


if __name__=='__main__':main()

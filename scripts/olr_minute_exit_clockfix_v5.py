"""Separate generation for the preserved terminal-observation failure.

No old input or performance artifact is rebound. Source-only evidence is reused
only after exact file and imported-parser authentication. All native case audits
still independently reparse consumed observations. No selection gate is changed.
"""
import argparse,copy,difflib,json,os,shutil,subprocess,sys,tempfile,traceback
from pathlib import Path
from scripts import olr_minute_exit_study_v5 as t
from scripts.olr_minute_exit_prepare_v5 import install
n,s=t.n,t.s
ORIGINAL=t.PARENT.parent/'minute_exit_v5'
ORIGINAL_OUT=t.REPO/'data/backtests/analysis/olr_minute_exit_20261011'
ROOT=ORIGINAL/'clockfix_generation_1'
OUT=ORIGINAL_OUT/'clockfix_generation_1'
CHANGED=('backtests/engine/olr_minute_v5.py','backtests/engine/olr_minute_audit_v5.py')
ARCHIVE=ORIGINAL/'qualification_failed_attempt_0'
PRESERVATION_SHA='cd087e06d8a2f06bc09e44edbe2235307ca807dd01b38e9f24c8388625b1b9aa'


def configure():
    t.ROOT,t.OUT,t.ENGINES=ROOT,OUT,ROOT/'native_engines'
    temp=ROOT/'runtime_tmp';temp.mkdir(parents=True,exist_ok=True)
    os.environ['TEMP']=os.environ['TMP']=str(temp);tempfile.tempdir=str(temp)
    t.activate()


def authenticate_old():
    assert n.sha(ARCHIVE/'preservation.json')==PRESERVATION_SHA
    preserved=n.read(ARCHIVE/'preservation.json')
    for old,item in preserved['files'].items():
        assert n.sha(item['archive'])==item['sha256'],old
    harness=n.read(ORIGINAL_OUT/'research_harness.json')
    assert n.sha(ORIGINAL_OUT/'research_harness.json')==preserved['files'][str(ORIGINAL_OUT/'research_harness.json')]['sha256']
    for path,digest in harness['artifacts'].items():assert n.sha(path)==digest,path
    assert not (ORIGINAL/'qualification_complete.json').exists()
    assert not list((ORIGINAL/'cases').rglob('assessment.json'))
    return harness


def resolve_origin_label(path,old_files,aliases):
    """Resolve only an already bound source-definition alias or an internal path."""
    old_files=Path(old_files).resolve()
    if str(path) in aliases:
        relative=Path(aliases[str(path)])
        if relative.is_absolute():raise ValueError('Absolute source alias is forbidden')
        return (old_files/relative).resolve().relative_to(old_files).as_posix()
    return Path(path).resolve().relative_to(old_files).as_posix()


def verify_source_reuse(old_files,new_files,old_manifest,new_manifest,proof,imports,allowed,aliases=None):
    """Exact file comparison, including every source and parser, not pass labels."""
    old_files,new_files=Path(old_files),Path(new_files)
    for name,digest in old_manifest['files'].items():
        assert n.sha(old_files/name)==digest,name
        if name not in allowed:
            assert new_manifest['files'].get(name)==digest,name
            assert n.sha(new_files/name)==digest,name
    for name,digest in new_manifest['files'].items():
        assert name in old_manifest['files'],name
        if name not in allowed:assert old_manifest['files'][name]==digest,name
    for name,digest in imports['native_imports'].items():
        assert name not in allowed,name
        assert old_manifest['files'][name]==new_manifest['files'][name]==digest,name
        assert n.sha(new_files/name)==digest,name
    for path,digest in proof['origins_read'].items():
        relative=resolve_origin_label(path,old_files,aliases or {})
        assert relative not in allowed,relative
        assert old_manifest['files'][relative]==new_manifest['files'][relative]==digest,relative
    assert proof['all_source_rows_recomputed'] and imports['passed']


def clone_window(w):
    parent=n.read(ORIGINAL/'native_engines'/w/'ready.json')
    old_files=Path(parent['files']);directory=t.ENGINES/w;files=directory/'inputs/files'
    assert not (directory/'ready.json').exists(),'Preserve a failed clone before restarting'
    manifest=copy.deepcopy(parent['manifest']);study=copy.deepcopy(parent['study'])
    identity=n.read(ORIGINAL/f'entry_identity_{w}.json')
    original_snapshot=Path(identity['old_snapshot_path']).relative_to(old_files).as_posix()
    assert parent['manifest']['files'][original_snapshot]==identity['old_snapshot_sha256']
    candidates=n.read(ROOT/'candidate_plan.json')['candidates']
    removed=set()
    for c in candidates:
        name='research/minute_exit_v5/preflight_snapshots/'+c['id']+'.json'
        removed.update((name,str(Path(name).with_suffix('.receipt.json')).replace('\\','/')))
    for name in removed:manifest['files'].pop(name)
    print(dict(window=w,phase='authenticate_and_copy_unchanged_inputs'),flush=True)
    for name,digest in manifest['files'].items():install(old_files/name,files/name,digest)
    diff=[]
    for name in CHANGED:
        before=(old_files/name).read_text(encoding='utf-8');after=(t.REPO/name).read_text(encoding='utf-8')
        compile(after,name,'exec');(files/name).write_text(after,encoding='utf-8',newline='\n')
        manifest['files'][name]=n.sha(files/name)
        diff.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+w+'/'+name,tofile='b/'+w+'/'+name))
    source=n.source_hash(files);manifest['source_generation_hash']=source
    study.update(code=str(files),manifest=str(directory/'inputs/manifest.json'),source_generation_hash=source,
        olr_snapshots=original_snapshot,canonical_olr_snapshot_hash=n.sig(n.read(files/original_snapshot)))
    study['configuration']['input_manifest_path']=study['manifest']
    n.write(files/study['spec_path'],{k:v for k,v in study.items() if k not in ('code','manifest','source_generation_hash','spec_path')})
    manifest['files'][study['spec_path']]=n.sha(files/study['spec_path'])
    n.write(study['manifest'],{**manifest,'manifest_hash':n.sig(manifest)})
    ready={**parent,'files':str(files),'study':study,'manifest':manifest,'parent_ready_sha256':n.sha(ORIGINAL/'native_engines'/w/'ready.json')}
    n.write(directory/'ready.json',ready)
    return certify_window(w,parent,ready,diff,removed)


def certify_window(w,parent,ready,diff,removed):
    old_files=Path(parent['files']);files=Path(ready['files']);manifest=ready['manifest'];study=ready['study'];source=study['source_generation_hash']
    assert n.source_hash(files)==source
    proof=n.read(ORIGINAL/f'native_source_preflight_{w}.json');imports=n.read(ORIGINAL/f'native_source_imports_{w}.json')
    assert proof['source_generation_hash']==parent['study']['source_generation_hash']
    worker=t.REPO/'scripts/olr_minute_exit_native_worker_v5.py'
    assert proof['worker_sha256']==imports['worker_sha256']==n.sha(worker)
    allowed=set(CHANGED)|removed|{study['spec_path']}
    definition=study['minute_source_definition']
    assert parent['manifest']['files'][definition]==manifest['files'][definition]==n.sha(files/definition)
    aliases=n.read(files/definition)['aliases']
    verify_source_reuse(old_files,files,parent['manifest'],manifest,proof,imports,allowed,aliases)
    # Preserve the origin proof byte-for-byte. This new envelope explicitly
    # authenticates reuse; it does not claim another origin reconstruction.
    original_proof=ROOT/'reused_evidence'/w/'original_source_preflight.json'
    original_imports=original_proof.with_name('original_source_imports.json')
    install(ORIGINAL/f'native_source_preflight_{w}.json',original_proof,n.sha(ORIGINAL/f'native_source_preflight_{w}.json'))
    install(ORIGINAL/f'native_source_imports_{w}.json',original_imports,n.sha(ORIGINAL/f'native_source_imports_{w}.json'))
    reused={**proof,'source_generation_hash':source,'manifest_sha256':n.sha(study['manifest']),
        'reuse_authenticated_utc':s.utc(),'origin_reconstruction_rerun':False,
        'original_receipt':str(original_proof),'original_receipt_sha256':n.sha(original_proof),
        'original_imports':str(original_imports),'original_imports_sha256':n.sha(original_imports),
        'reuse_worker_sha256':n.sha(__file__),'exact_source_and_imported_parser_files_preserved':True,
        'origin_generation':str(old_files),'new_generation':str(files),'only_allowed_changes':sorted(allowed)}
    target=ROOT/f'native_source_preflight_{w}.json'
    if target.exists():
        existing=n.read(target)
        for key in ('original_receipt_sha256','original_imports_sha256','source_generation_hash','manifest_sha256','only_allowed_changes'):
            assert existing[key]==reused[key],key
        # Earlier successful receipt retains its archived creator identity.
        preservation=n.read(ROOT/'preparation_failed_attempt_0/preservation.json')['files']
        assert n.sha(target)==preservation[str(target)]['sha256']
        assert existing['reuse_worker_sha256']==preservation[str(t.REPO/'scripts/olr_minute_exit_clockfix_v5.py')]['sha256']
    else:n.write(target,reused)
    import_target=ROOT/f'native_source_imports_{w}.json'
    if import_target.exists():assert n.read(import_target)==imports
    else:n.write(import_target,imports)
    bundle=OUT/'review_bundle'/w;bundle.mkdir(parents=True,exist_ok=True)
    old_patch=(ORIGINAL_OUT/'review_bundle'/w/'implementation.patch').read_text(encoding='utf-8')
    (bundle/'implementation.patch').write_text(old_patch+''.join(diff),encoding='utf-8')
    print(dict(window=w,phase='exact_source_reuse_authenticated',source_generation_hash=source),flush=True)
    return source


def resume_window(w):
    recovery=n.read(ROOT/'preparation_resume.json')
    preserved=ROOT/'preparation_failed_attempt_0/preservation.json'
    assert n.sha(preserved)==recovery['preservation_sha256']
    saved=n.read(preserved)['files'];path=t.ENGINES/w/'ready.json'
    assert n.sha(path)==saved[str(path)]['sha256']
    ready=n.read(path);manifest_path=Path(ready['study']['manifest'])
    assert n.sha(manifest_path)==saved[str(manifest_path)]['sha256']
    parent=n.read(ORIGINAL/'native_engines'/w/'ready.json');files=Path(ready['files']);old_files=Path(parent['files'])
    assert ready['parent_ready_sha256']==n.sha(ORIGINAL/'native_engines'/w/'ready.json')
    diff=[]
    for name in CHANGED:
        assert n.sha(files/name)==n.sha(t.REPO/name)==ready['manifest']['files'][name]
        diff.extend(difflib.unified_diff((old_files/name).read_text(encoding='utf-8').splitlines(True),
            (files/name).read_text(encoding='utf-8').splitlines(True),fromfile='a/'+w+'/'+name,tofile='b/'+w+'/'+name))
    removed=set()
    for c in n.read(ROOT/'candidate_plan.json')['candidates']:
        name='research/minute_exit_v5/preflight_snapshots/'+c['id']+'.json'
        removed.update((name,str(Path(name).with_suffix('.receipt.json')).replace('\\','/')))
    print(dict(window=w,phase='authenticate_completed_copy_without_rebuilding'),flush=True)
    return certify_window(w,parent,ready,diff,removed)


def prepare(resume=False):
    configure();harness=authenticate_old()
    assert not (ROOT/'preparation_complete.json').exists(),'Do not rerun completed generation preparation'
    assert not (ROOT/'cases').exists()
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('registration.json','candidate_plan.json','source_preparation_complete.json','entry_identity_repair.json','temporary_storage_repair.json'):
        install(ORIGINAL/name,ROOT/name,n.sha(ORIGINAL/name))
    install(ROOT/'registration.json',OUT/'registration.json',n.sha(ROOT/'registration.json'))
    protocol=n.read(ORIGINAL/'protocol.json');protocol['generation_directories']={w:str(t.ENGINES) for w in ('train','oos')};n.write(ROOT/'protocol.json',protocol)
    mechanism=n.read(ORIGINAL/'mechanism_plan.json');mechanism['snapshot_acceleration']['cache_root']=str(ROOT/'snapshot_cache')
    sources={w:(resume_window(w) if resume else clone_window(w)) for w in ('train','oos')}
    mechanism['snapshot_acceleration']['validated_source_hashes']=sources;n.write(ROOT/'mechanism_plan.json',mechanism)
    n.write(ROOT/'preparation_complete.json',dict(utc=s.utc(),sources=sources,
        source_preparation_sha256=n.sha(ROOT/'source_preparation_complete.json'),
        original_partial_control_failure_preserved=str(ARCHIVE),new_generation_performance_started=False))
    changed=set(CHANGED)|{'tests/strategy_olr/test_minute_exit_v5.py'}
    for name,digest in harness['files'].items():
        if name not in changed:assert n.sha(t.REPO/name)==digest,name
    files={name:n.sha(t.REPO/name) for name in harness['files']}
    for name in ('scripts/olr_minute_exit_clockfix_v5.py','tests/strategy_olr/test_minute_clockfix_reuse_v5.py'):
        files[name]=n.sha(t.REPO/name)
    n.write(ROOT/'engineering_controller_harness.json',dict(files=files,utc=s.utc(),
        original_harness_sha256=n.sha(ORIGINAL_OUT/'research_harness.json'),preservation_sha256=PRESERVATION_SHA))


def preflight():
    configure()
    from scripts import olr_minute_exit_preflight_v5 as p
    p.preflight()
    n.write(ROOT/'engineering_complete.json',dict(utc=s.utc(),no_performance_executed=True,
        preflight_sha256=n.sha(ROOT/'preflight_complete.json'),harness_sha256=n.sha(ROOT/'engineering_controller_harness.json'),
        preserved_partial_control_failure=str(ARCHIVE),repair='Nonexecuting closing-observation timers; capacity unchanged'))


def freeze():
    configure()
    from scripts import olr_minute_exit_freeze_v5 as f
    f.main()
    # Additional recovery evidence is bound before this generation's first
    # replay. The original frozen harness remains untouched.
    value=n.read(OUT/'research_harness.json')
    extras=[ROOT/'launch_harness.json',ROOT/'reuse_tests.log',ROOT/'reuse_tests.xml',ARCHIVE/'preservation.json',ORIGINAL_OUT/'research_harness.json']
    if (ROOT/'preparation_resume.json').exists():
        extras.extend((ROOT/'preparation_resume.json',ROOT/'preparation_failed_attempt_0/preservation.json'))
    for w in ('train','oos'):extras.extend((ROOT/'reused_evidence'/w).glob('*.json'))
    value['artifacts'].update({str(p):n.sha(p) for p in extras})
    value['terminal_observation_repair_generation']=True
    n.write(OUT/'research_harness.json',value)


def child(phase):
    bound=n.read(ROOT/'launch_harness.json')
    for name,digest in bound['files'].items():assert n.sha(t.REPO/name)==digest,name
    command=[sys.executable,'-B','-u','-m','scripts.olr_minute_exit_clockfix_v5',phase]
    with (ROOT/(phase+'.controller.log')).open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=t.REPO,stdout=log,stderr=subprocess.STDOUT)
        query="Get-CimInstance Win32_Process -Filter 'ProcessId="+str(process.pid)+"' | Select-Object CommandLine,@{n='creation_time_utc';e={$_.CreationDate.ToUniversalTime().ToString('o')}} | ConvertTo-Json -Compress"
        actual=json.loads(subprocess.check_output(['powershell.exe','-NoProfile','-Command',query],text=True))
        n.write(ROOT/(phase+'_process.json'),dict(pid=process.pid,command_line=actual['CommandLine'],creation_time_utc=actual['creation_time_utc'],launch_utc=s.utc()))
        print(dict(phase=phase,pid=process.pid,status='running'),flush=True)
        rc=process.wait()
    if rc:raise RuntimeError('Preserve failed clockfix phase before repair: '+phase)
    print(dict(phase=phase,status='complete'),flush=True)


def resume_prepare():prepare(resume=True)


def run(resume=False):
    configure()
    for phase in ('resume_prepare' if resume else 'prepare','preflight','freeze','qualify'):child(phase)
    n.write(ROOT/'controller_complete.json',dict(utc=s.utc(),qualification_complete_sha256=n.sha(ROOT/'qualification_complete.json'),report_review_remaining=True))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('run','run_resume','prepare','resume_prepare','preflight','freeze','qualify','report','verify'));ap.add_argument('--chart-reviewed',action='store_true');args=ap.parse_args()
    configure()
    try:
        if args.phase=='run_resume':run(resume=True)
        elif args.phase in ('run','prepare','resume_prepare','preflight','freeze'):globals()[args.phase]()
        elif args.phase=='qualify':
            from scripts import olr_minute_exit_qualify_v5 as q
            q.qualify()
        elif args.phase=='report':
            from scripts import olr_minute_exit_report_v5 as r
            r.main()
        else:
            from scripts import olr_minute_exit_verify_v5 as v
            v.main(args.chart_reviewed)
    except Exception:
        n.write(ROOT/(args.phase+'_failure.json'),dict(utc=s.utc(),traceback=traceback.format_exc()));raise

"""Build a new native generation on E; never write previous frozen inputs."""
import copy,difflib,json,os,shutil,sqlite3,subprocess,sys
from pathlib import Path
from scripts import olr_minute_exit_study_v5 as t
from scripts.olr_minute_exit_patches_v5 import patch
n,s,b=t.n,t.s,t.b


def authenticate_materialization():
    receipt=n.read(t.ROOT/'source_preparation_complete.json')
    assert receipt['script_sha256']==n.sha(t.REPO/'scripts/olr_minute_exit_materialize_v5.py')
    for w,digest in receipt['receipts'].items():
        path=t.ROOT/'source_inputs'/w/'complete.json';assert n.sha(path)==digest
        row=n.read(path);assert n.sha(row['database'])==row['database_sha256']
        for name,expected in row['sources'].items():
            if n.sha(name)!=expected:
                reuse=receipt['authenticated_completed_reuse'][w]
                assert w=='train' and Path(name).name=='olr_minute_exit_materialize_v5.py'
                assert reuse['receipt_sha256']==digest and n.sha(reuse['archived_builder'])==expected==reuse['original_builder_sha256']
    return receipt


def install(source,target,digest):
    source,target=Path(source),Path(target)
    assert n.sha(source)==digest,str(source)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        assert n.sha(target)==digest,str(target);return
    if source.drive.lower()==target.drive.lower() and source.suffix.lower() in ('.parquet','.sqlite','.db','.gz','.pickle','.pkl'):
        os.link(source,target)
    else:shutil.copy2(source,target)
    assert n.sha(target)==digest


def prepare_window(w):
    directory=t.ENGINES/w;ready_path=directory/'ready.json'
    if ready_path.exists():return n.read(ready_path)
    if directory.exists():raise FileExistsError('Preserve incomplete native generation before repair: '+str(directory))
    source_ready=Path.home()/f'.codex/research/olr_repair_v4/{w}/ready.json'
    parent=n.read(source_ready);old_files=Path(parent['files']);files=directory/'inputs/files'
    manifest=copy.deepcopy(parent['manifest']);manifest.pop('manifest_hash',None)
    study=copy.deepcopy(parent['study'])
    for k in ('capacity_checkpoint','capacity_checkpoint_completion','capacity_checkpoint_result'):study.pop(k,None)
    print(dict(window=w,phase='copy_authenticated_parent',files=len(manifest['files'])),flush=True)
    for name,digest in manifest['files'].items():install(old_files/name,files/name,digest)
    diffs=[];changes=[]
    for name in t.PATCHED:
        target=files/name;original=target.read_text(encoding='utf-8') if target.exists() else ''
        value=(t.REPO/name).read_text(encoding='utf-8') if name in t.NEW_MODULES else patch(name,original)
        compile(value,str(target),'exec');target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(value,encoding='utf-8',newline='\n');manifest['files'][name]=n.sha(target)
        changes.append(dict(path=name,before=n.sha(old_files/name) if (old_files/name).exists() else None,after=n.sha(target)))
        diffs.extend(difflib.unified_diff(original.splitlines(True),value.splitlines(True),fromfile='a/'+w+'/'+name,tofile='b/'+w+'/'+name))
    aliases={};digest_names={v:k for k,v in manifest['files'].items()}
    def bind(source,expected=None,name=None):
        source=Path(source);expected=expected or n.sha(source)
        destination=name or digest_names.get(expected) or ('research/minute_exit_v5/origins/'+expected+'/'+source.name)
        install(source,files/destination,expected);manifest['files'][destination]=expected
        aliases[str(source)]=destination;digest_names.setdefault(expected,destination)
        return destination
    material=n.read(t.ROOT/'source_inputs'/w/'complete.json')
    for name,item in material.get('native_origin_dependencies',{}).items():
        assert name not in manifest['files'] and not (files/name).exists()
        event=n.read(old_files/item['event'])
        assert n.sha(old_files/item['event'])==item['event_sha256']
        assert event['source_files'][name]==item['sha256']
        bind(item['path'],item['sha256'],name)
    for name,digest in material['sources'].items():
        source=Path(name)
        if n.sha(source)!=digest:
            assert w=='train' and source.name=='olr_minute_exit_materialize_v5.py'
            source=t.ROOT/'source_failed_attempt_0/materialize_program.py';assert n.sha(source)==digest
        actual=bind(source,digest);aliases[name]=actual
    plan_path=t.SOURCE_OUT/'minute_exit_source_plan.json';plan=n.read(plan_path)
    for name,digest in plan['sources'].items():bind(name,digest)
    for row in plan['windows'][w]['primary'].values():bind(row['path'],row['sha256'])
    source_plan=bind(plan_path,name='research/minute_exit_v5/minute_source_plan.json')
    resolution=[bind(t.SOURCE_OUT/'minute_exit_source_resolution'/stage/'resolution.json') for stage in (w,w+'_completion')]
    minute_db=bind(material['database'],material['database_sha256'],'research/minute_exit_v5/market_1m.sqlite')
    definition=dict(version='olr-minute-source-definition-v5',window=w,source_plan=source_plan,
        aliases=aliases,resolution_receipts=resolution,minute_database=minute_db,canonical_database=study['database'])
    replacements={};shared=manifest['evidence']['shared_market_inputs']
    if w=='oos':
        correction_path=t.ROOT/'clock_source_correction/complete.json';correction=n.read(correction_path)
        definition['closing_clock_correction']=bind(correction_path,name='research/minute_exit_v5/closing_clock_correction.json')
        definition['original_canonical_database']=study['database']
        calendar_name='kis_core/data/krx_intraday_halts.yaml'
        definition['original_calendar']=bind(old_files/calendar_name,name='research/minute_exit_v5/original_calendar.yaml')
        database=bind(correction['database'],correction['database_sha256'],'research/minute_exit_v5/market_5m_clock.sqlite')
        study.update(database=database,database_hash=correction['database_sha256']);definition['canonical_database']=database
        for old,item in correction['coarse_files'].items():
            name=bind(item['path'],item['sha256'],'research/minute_exit_v5/coarse/'+Path(old).name)
            replacements[old]=name;shared['files'][name]=manifest['files'][name]
        for symbol,names in shared['intraday_files'].items():shared['intraday_files'][symbol]=[replacements.get(name,name) for name in names]
        # Parent text files were copied, never hardlinked. Replace this NEW YAML only.
        shutil.copy2(correction['calendar'],files/calendar_name);manifest['files'][calendar_name]=n.sha(files/calendar_name)
        for original,target in list(aliases.items()):
            if target==calendar_name:aliases[original]=definition['original_calendar']
        shared['files'][calendar_name]=manifest['files'][calendar_name]
        for row in correction['corrections']:
            name=bind(t.ROOT/'clock_source_correction'/row['event'],row['event_sha256'],row['event'])
            shared['files'][name]=manifest['files'][name]
        shared['market_input_hash']=n.sig({k:v for k,v in shared.items() if k!='market_input_hash'})
        study['configuration']['market_input_hash']=shared['market_input_hash']
    definition_name='research/minute_exit_v5/minute_source_definition.json'
    n.write(files/definition_name,definition);manifest['files'][definition_name]=n.sha(files/definition_name)
    study['minute_source_definition']=definition_name;study['minute_original_grids']={}
    reference=t.PARENT.parent/'participation_v5/cases/caa25a36d10de3dc4'/w
    for scenario in ('baseline','combined','baseline_capped'):
        folder=reference/('100000000_'+scenario);s.forensic.authenticate(folder)
        name=bind(folder/'equity.json',name='research/minute_exit_v5/reference_marks/'+scenario+'.json')
        study['minute_original_grids'][scenario]=name
    # Changes to metadata/source identity are explicit; no performance is run here.
    actual=n.source_hash(files);manifest['source_generation_hash']=actual
    study.update(code=str(files),source_generation_hash=actual,manifest=str(directory/'inputs/manifest.json'),spec_path='research/minute_exit_v5/control_spec.json')
    study['configuration']['input_manifest_path']=study['manifest']
    draft=directory/'acceptance_request.json';n.write(draft,dict(files=str(files),manifest=manifest,study=study,coarse_replacements=replacements))
    print(dict(window=w,phase='recompute_native_acceptance'),flush=True)
    with (directory/'acceptance.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-B','-u',str(t.REPO/'scripts/olr_minute_exit_native_worker_v5.py'),'acceptance',str(draft)],cwd=files,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Native source acceptance failed: '+str(directory/'acceptance.log'))
    imports=n.read(directory/'acceptance_imports.json')
    assert imports['passed'] and imports['worker_sha256']==n.sha(t.REPO/'scripts/olr_minute_exit_native_worker_v5.py')
    manifest=n.read(directory/'acceptance_complete.json')
    n.write(files/study['spec_path'],{k:v for k,v in study.items() if k not in ('code','manifest','source_generation_hash','spec_path')})
    manifest['files'][study['spec_path']]=n.sha(files/study['spec_path'])
    n.verify_market_bindings(manifest);assert not n.retained_parser_mismatches(manifest,files)
    n.write(study['manifest'],{**manifest,'manifest_hash':n.sig(manifest)})
    record=dict(window=w,files=str(files),study=study,manifest=manifest,parent_ready_sha256=n.sha(source_ready),changes=changes)
    n.write(ready_path,record)
    patch_path=t.OUT/'review_bundle'/w/'implementation.patch';patch_path.parent.mkdir(parents=True,exist_ok=True)
    patch_path.write_text(''.join(diffs),encoding='utf-8')
    print(dict(window=w,phase='native_generation_ready',source_hash=actual),flush=True)
    return record


def prepare():
    assert not (t.ROOT/'cases').exists(),'Never change a generation after outcomes'
    authenticate_materialization();t.register();records=[prepare_window(w) for w in ('train','oos')]
    hashes={r['window']:r['study']['source_generation_hash'] for r in records}
    mechanism=n.read(t.ROOT/'mechanism_plan.json');mechanism['snapshot_acceleration']['validated_source_hashes']=hashes
    n.write(t.ROOT/'mechanism_plan.json',mechanism)
    n.write(t.ROOT/'preparation_complete.json',dict(utc=s.utc(),sources=hashes,ready_hashes={w:n.sha(t.ENGINES/w/'ready.json') for w in ('train','oos')},
        source_preparation_sha256=n.sha(t.ROOT/'source_preparation_complete.json'),performance_outcomes_consumed=False,
        remaining='Fresh native snapshot parity, actual origin/calendar/source audit, frozen harness and tests before native performance.'))


if __name__=='__main__':
    prepare()

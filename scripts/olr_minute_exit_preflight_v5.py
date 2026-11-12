"""Independent full source reconstruction and native entry parity, before P&L."""
from datetime import date
from pathlib import Path
import copy,json,os,sqlite3,subprocess,sys
from scripts import olr_minute_exit_study_v5 as t
from scripts.olr_minute_exit_entry_identity_v5 import normalize_research_identity
n,s,b=t.n,t.s,t.b


def semantic_snapshots(rows):
    # These exact fields describe source/build identity, not trade selection.
    # Keep every candidate feature/rank/sizing value and all rejection metadata.
    ignored={'artifact_hash','source_fingerprint','current_sector_history_builder_sha256'}
    def clean(value):
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k not in ignored}
        if isinstance(value,list):return [clean(v) for v in value]
        return value
    return clean(rows)


def legacy_snapshot_path(identity,files,manifest):
    """A resumed audit must still compare against the original retained snapshot."""
    path=Path(identity['old_snapshot_path']).resolve();files=Path(files).resolve()
    if not path.is_relative_to(files):raise ValueError('Legacy entry evidence outside generation')
    expected=identity['old_snapshot_sha256']
    if manifest['files'].get(path.relative_to(files).as_posix())!=expected or n.sha(path)!=expected:
        raise ValueError('Legacy entry evidence changed')
    return path


def preflight():
    if (t.ROOT/'preflight_complete.json').exists():raise FileExistsError('Do not repeat completed preflight')
    if (t.ROOT/'cases').exists():raise RuntimeError('Preflight cannot change after outcomes')
    catalog=n.read(t.ROOT/'candidate_plan.json')['candidates'];evidence=[]
    for w in ('train','oos'):
        ready_path=t.ENGINES/w/'ready.json';ready=n.read(ready_path);files=Path(ready['files'])
        proof=t.ROOT/f'native_source_preflight_{w}.json'
        if not proof.exists():
            log=t.ROOT/f'native_source_preflight_{w}.log'
            with log.open('x',encoding='utf-8') as stream:
                rc=subprocess.run([sys.executable,'-B','-u',str(t.REPO/'scripts/olr_minute_exit_native_worker_v5.py'),'sources',str(t.ROOT),w],cwd=files,stdout=stream,stderr=subprocess.STDOUT).returncode
            if rc:raise RuntimeError('Independent source preflight failed: '+str(log))
        receipt=n.read(proof);assert receipt['worker_sha256']==n.sha(t.REPO/'scripts/olr_minute_exit_native_worker_v5.py') and receipt['source_generation_hash']==ready['study']['source_generation_hash']
        imports=n.read(t.ROOT/f'native_source_imports_{w}.json')
        assert imports['passed'] and imports['worker_sha256']==n.sha(t.REPO/'scripts/olr_minute_exit_native_worker_v5.py')
        output=[]
        for candidate in catalog:
            name='research/minute_exit_v5/preflight_snapshots/'+candidate['id']+'.json';target=files/name
            candidate_path=t.ROOT/'preflight_candidates'/(candidate['id']+'.json');n.write(candidate_path,candidate)
            if target.exists():
                cp=n.read(target.with_suffix('.receipt.json'));assert cp['snapshots_sha256']==n.sha(target) and cp['source_hash']==ready['study']['source_generation_hash']
                assert cp['candidate_hash']==candidate['mutations_hash'] and cp['acceleration_sha256']==n.sha(t.REPO/'scripts/olr_minute_exit_snapshots_v5.py')
            else:
                log=t.ROOT/f'snapshot_preflight_{w}_{candidate["id"]}.log'
                rc=n.run_child([sys.executable,'-B','-u',str(Path(n.__file__).resolve()),'snapshots','--root',str(t.ROOT),
                    '--window',w,'--candidate',str(candidate_path),'--output',str(target)],files,log)
                if rc:raise RuntimeError('Native snapshot rebuild failed: '+str(log))
            output.append((name,n.read(target)))
        assert output[0][1]==output[1][1],'Exit clock flag changed native entry artifacts'
        identity_path=t.ROOT/f'entry_identity_{w}.json'
        if not identity_path.exists():
            identity_log=t.ROOT/f'entry_identity_{w}.log'
            rc=n.run_child([sys.executable,'-B','-u',str(t.REPO/'scripts/olr_minute_exit_entry_identity_v5.py'),str(t.ROOT),w],files,identity_log)
            if rc:raise RuntimeError('Independent entry identity audit failed: '+str(identity_log))
        identity=n.read(identity_path)
        assert identity['worker_sha256']==n.sha(t.REPO/'scripts/olr_minute_exit_entry_identity_v5.py')
        assert identity['source_hash']==ready['study']['source_generation_hash'] and identity['window']==w
        assert set(identity['artifacts'])=={str(files/name) for name,_ in output}
        for path,item in identity['artifacts'].items():
            assert n.sha(path)==item['sha256'] and n.sha(Path(path).with_suffix('.receipt.json'))==item['receipt_sha256']
        for path,digest in identity['native_import_hashes'].items():
            assert ready['manifest']['files'][path]==n.sha(files/path)==digest
        old=n.read(legacy_snapshot_path(identity,files,ready['manifest']))
        # Full-config hashes include historical execution-only defaults. They are
        # checked independently above, never blindly ignored or changed on disk.
        normalized=normalize_research_identity(output[0][1],identity['new_research_hash'],identity['legacy_research_hash'])
        a,z=semantic_snapshots(old),semantic_snapshots(normalized)
        if a!=z:
            n.write(t.ROOT/f'entry_parity_failure_{w}.json',dict(old=a,new=z))
            raise ValueError('Native entry fields differ; inspect parity failure before performance')
        manifest=ready['manifest'];study=ready['study'];study['olr_snapshots']=output[0][0]
        study['canonical_olr_snapshot_hash']=n.sig(output[0][1])
        for name,rows in output:
            manifest['files'][name]=n.sha(files/name)
            manifest['files'][str(Path(name).with_suffix('.receipt.json')).replace('\\','/')]=n.sha((files/name).with_suffix('.receipt.json'))
        n.write(files/study['spec_path'],{k:v for k,v in study.items() if k not in ('code','manifest','source_generation_hash','spec_path')})
        manifest['files'][study['spec_path']]=n.sha(files/study['spec_path'])
        n.write(study['manifest'],{**manifest,'manifest_hash':n.sig(manifest)});n.write(ready_path,ready)
        evidence.append(dict(window=w,source_preflight_sha256=n.sha(proof),native_control_and_E_artifacts_exact=True,
            entry_identity_path=str(t.ROOT/f'entry_identity_{w}.json'),entry_identity_sha256=n.sha(t.ROOT/f'entry_identity_{w}.json'),
            retained_entry_features_rank_size_rejections_exact=True,snapshot_path=str(files/output[0][0]),snapshot_sha256=n.sha(files/output[0][0]),
            old_snapshot_sha256=n.sig(old),provenance_only_fields=['artifact_hash','source_fingerprint','current_sector_history_builder_sha256']))
    n.write(t.ROOT/'preflight_complete.json',dict(utc=s.utc(),evidence=evidence,worker_sha256=n.sha(__file__),
        source_hashes=n.read(t.ROOT/'preparation_complete.json')['sources'],registration_sha256=n.sha(t.ROOT/'registration.json'),no_pnl_consumed=True))


if __name__=='__main__':
    preflight()

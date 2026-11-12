"""Independent native config/artifact authentication; no price or P&L changes."""
import copy,hashlib,importlib.util,json,sys
from pathlib import Path


def normalize_research_identity(rows,expected,canonical):
    """Validate, then map only known full-config provenance fields for comparison."""
    result=copy.deepcopy(rows)
    for row in result:
        for record in [row,*row['candidates']]:
            metadata=record['metadata']
            if metadata.get('research_config_hash')!=expected:
                raise ValueError('Unexplained research configuration identity')
            metadata['research_config_hash']=canonical
    return result


def main(root,window):
    root=Path(root);repo=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('isolated_native_worker',repo/'scripts/olr_minute_exit_native_worker_v5.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    ready=worker.read(root/'native_engines'/window/'ready.json');files=worker.isolate(ready['files'])
    from strategy_olr.config import OLRConfig
    from strategy_olr.research import _research_config_fingerprint,final_candidate_config_fingerprint
    from strategy_olr.models import OLRDailySnapshot
    plan=worker.read(root/'candidate_plan.json');catalog=plan['candidates']
    request=worker.read(files/ready['study']['olr_request'])
    control=next(c for c in catalog if c['mutations']['olr.research.minute_exit'] is False)
    pure={k:v for k,v in control['mutations'].items() if not k.startswith('capacity_')}
    cfg=OLRConfig.from_mapping({**request['config'],'initial_mutations':pure,'initial_equity':100000000,
        'date_range':ready['study']['configuration']['date_range']},pure)
    research_hash=_research_config_fingerprint(cfg);candidate_hash=final_candidate_config_fingerprint(cfg)
    old_path=files/ready['study']['olr_snapshots']
    assert ready['manifest']['files'][ready['study']['olr_snapshots']]==worker.sha(old_path)
    old=worker.read(old_path);old_hashes={x['metadata']['research_config_hash'] for x in old}
    assert len(old_hashes)==1,'Legacy provenance changed within the window'
    old_hash=next(iter(old_hashes));normalize_research_identity(old,old_hash,old_hash)
    checked={}
    for candidate in catalog:
        path=files/'research/minute_exit_v5/preflight_snapshots'/(candidate['id']+'.json')
        rows=worker.read(path);receipt=worker.read(path.with_suffix('.receipt.json'))
        assert receipt['snapshots_sha256']==worker.sha(path)
        assert receipt['source_hash']==ready['study']['source_generation_hash']
        assert receipt['candidate_hash']==candidate['mutations_hash']
        assert receipt['acceleration_sha256']==worker.sha(repo/'scripts/olr_minute_exit_snapshots_v5.py')
        normalize_research_identity(rows,research_hash,old_hash)
        for row in rows:
            assert OLRDailySnapshot.from_json_dict(row).artifact_hash==row['artifact_hash'],'Native artifact identity mismatch'
            assert row['metadata']['final_candidate_config_hash']==candidate_hash
            assert row['metadata']['candidate_config_hash']==candidate_hash
        checked[str(path)]={'sha256':worker.sha(path),'receipt_sha256':worker.sha(path.with_suffix('.receipt.json')),'sessions':len(rows)}
    imports=worker.verify_imports(files,ready['manifest'])
    worker.write(root/f'entry_identity_{window}.json',dict(window=window,worker_sha256=worker.sha(__file__),
        source_hash=ready['study']['source_generation_hash'],old_snapshot_path=str(old_path),old_snapshot_sha256=worker.sha(old_path),
        new_research_hash=research_hash,legacy_research_hash=old_hash,final_candidate_hash=candidate_hash,
        artifacts=checked,native_import_hashes=imports,raw_artifacts_preserved=True,no_pnl_consumed=True,
        scope='Both policies use canonical entry config with only minute-exit flag False. Independently recomputed hashes; legacy full-config provenance mapped only at snapshot/candidate metadata for comparison.'))


if __name__=='__main__':main(*sys.argv[1:])

"""Finalize reviewed sources before the first native trial, never during one."""
import copy
import difflib
import shutil
import subprocess
import sys
from scripts import olr_precision_study_v3 as t
from scripts import olr_precision_prepare_v3 as preparation
n,s=t.n,t.s


def main():
    assert (t.ROOT/'preparation_complete.json').exists()
    assert not (t.ROOT/'cases').exists(), 'Native generations are immutable once execution begins'
    records=[];hashes={};diffs=[]
    archive=t.OUT/'prequalification_source_revision'
    archive.mkdir(parents=True,exist_ok=True)
    for w in ('train','oos'):
        ready_path=t.ENGINES/w/'ready.json';saved=n.read(ready_path)
        files=t.ENGINES/w/'inputs/files';study=saved['study'];manifest=copy.deepcopy(saved['manifest'])
        parent=n.read(t.PARENT/'runtime'/t.REFERENCE/w/'baseline/study.json')
        parent_files=__import__('pathlib').Path(parent['code'])
        changes=[]
        if not (archive/(w+'_ready.json')).exists():n.write(archive/(w+'_ready.json'),saved)
        for name in t.PATCHED:
            target=files/name;before=parent_files/name
            assert n.sha(target)==manifest['files'][name],name
            old=before.read_text(encoding='utf-8') if before.exists() else ''
            value=(t.REPO/name).read_text(encoding='utf-8') if name in t.NEW_MODULES or name=='scripts/olr_liquidation_panel_v3.py' else preparation.patch(name,old)
            compile(value,str(target),'exec')
            if target.read_text(encoding='utf-8')!=value:
                prior=archive/w/name;prior.parent.mkdir(parents=True,exist_ok=True)
                if not prior.exists():shutil.copy2(target,prior)
                target.write_text(value,encoding='utf-8',newline='\n')
            manifest['files'][name]=n.sha(target)
            dest=t.OUT/'review_bundle'/w/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,dest)
            changes.append({'window':w,'name':name,'before':n.sha(before) if before.exists() else None,'after':n.sha(target)})
            diffs.extend(difflib.unified_diff(old.splitlines(True),value.splitlines(True),fromfile='a/'+w+'/'+name,tofile='b/'+w+'/'+name))
        actual=subprocess.check_output([sys.executable,'-B','-c','from backtests.auto.shared.cache_keys import source_generation_hash; print(source_generation_hash())'],cwd=files,text=True).strip()
        hashes[w]=actual;manifest['source_generation_hash']=actual;study['source_generation_hash']=actual
        n.verify_market_bindings(manifest);assert not n.retained_parser_mismatches(manifest,files)
        n.write(study['manifest'],{**manifest,'manifest_hash':n.sig(manifest)})
        record={**saved['preparation_record'],'new_source':actual,'changes':changes}
        n.write(ready_path,{**saved,'study':study,'manifest':manifest,'preparation_record':record});records.append(record)
    prepared=n.read(t.ROOT/'preparation_complete.json')
    prepared.update(sources=hashes,changes=records,prequalification_finalized_utc=s.utc(),
        prequalification_repair='Persist entry-session observed low across later partial buy metadata merges; no native outcomes existed at revision.')
    n.write(t.ROOT/'preparation_complete.json',prepared)
    plan=n.read(t.ROOT/'mechanism_plan.json');plan['snapshot_acceleration']['validated_source_hashes']=hashes;n.write(t.ROOT/'mechanism_plan.json',plan)
    (t.OUT/'review_bundle/implementation.patch').write_text(''.join(diffs),encoding='utf-8')
    n.write(t.OUT/'implementation_receipt.json',{'changed_sources':[c for r in records for c in r['changes']],
        'default_off':True,'promotion':False,'prequalification_finalized_utc':s.utc()})
    print({'finalized':True,'sources':hashes})


if __name__=='__main__':main()

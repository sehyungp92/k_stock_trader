import json
from pathlib import Path
import pytest
from scripts import olr_native_acceleration as a
from scripts import olr_round1_comprehensive_study as n


def test_additional_native_jobs_require_headroom():
    policy={'minimum_available_mb_for_additional_job':3072}
    assert a.additional_job_allowed(policy,0,1000)
    assert not a.additional_job_allowed(policy,1,3071)
    assert a.additional_job_allowed(policy,1,3072)
    assert not a.additional_job_allowed(policy,1,None)
    assert a.additional_job_allowed(None,1,None)


def test_replay_generation_preserves_inputs_and_detects_receipt_tampering(tmp_path):
    old=tmp_path/'old';new=tmp_path/'inputs/files';old.mkdir();new.mkdir(parents=True)
    (old/'data.json').write_text('[1,2,3]')
    manifest={'files':{'data.json':a.sha(old/'data.json')},'source_generation_hash':'old','source_root':str(old)}
    mp=tmp_path/'original.manifest.json';a.write(mp,{**manifest,'manifest_hash':a.signature(manifest)})
    study={'code':str(old),'manifest':str(mp),'spec_path':'spec.json','source_generation_hash':'old',
           'configuration':{'date_range':{'start':'a','end':'b'}},'candidates':{'candidate':{'x':5}}}
    original=tmp_path/'study.json';a.write(original,study)
    generation={'engine':str(new),'original_source_hash':'old','source_generation_hash':'new',
                'files':{},'changed_files':{}}
    gp=tmp_path/'generation.json';a.write(gp,generation)
    rp=tmp_path/'replay_study.json';a.prepare_replay(original,gp,rp)
    assert (new/'data.json').read_bytes()==(old/'data.json').read_bytes()
    resolved=n.execution_registration(original,'new')
    assert resolved['candidates']==study['candidates']
    assert n.execution_registration(original,'old')==study
    assert resolved['configuration']['date_range']==study['configuration']['date_range']
    rp.write_text(rp.read_text()+' ')
    with pytest.raises(ValueError,match='receipt'):n.execution_registration(original,'new')


def test_low_memory_queue_finishes_all_jobs_serially(tmp_path,monkeypatch):
    from scripts import olr_round1_mechanism_study as m
    import threading,time
    a.write(tmp_path/'mechanism_plan.json',{'resource_admission':{'minimum_available_mb_for_additional_job':3072}})
    monkeypatch.setattr(a,'available_memory_mb',lambda:2000)
    counts={'active':0,'peak':0};lock=threading.Lock()
    def evaluate(root,candidate,window):
        with lock:
            counts['active']+=1;counts['peak']=max(counts['peak'],counts['active'])
        time.sleep(.02)
        with lock:counts['active']-=1
        return {'status':'completed','qualified':True}
    monkeypatch.setattr(m,'evaluate',evaluate)
    jobs=[dict(candidate={'id':str(i),'label':str(i)},window='train',index=i+1) for i in range(3)]
    m.run_evaluation_jobs(tmp_path,jobs,2,'test')
    assert counts=={'active':0,'peak':1}
    assert a.read(tmp_path/'progress_train.json')['completed_evaluations']==3

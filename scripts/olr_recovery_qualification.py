"""Source-bound evidence projection for the frozen OLR recovery workflow.

Native execution stays byte-for-byte unchanged. The supplemental orchestrator
projects actual audit counts into the shared framework; it neither fills missing
counts with zeros nor changes any economic, support or source-audit requirement.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('register','qualify','accept','diagnose'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();root=a.root.resolve();registration=root/'evidence_workflow_registration.json'
    study=read(root/'study_100000000.json');sys.path.insert(0,study['code'])
    from backtests.strategies.olr.allocation_recovery import (
        AllocationRecoveryPlugin,CAPITALS,SCENARIOS,location,checked_case,write_json)
    from backtests.auto.shared.phase_runner import PhaseRunner
    if a.command=='register':
        if registration.exists():raise FileExistsError('Evidence workflow is already frozen')
        screen=read(root/'screen_verdict.json')
        if not screen['complete']:raise ValueError('Screen must be complete')
        frozen=root/'evidence_workflow';frozen.mkdir(exist_ok=False)
        wrapper=frozen/Path(__file__).name;shutil.copy2(__file__,wrapper)
        helper=Path(__file__).resolve().parents[1]/'backtests/strategies/olr/recovery_evidence.py'
        shutil.copy2(helper,frozen/'recovery_evidence.py')
        sparse=read(root/'sparse_recycling_diagnostic_registration.json')
        write(registration,{'registered_at_utc':datetime.now(timezone.utc).isoformat(),
            'purpose':'Expose already-recorded native audit counts to the shared metric contract; preserve every original gate and native replay byte',
            'qualification_candidates':screen['eligible'],'diagnostic_candidate':sparse['profile'],
            'diagnostic_registration_hash':digest(root/'sparse_recycling_diagnostic_registration.json'),
            'original_registration_hash':digest(root/'registration.json'),
            'screen_verdict_hash':digest(root/'screen_verdict.json'),
            'native_source_generation_hash':study['source_generation_hash'],
            'study_hashes':{str(c):digest(root/f'study_{c}.json') for c in CAPITALS},
            'wrapper':str(wrapper),'wrapper_hash':digest(wrapper),
            'helper':str(frozen/'recovery_evidence.py'),'helper_hash':digest(frozen/'recovery_evidence.py'),
            'no_gate_relaxation':True,'diagnostic_promotion_permitted':False,
            'independent_evidence':False,'live_activated':False})
        print(json.dumps({'registered':str(registration)}),flush=True);return
    plan=read(registration)
    if Path(__file__).resolve()!=Path(plan['wrapper']).resolve() or digest(Path(__file__))!=plan['wrapper_hash']:
        raise ValueError('Run the exact frozen evidence workflow')
    if digest(Path(plan['helper']))!=plan['helper_hash']:raise ValueError('Metric projection changed')
    for filename,key in [('registration.json','original_registration_hash'),
                         ('screen_verdict.json','screen_verdict_hash'),
                         ('sparse_recycling_diagnostic_registration.json','diagnostic_registration_hash')]:
        if digest(root/filename)!=plan[key]:raise ValueError('Research registration changed')
    for capital,expected in plan['study_hashes'].items():
        if digest(root/f'study_{capital}.json')!=expected:raise ValueError('Native study changed')
    spec=importlib.util.spec_from_file_location('registered_recovery_evidence',plan['helper'])
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    if a.command=='qualify':
        if not read(root/'qualify_verdict.json')['complete']:raise ValueError('Native capital matrix is incomplete')
        names=plan['qualification_candidates'];mode='qualify_evidence'
    elif a.command=='accept':
        q=read(root/'qualify_evidence_verdict.json')
        if not q['complete'] or not q['eligible'] or q['choice'].get('recipe') not in q['eligible']:
            raise ValueError('No challenger passed reconciled shared-framework qualification')
        names=q['eligible'];mode='accept'
        if not set(names).issubset(plan['qualification_candidates']):raise ValueError('Unregistered finalist')
    else:
        names=[plan['diagnostic_candidate']];mode='diagnose_evidence'
        # The ordinary qualification owns baseline computation. Never race it
        # by starting another writer for an incomplete native case.
        if any(not (location(root,'median',c,s)/'completion.json').exists() for c in CAPITALS for s in SCENARIOS):
            raise ValueError('Wait for all native median controls before the supplemental diagnostic')
    class EvidenceWorkflow(AllocationRecoveryPlugin):
        def __init__(self):
            super().__init__(root,mode='screen',workers=a.workers)
            self.names=names;self.mode=mode;self.capitals=CAPITALS
        def evaluate(self,name):
            record=super().evaluate(name);audits=[];results=[];receipts=[]
            for capital in self.capitals:
                for scenario in SCENARIOS:
                    path=location(root,name,capital,scenario,self.mode=='accept')
                    result,audit=checked_case(path);results.append(result);audits.append(audit)
                    receipts.append({'capital':capital,'scenario':scenario,
                        'path':str(path),'execution_audit_hash':digest(path/'execution_audit.json')})
            record['metrics'].update(helper.hygiene_metrics(audits,results))
            record['native_hygiene_evidence']=receipts
            record['evidence_workflow_registration_hash']=digest(registration)
            write_json(root/(self.mode+'_comparisons')/f'{name}.json',record)
            return record
        def build_end_of_round_artifacts(self,state):
            artifacts=super().build_end_of_round_artifacts(state)
            verdict=read(root/(self.mode+'_verdict.json'))
            if a.command=='diagnose' and (verdict['eligible'] or verdict['choice']!={'recipe':'median'}):
                raise ValueError('Sparse diagnostic cannot promote a policy')
            verdict.update(evidence_workflow_registration_hash=digest(registration),
                diagnostic_only=a.command=='diagnose',native_source_unchanged=True)
            write(root/(self.mode+'_verdict.json'),verdict)
            return artifacts
    PhaseRunner(EvidenceWorkflow(),root/('phased_'+mode+'_verified'),round_name='OLR recovery '+mode,
        max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()


if __name__=='__main__':main()

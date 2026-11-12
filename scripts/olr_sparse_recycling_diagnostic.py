"""Capital diagnostic for an economic winner with insufficient direct support.

This cannot promote a policy or relax the original support gate. Its purpose is
to measure account-size robustness of a sparse intervention, not to convert a
failed screen into acceptance. The immutable wrapper identity is registered
separately from the unchanged native execution source.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import sys


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('register','run'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--profile',default='native_entry_recycle2')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();root=args.root.resolve()
    study=read(root/'study_100000000.json')
    sys.path.insert(0,study['code'])
    from backtests.strategies.olr.allocation_recovery import (
        AllocationRecoveryPlugin, CAPITALS, SCENARIOS, checked_case, cell_failures, support)
    from backtests.auto.shared.phase_runner import PhaseRunner
    registration=root/'sparse_recycling_diagnostic_registration.json'
    if args.command=='register':
        if registration.exists():
            raise FileExistsError('Diagnostic registration is immutable')
        profiles=read(root/'registration.json')['profiles']
        if args.profile not in profiles or args.profile=='median':
            raise ValueError('Diagnostic must use an already registered candidate')
        anchors=[]
        for scenario in SCENARIOS:
            path=root/'cases'/f'{args.profile}_100000000_{scenario}'
            control=root/'cases'/f'median_100000000_{scenario}'
            result,_=checked_case(path);baseline,_=checked_case(control)
            if cell_failures(result,baseline,True):
                raise ValueError('Diagnostic requires both anchor economic gates to pass')
            anchors.append({'path':str(path),'completion_hash':digest(path/'completion.json'),
                'control':str(control),'control_completion_hash':digest(control/'completion.json')})
        evidence=support(root/'cases'/f'{args.profile}_100000000_baseline')
        if evidence['distinct_issuer_days']>=10:
            raise ValueError('A supported candidate belongs in ordinary qualification')
        frozen=root/'diagnostic_tools'/Path(__file__).name
        frozen.parent.mkdir(exist_ok=True);shutil.copy2(__file__,frozen)
        data={'registered_at_utc':datetime.now(timezone.utc).isoformat(),
            'purpose':'Describe capital robustness of an economically improved but support-rejected policy; no promotion',
            'profile':args.profile,'settings':profiles[args.profile],
            'capitals':list(CAPITALS),'scenarios':list(SCENARIOS),
            'native_source_generation_hash':study['source_generation_hash'],
            'study_hashes':{str(c):digest(root/f'study_{c}.json') for c in CAPITALS},
            'original_registration_hash':digest(root/'registration.json'),
            'anchors':anchors,'support':evidence,
            'wrapper':str(frozen),'wrapper_hash':digest(frozen),
            'selection_gates':'All original return, drawdown, frequency, execution, capital-gap and direct-support gates remain enforced',
            'promotion_permitted':False,'independent_evidence':False}
        registration.write_text(json.dumps(data,indent=2),encoding='utf-8')
        print(json.dumps({'registered':str(registration),'support':evidence}),flush=True)
        return
    plan=read(registration)
    if digest(Path(__file__))!=plan['wrapper_hash'] or str(Path(__file__).resolve())!=str(Path(plan['wrapper']).resolve()):
        raise ValueError('Execute the exact frozen diagnostic wrapper')
    if digest(root/'registration.json')!=plan['original_registration_hash']:
        raise ValueError('Original selection registration changed')
    for capital,expected in plan['study_hashes'].items():
        if digest(root/f'study_{capital}.json')!=expected:
            raise ValueError('Native study locator changed')
    if read(root/'registration.json')['profiles'][plan['profile']]!=plan['settings']:
        raise ValueError('Diagnostic changed the policy')
    class SparseDiagnostic(AllocationRecoveryPlugin):
        def __init__(self):
            # The registered supplemental diagnostic chooses its sole policy;
            # native evaluation and ALL rejection gates are inherited unchanged.
            super().__init__(root,mode='screen',workers=args.workers)
            self.mode='diagnose';self.names=[plan['profile']]
            self.capitals=tuple(plan['capitals'])
        def build_end_of_round_artifacts(self,state):
            result=super().build_end_of_round_artifacts(state)
            verdict=read(root/'diagnose_verdict.json')
            if verdict['eligible'] or verdict['choice']!={'recipe':'median'}:
                raise ValueError('Sparse diagnostic unexpectedly attempted promotion')
            return result
    PhaseRunner(SparseDiagnostic(),root/'phased_diagnose',round_name='Sparse recycling capital diagnostic',
        max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()


if __name__=='__main__':
    main()

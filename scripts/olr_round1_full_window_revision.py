"""Register revision 9: preserve IS, replace June-only OOS with continuous April-July.

This is an immutable migration, not a new candidate search or source-data repair.
The supervisor performs the full native baseline audit and reconstruction bridges
before dispatching the retained granular and mechanism experiments.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import olr_round1_comprehensive_study as native
from scripts import olr_round1_mechanism_study as mechanism

OLD = native.DEFAULT_ROOT
ROOT = OLD/'full_oos_20260401_20260710'
FULL = Path.home()/'.codex/research/olr_round1_oos_20261009'


def register():
    if (ROOT/'protocol.json').exists():
        raise ValueError('Revision is already registered; resume prepare/run instead')
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ('candidate_plan.json', 'historical_leaf_coverage_audit.json',
                 'lineage_attribution.json', 'adaptive_review_requirements.json', 'superseded_run.json'):
        shutil.copy2(OLD/name, ROOT/name)
    for name in ('mechanism_diagnostics.json', 'mechanism_followup_plan.json',
                 'matched_length_is_windows.json', 'preliminary_mechanism_analysis.txt',
                 'revised_study_plan.txt'):
        destination = ROOT/'june_july_diagnostic'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OLD/name, destination)
    for name in ('lineage', 'revisions/revision_8_supported_exit_contract/failed_intrabar_stop_oos',
                 'builder_bridge/train'):
        shutil.copytree(OLD/name, ROOT/name)
    (ROOT/'adaptive_plans').mkdir(exist_ok=True)
    shutil.copy2(OLD/'adaptive_plans/early_interactions.json', ROOT/'adaptive_plans/early_interactions.json')
    old_generation = native.read(OLD/'generations_v2/oos/ready.json')
    native.write(ROOT/'common_python_study.json', old_generation['study'])
    protocol = copy.deepcopy(native.read(OLD/'protocol.json'))
    protocol.update(harness_revision=9, harness_sha256=native.sha(native.__file__),
        orchestration_sha256=native.sha(mechanism.__file__),
        windows={'train':['2025-08-07','2026-03-31'],'oos':['2026-04-01','2026-07-10']},
        generation_directories={'train':str(OLD/'generations_v4'),'oos':'generations_v9'},
        reusable_generation_directories={'oos':'no_reusable_oos_generation'},
        common_python_study={'path':str(ROOT/'common_python_study.json'),
                             'sha256':native.sha(ROOT/'common_python_study.json')},
        oos_scope='Continuous April 1-July 10 account, consumed development after adaptive search. No independent validation claim.',
        other_history='June-only trials are retained in the previous root and labelled diagnostic. They are never reused as full-window OOS results.',
        refresh_baseline_mechanism_diagnostics=True,
        revision={'at_utc':mechanism.utc(),'reason':'User corrected OOS to April 1-July 10; IS unchanged. One account path, no segment stitching.',
                  'previous_root':str(OLD),'unchanged_is_results_reused':True},
        analysis_requirements={'required_review':'Refresh all period-level diagnostics on the full 69-session OOS. Review old June-only mechanisms as hypotheses, not full-window conclusions.',
            'historical_leaf_coverage_audit':'historical_leaf_coverage_audit.json',
            'lineage_attribution':'lineage_attribution.json','mechanism_diagnostics':'mechanism_diagnostics.json',
            'adaptive_complements':'adaptive_review_requirements.json'})
    parent = FULL/'validation/study.json'
    protocol['parent_studies']['oos'] = {'path':str(parent),'sha256':native.sha(parent)}
    native.write(ROOT/'protocol.json', protocol)
    plan = copy.deepcopy(native.read(OLD/'mechanism_plan.json'))
    plan.update(version=9,runner_sha256=native.sha(mechanism.__file__),native_evaluator_sha256=native.sha(native.__file__))
    plan['source_evidence'] = {
        ('june_july_diagnostic/'+name if name in ('mechanism_diagnostics.json','mechanism_followup_plan.json') else name):digest
        for name,digest in plan['source_evidence'].items()}
    plan['window_correction'] = {'windows':protocol['windows'], 'all_original_definitions_retained':True,
        'previous_diagnostics_scope':'June 1-July 10 only; recompute full-window mechanisms after audited baseline.'}
    native.write(ROOT/'mechanism_plan.json',plan)
    protocol['mechanism_plan_sha256'] = native.sha(ROOT/'mechanism_plan.json')
    native.write(ROOT/'protocol.json',protocol)
    reused=[]
    for original in sorted((OLD/'cases').glob('*/train/100000000_baseline')):
        if not (original/'completion.json').exists():
            continue
        result=native.read(original/'result.json')
        audit=native.read(original/'execution_audit.json')
        if not (result.get('audit_status')=='passed' and audit.get('audit_pass')):
            continue
        cid=original.parent.parent.name
        registration=OLD/'registrations/train'/(cid+'_100000000')
        native.verify_window_contract(ROOT,'train',native.read(registration/'study.json'),original)
        shutil.copytree(original,ROOT/'cases'/cid/'train/100000000_baseline')
        shutil.copytree(registration,ROOT/'registrations/train'/registration.name)
        # Assessments carry an absolute output directory; rewrite only the copied wrapper.
        output=ROOT/'cases'/cid/'train/100000000_baseline'
        if (output/'assessment.json').exists():
            assessment=native.read(output/'assessment.json')
            assessment['output']=str(output)
            native.write(output/'assessment.json',assessment)
        candidate=native.read(registration/'candidate.json')
        assessment=native.evaluate(ROOT,candidate,'train')
        mechanism.validate_cached(ROOT,candidate,'train')
        reused.append({'id':cid,'original_output':str(original),'output':str(output),
                       'return_pct':assessment['return_pct'],'trades':assessment['trades']})
    receipt={'revision':9,'created_utc':mechanism.utc(),'root':str(ROOT),'windows':protocol['windows'],
             'previous_root':str(OLD),'reused_is_results':reused,'reused_oos_results':[],
             'full_oos_parent_preflight':native.read(FULL/'replay_preflight.json'),
             'parent_preflight_sha256':native.sha(FULL/'replay_preflight.json'),
             'common_python_note':'Retain previous common audited Python, including inactive stale-release opt-in. Preserve full-window calendar, market observations, feature database, and source acceptance.',
             'full_oos_performance':'pending native audit; parent preflight is not a performance result'}
    native.write(ROOT/'revision9_registration.json',receipt)
    native.write(OLD/'superseded_mechanism.json',receipt)
    native.write(OLD/'active_study.json',{'root':str(ROOT),'revision':9,'windows':protocol['windows']})
    native.event(ROOT,'full_window_revision_registered',reused_is=len(reused),oos_reused=0)


def prepare():
    for window in ('oos','train'):
        info=native.prepare_window(ROOT,window)
        rows=[]
        for candidate in mechanism.catalog(ROOT).values():
            try:
                config=native.candidate_preflight(ROOT,window,candidate)
                unsupported=mechanism.unsupported_exit_orders(config)
                rows.append({'id':candidate['id'],'status':'unsupported_execution_contract' if unsupported else 'supported',
                             'unsupported':unsupported})
            except ValueError as exc:
                rows.append({'id':candidate['id'],'status':'invalid_configuration','error':str(exc)})
        from collections import Counter
        native.write(ROOT/f'revision9_preflight_{window}.json',{'window':window,
            'date_range':info['study']['configuration']['date_range'],'counts':dict(Counter(r['status'] for r in rows)),
            'source_hash':info['study']['source_generation_hash'],'candidates':rows})
    native.event(ROOT,'full_window_preparation_complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['register','prepare'])
    args=parser.parse_args()
    register() if args.command=='register' else prepare()

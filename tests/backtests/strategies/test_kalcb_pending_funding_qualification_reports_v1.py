import importlib.util
from pathlib import Path
from copy import deepcopy
import pytest
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound
from backtests.strategies.kalcb.pending_funding_qualification_full_diagnostics_v1 import (
    authenticated_protocol, truthful_status, STATUS)
from backtests.strategies.kalcb.pending_funding_qualification_report_checks_v1 import check_report, OLD_STATUS
from backtests.strategies.kalcb.pending_funding_qualification_assessment_v1 import completion_gate


def test_exact_four_stress_design_and_reuse_count(tmp_path):
    done=tmp_path/'complete.json';write(done,{'complete':True})
    profiles={n:{p:{s:{} for s in ('baseline','participation','cost','combined')}
        for p in ('is','oos')} for n in ('control','pending_100')}
    p={'version':'pending-funding-qualification-full-reports-v1','programs':[],
        'bindings':[bound(done)],'assessment_completion':bound(done),'native_case_count':16,
        'report_count':16,'fresh_report_count':8,'reused_report_count':8,'profiles':profiles,
        'recipes':{n:{} for n in profiles}}
    write(tmp_path/'protocol.json',p)
    assert authenticated_protocol(tmp_path)['report_count']==16
    p['profiles']['pending_100']['oos'].pop('cost');write(tmp_path/'protocol.json',p)
    with pytest.raises(ValueError,match='Missing'):authenticated_protocol(tmp_path)


def fixture_report(root,reused=False):
    status=OLD_STATUS if reused else STATUS
    activity={'gross_winning_net_pnl_krw':200.,'gross_losing_net_pnl_krw':-100.}
    account={'metrics':{'return_pct':1.,'drawdown_pct':2.},'activity':activity,'support':{'orders':[]},'funding':{}}
    recipe={'prior':100,'reservation':2}
    ctx={'profile':{'primary_profile':'pending_100','period':'oos','recipe':recipe,
        'result_paths':{'cost':str(root/'native/result.json')}},'scenario':'cost',
        'production_enabled':False,'canonical_replaced':False,'profitable_activity':activity,
        'actual_intervention_support':account['support'],'funding':{}}
    full={'promotion_status':status,'diagnostics_summary':{'promotion_status':status,'pending_funding_context':ctx},
        'final_metrics':{'net_return_pct':.01,'max_drawdown_pct':.02,'profit_factor':2.}}
    write(root/'full.json',full);write(root/'integrity.json',{'passed':True,'native_evidence_unchanged':True})
    write(root/'status.json',{'payload':{'promotion_status':status}})
    marker='SCREENING' if reused else 'FOUR-STRESS EVIDENCE'
    (root/'report.txt').write_text('AUDITED OPENING PENDING-FUNDING '+marker+' '+status,encoding='utf-8')
    refresh_receipt(root)
    return account,recipe


def refresh_receipt(root):
    write(root/'completion.json',{'complete':True,'economic_promotion':False,
        **{k:bound(root/v) for k,v in {'report':'report.txt','full_diagnostics':'full.json',
            'integrity':'integrity.json','framework_status':'status.json'}.items()}})


@pytest.mark.parametrize('location',['top','summary','framework'])
def test_inherited_acceptance_cannot_survive_any_report_surface(tmp_path,location):
    account,recipe=fixture_report(tmp_path)
    if location=='framework':
        x=read(tmp_path/'status.json');x['payload']['promotion_status']='accepted';write(tmp_path/'status.json',x)
    else:
        x=read(tmp_path/'full.json')
        (x if location=='top' else x['diagnostics_summary'])['promotion_status']='accepted';write(tmp_path/'full.json',x)
    refresh_receipt(tmp_path)
    with pytest.raises(ValueError,match='status'):
        check_report(tmp_path/'completion.json','pending_100/oos/cost',account,recipe,tmp_path/'native')


@pytest.mark.parametrize('field',['profit_factor','net_return_pct','max_drawdown_pct'])
def test_exact_metrics_are_checked_even_with_rebound_artifacts(tmp_path,field):
    account,recipe=fixture_report(tmp_path)
    x=read(tmp_path/'full.json');x['final_metrics'][field]+=1.;write(tmp_path/'full.json',x);refresh_receipt(tmp_path)
    with pytest.raises(ValueError,match='drift'):
        check_report(tmp_path/'completion.json','pending_100/oos/cost',account,recipe,tmp_path/'native')


def test_reused_screening_report_keeps_original_scope(tmp_path):
    account,recipe=fixture_report(tmp_path,True)
    check=check_report(tmp_path/'completion.json','pending_100/oos/cost',account,recipe,tmp_path/'native',True)
    assert check['reused_v25_report'] and 'Original V25' in check['scope']
    with pytest.raises(ValueError,match='status'):
        check_report(tmp_path/'completion.json','pending_100/oos/cost',account,recipe,tmp_path/'native',False)


def test_reuse_cannot_point_at_another_native_case(tmp_path):
    account,recipe=fixture_report(tmp_path)
    with pytest.raises(ValueError,match='source'):
        check_report(tmp_path/'completion.json','pending_100/oos/cost',account,recipe,tmp_path/'other')


def test_report_hash_tampering_is_rejected(tmp_path):
    account,recipe=fixture_report(tmp_path)
    (tmp_path/'report.txt').write_text('modified',encoding='utf-8')
    with pytest.raises(ValueError,match='Changed binding'):
        check_report(tmp_path/'completion.json','pending_100/oos/cost',account,recipe,tmp_path/'native')


def test_both_analysis_statuses_override_generic_acceptance():
    a={'promotion_status':'accepted','diagnostics_summary':{'promotion_status':'accepted'}}
    assert truthful_status(a)['promotion_status']==a['diagnostics_summary']['promotion_status']==STATUS


def test_assessment_requires_unique_complete_audited_cells(tmp_path):
    write(tmp_path/'supervisor_start.json',{})
    write(tmp_path/'status.json',{'stage':'awaiting_joint_assessment','child':None})
    write(tmp_path/'stage_completion.json',{'complete':True,'fresh_cases':8,'authenticated_reuse':8,'accounted_cases':16})
    rows=[{'name':n,'period':p,'scenario':s,'reused':s in ('baseline','combined'),
        'metrics':{'audit_status':'passed'},'identity_passed':True}
        for n in ('control','pending_100') for p in ('is','oos') for s in ('baseline','participation','cost','combined')]
    write(tmp_path/'progress.json',rows)
    assert len(completion_gate(tmp_path,alive=lambda _:False))==16
    rows[-1]=deepcopy(rows[0]);write(tmp_path/'progress.json',rows)
    with pytest.raises(ValueError,match='duplicate'):completion_gate(tmp_path,alive=lambda _:False)


def waiter():
    spec=importlib.util.spec_from_file_location('qualification_wait',Path(__file__).resolve().parents[3]/
        'scripts/kalcb_pending_funding_qualification_analysis_wait_v1.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_live_native_never_registers_assessment(tmp_path):
    write(tmp_path/'status.json',{'stage':'native_case'})
    assert waiter().ready(tmp_path,{},check_alive=lambda _:True) is False


def test_failed_native_never_automatically_restarts(tmp_path):
    write(tmp_path/'status.json',{'stage':'attention_required'})
    with pytest.raises(ValueError,match='failed'):waiter().ready(tmp_path,{},check_alive=lambda _:False)


def test_exit_without_complete_native_results_cannot_run_analysis(tmp_path):
    write(tmp_path/'status.json',{'stage':'native_case','child':None});write(tmp_path/'supervisor_start.json',{})
    with pytest.raises(ValueError,match='complete'):waiter().ready(tmp_path,{},check_alive=lambda _:False)

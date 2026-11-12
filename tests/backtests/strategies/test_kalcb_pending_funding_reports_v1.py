from copy import deepcopy
import json
import pytest
from backtests.strategies.kalcb.pending_funding_full_diagnostics_v1 import truthful_status,STATUS,authenticated_protocol


def test_both_report_statuses_override_inherited_acceptance():
    a={'promotion_status':'accepted','diagnostics_summary':{'promotion_status':'accepted'}}
    truthful_status(a)
    assert a['promotion_status']==a['diagnostics_summary']['promotion_status']==STATUS
    assert 'no_promotion' in STATUS


def test_protocol_requires_exact_sixteen_case_design(tmp_path):
    done=tmp_path/'completion.json';done.write_text(json.dumps({'complete':True}),encoding='utf-8')
    from backtests.strategies.kalcb.retention_research_io_v1 import bound
    profiles={n:{p:{s:{} for s in ('baseline','combined')} for p in ('is','oos')}
        for n in ('control','control_prior125','pending_100','pending_125')}
    protocol={'version':'pending-funding-full-reports-v1','programs':[],'bindings':[bound(done)],
        'assessment_completion':bound(done),'report_count':16,'native_case_count':16,'profiles':profiles,
        'recipes':{n:{} for n in profiles}}
    target=tmp_path/'protocol.json';target.write_text(json.dumps(protocol),encoding='utf-8')
    assert authenticated_protocol(tmp_path)['report_count']==16
    protocol['profiles']['pending_100']['oos'].pop('combined');target.write_text(json.dumps(protocol),encoding='utf-8')
    with pytest.raises(ValueError,match='Missing'):authenticated_protocol(tmp_path)


def test_report_context_has_no_old_exit_or_fixed_benchmark_claims():
    import inspect
    from backtests.strategies.kalcb.pending_funding_full_diagnostics_v1 import PendingFundingReportPlugin
    text=inspect.getsource(PendingFundingReportPlugin)
    for incorrect in ('Opening-thesis exit','retained_screening_assessment','audited_exit_qualification_context','Fixed benchmark preserves'):
        assert incorrect not in text
    assert 'No independent validation' in text
    assert 'Further economic experiments require recommendation' in text

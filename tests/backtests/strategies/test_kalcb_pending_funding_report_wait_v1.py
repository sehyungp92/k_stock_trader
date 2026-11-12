import importlib.util,json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('pending_wait',Path(__file__).resolve().parents[3]/'scripts/kalcb_pending_funding_report_wait_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_ready_never_runs_analysis_while_native_alive(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'stage':'native_case'}),encoding='utf-8')
    assert m.ready(tmp_path,{'pid':1},lambda _:True) is False


def test_failed_native_dependency_is_not_restarted(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'stage':'attention_required'}),encoding='utf-8')
    with pytest.raises(ValueError,match='failed'):m.ready(tmp_path,{'pid':1},lambda _:False)


def test_exit_without_completion_cannot_start_reports(tmp_path):
    for n,data in [('status.json',{'stage':'native_case','child':None}),('supervisor_start.json',{'pid':1})]:
        (tmp_path/n).write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError,match='incomplete'):m.ready(tmp_path,{'pid':1},lambda _:False)

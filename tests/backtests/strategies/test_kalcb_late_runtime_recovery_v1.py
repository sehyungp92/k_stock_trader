import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
import pytest
from backtests.strategies.kalcb.late_runtime_recovery_v1 import calendar_extension


def calendars(tmp_path):
    roots=[tmp_path/'old',tmp_path/'new']
    for i,root in enumerate(roots):
        d=root/'kis_core/data';d.mkdir(parents=True)
        schedule={'coverage_start':'2025-05-12','coverage_end':'2026-10-08' if i else '2026-05-12',
          'regular':{'open':'09:00','close':'15:30'},'markets':['KOSPI'], 'exceptions':{},'symbol_suspensions':[], 'sources':{'regular':'retained source'}}
        (d/'krx_sessions.yaml').write_text(yaml.safe_dump(schedule))
        (d/'krx_holidays.yaml').write_text('same holiday bytes')
        regimes={'halts':[], 'evidence_files':{},'markets':{'KOSPI':['000100']}}
        if i:regimes['halts'].append({'date':'2026-06-08','market':'KOSPI'})
        (d/'krx_intraday_halts.yaml').write_text(yaml.safe_dump(regimes))
    return roots


def test_full_horizon_extension_preserves_past_rules(tmp_path):
    old,new=calendars(tmp_path)
    assert calendar_extension(old,new,'2025-05-12','2026-07-10')['new_halts']==1


@pytest.mark.parametrize('change',['short_coverage','past_hours','past_halt','holidays'])
def test_bad_calendar_successor_fails_closed(tmp_path,change):
    old,new=calendars(tmp_path);d=new/'kis_core/data'
    if change=='holidays':(d/'krx_holidays.yaml').write_text('altered')
    elif change=='past_halt':
        p=d/'krx_intraday_halts.yaml';value=yaml.safe_load(p.read_text());value['halts'].append({'date':'2025-06-01'});p.write_text(yaml.safe_dump(value))
    else:
        p=d/'krx_sessions.yaml';value=yaml.safe_load(p.read_text())
        if change=='short_coverage':value['coverage_end']='2026-05-12'
        else:value['regular']['open']='09:30'
        p.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError):calendar_extension(old,new,'2025-05-12','2026-07-10')


def test_actual_oos_frozen_runtime_covers_pre_is_to_final_oos():
    root=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011/v5')
    profile=json.loads((root/'cases/identity_100/oos/input_profile.json').read_text())
    code=Path(profile['folder'])/'inputs/files'
    script="from datetime import date; from strategy_common.source_quality import expected_labels; print([len(expected_labels('000100',date.fromisoformat(d))) for d in ('2025-05-12','2026-05-13','2026-06-08','2026-07-10')])"
    proc=subprocess.run([sys.executable,'-B','-c',script],cwd=code,env={**os.environ,'PYTHONPATH':str(code),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    assert all(v>0 for v in json.loads(proc.stdout.strip().splitlines()[-1]))


def test_successor_uses_registered_runtime_and_shared_lifecycle():
    text=Path('scripts/kalcb_late_entry_study_v6.py').read_text();tree=ast.parse(text)
    assert "code=Path(protocol['producer_runtime']['folder'])/'inputs/files'" in text
    assert 'PhaseRunner(' in text and 'authenticate_predecessor' in text
    assert not any(isinstance(n,ast.ImportFrom) and n.module and 'kalcb_late_entry_study' in n.module for n in ast.walk(tree))

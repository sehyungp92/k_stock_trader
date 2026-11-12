from dataclasses import replace
from datetime import datetime,timedelta
from types import SimpleNamespace
import json
import sqlite3
from pathlib import Path
import pytest
from backtests.strategies.kalcb.late_entry_v1 import completed_confirmation,flat_blocked,continuous_connection,cumulative_timed_policy,append_snapshot
from backtests.strategies.kalcb.late_entry_io_v1 import make_overrides,exactly
from backtests.strategies.kalcb.retention_research_io_v1 import bound
from scripts.kalcb_late_entry_study_v1 import recipes


def bars():
    start=datetime.fromisoformat('2026-04-01T09:00:00+09:00')
    return [SimpleNamespace(timestamp=start+timedelta(minutes=5*i),open=100+i,high=102+i,low=99+i,close=101+i,volume=1000) for i in range(12)]


def test_future_perturbations_do_not_change_confirmation():
    b=bars();at=b[0].timestamp+timedelta(minutes=60)
    expected=completed_confirmation(b,at)
    later=SimpleNamespace(timestamp=at,open=1e9,high=1e12,low=1,close=1,volume=1e14)
    assert completed_confirmation(b+[later],at)==expected


@pytest.mark.parametrize('index',[0,6,10,11])
def test_missing_completed_prefix_rejected(index):
    b=bars();at=b[0].timestamp+timedelta(minutes=60);b.pop(index)
    with pytest.raises(ValueError,match='Incomplete'):completed_confirmation(b,at)


def test_own_contemporaneous_vwap_not_final_vwap():
    b=bars();b[11].close=1000;b[11].high=1000;b[11].volume=1000000
    assert completed_confirmation(b,b[0].timestamp+timedelta(minutes=60))[1]


@pytest.mark.parametrize('kind',['position','open_count','pending','core_position'])
def test_flat_guard_checks_actual_funding_state(kind):
    p=SimpleNamespace(open_positions=int(kind=='open_count'),positions={'X':1} if kind=='position' else {})
    s=SimpleNamespace(symbols={'X':SimpleNamespace(pending_entry_order_id='p' if kind=='pending' else '',position=object() if kind=='core_position' else None)})
    assert flat_blocked(s,p)


def test_flat_guard_allows_flat_account():
    assert not flat_blocked(SimpleNamespace(symbols={}),SimpleNamespace(open_positions=0,positions={}))


def test_continuous_union_retains_history_and_one_boundary(tmp_path):
    files=[]
    for n,days in [('old',['2025-05-12','2026-03-31']),('new',['2026-03-31','2026-04-01'])]:
        p=tmp_path/(n+'.sqlite');c=sqlite3.connect(p);c.execute('CREATE TABLE bars(day,symbol,ts,o,h,l,c,v)')
        for d in days:c.execute('INSERT INTO bars VALUES(?,?,?,?,?,?,?,?)',(d,'X',d+'T09:00:00+09:00',1,2,1,2,10))
        c.commit();c.close();files.append(str(p))
    c=continuous_connection(files)
    assert [r[0] for r in c.execute('SELECT day FROM bars ORDER BY day')]==['2025-05-12','2026-03-31','2026-04-01']
    with pytest.raises(sqlite3.OperationalError):c.execute("INSERT INTO history.bars VALUES('x','x','x',1,1,1,1,1)")
    c.close()


def test_finite_design_has_two_parents_and_backouts():
    r=recipes();assert len(r)==8
    for q in (100,125):
        assert {(s['quantity'],s['flat_only']) for s in r.values() if s['prior']==q}=={(0,False),(.25,False),(.25,True),(.5,True)}


def test_patch_never_silently_matches_wrong_source():
    with pytest.raises(ValueError):exactly('aa','a','b')


def test_actual_parent_patch_is_syntax_valid_and_scoped(tmp_path):
    b=Path('E:/CodexResearch/k_stock_trader/kalcb_selective_recovery_20261010')
    p=json.loads((b/'v8_population_reports/artifacts/profiles/expanding_removal_prior100/is/research_profile.json').read_text())
    code=Path(p['execution_source']['code']);before={f:bound(code/f) for f in ['strategy_kalcb/core/logic.py','scripts/kalcb_executed_value_replay.py','backtests/engine/qualification_evidence.py']}
    out=make_overrides(code,tmp_path/'overrides',{})
    assert len(out)==3
    for path in out.values():compile(Path(path['path']).read_text(),path['path'],'exec')
    assert before=={f:bound(code/f) for f in before}
    scope=Path(out['backtests/engine/qualification_evidence.py']['path']).read_text()
    assert "expected=('10:00','15:20')" in scope
    logic=Path(out['strategy_kalcb/core/logic.py']['path']).read_text()
    assert 'later_entry_requires_flat_account' in logic
    assert 'from backtests.strategies.kalcb.late_entry_v1 import flat_blocked' not in logic


def test_existing_parent_timed_policy_exact_bridge():
    from scripts.kalcb_executed_value_replay import verify_policy,timed_policy
    b=Path('E:/CodexResearch/k_stock_trader/kalcb_selective_recovery_20261010')
    p=json.loads((b/'v8_population_reports/artifacts/profiles/expanding_removal_prior100/is/research_profile.json').read_text())
    directory=Path(p['policy']['directory']);m,s=verify_policy(directory,'interaction_cost_margin')
    old=timed_policy(directory,m,s);new=cumulative_timed_policy(directory,m,s)
    assert {str(k):[v.to_json_dict() for v in vv] for k,vv in old.items()}=={str(k):[v.to_json_dict() for v in vv] for k,vv in new.items()}


def test_maturity_embargo_excludes_future_outcome():
    from backtests.strategies.kalcb.action_value import training_indices,ModelSpec
    sessions=['2026-03-26','2026-03-27','2026-03-30','2026-03-31','2026-04-01']
    rows=[{'trade_date':d,'opportunity_id':d,'action':{'ready':True}} for d in sessions]
    labels={d:{'label_valid':True,'label_available_at':d+'T15:35:00+09:00'} for d in sessions}
    labels['2026-03-27']['label_available_at']='2026-04-02T10:00:00+09:00'
    indexes,_=training_indices(rows,labels,sessions,'2026-04-01',ModelSpec('test'),min_sessions=1)
    assert indexes==[0,2]


def test_two_checkpoints_lock_previous_candidates_and_validate_new_clock(tmp_path):
    from scripts.kalcb_executed_value_replay import verify_policy,timed_policy
    b=Path('E:/CodexResearch/k_stock_trader/kalcb_selective_recovery_20261010')
    p=json.loads((b/'v8_population_reports/artifacts/profiles/expanding_removal_prior100/is/research_profile.json').read_text())
    directory=Path(p['policy']['directory']);m,opening=verify_policy(directory,'interaction_cost_margin');updates=timed_policy(directory,m,opening)
    day=next(iter(updates));old=updates[day][0];at=old.generated_at+timedelta(minutes=15)
    c=replace(old.candidates[-1],symbol='999999',metadata={**old.candidates[-1].metadata,'entry_checkpoint_at':at.isoformat(),'feature_available_at':at.isoformat()})
    new=replace(old,generated_at=at,candidates=old.candidates+(c,))
    admissions=[json.loads(line) for line in (directory/m['timed_admission_file']).read_text().splitlines() if json.loads(line)['trade_date']==str(day)]
    admissions.append({'trade_date':str(day),'symbol':c.symbol,'available_at':at.isoformat(),'score':c.selection_score})
    (tmp_path/'admissions').write_text('\n'.join(map(json.dumps,admissions)))
    def write_snap(snap):
        (tmp_path/'snapshots').write_text('\n'.join(json.dumps(s.to_json_dict()) for s in (old,snap)))
    mm={**m,'timed_snapshot_file':'snapshots','timed_admission_file':'admissions','observation_symbols':m['observation_symbols']+['999999'],
        'files':{'snapshots':'test','admissions':'test'},'contract':{'timed_budget':13}}
    write_snap(new);assert len(cumulative_timed_policy(tmp_path,mm,{day:opening[day]})[day])==2
    altered=replace(old.candidates[0],selection_score=old.candidates[0].selection_score+1)
    write_snap(replace(new,candidates=(altered,)+new.candidates[1:]))
    with pytest.raises(ValueError,match='Changed incumbent'):cumulative_timed_policy(tmp_path,mm,{day:opening[day]})
    write_snap(replace(new,candidates=old.candidates+(replace(c,metadata={**c.metadata,'feature_available_at':(at+timedelta(minutes=5)).isoformat()}),)))
    with pytest.raises(ValueError,match='Future feature'):cumulative_timed_policy(tmp_path,mm,{day:opening[day]})

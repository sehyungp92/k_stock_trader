from copy import deepcopy
from datetime import date,timedelta
from pathlib import Path
import ast
import importlib.util
import json
import math
import numpy as np
import pytest
from backtests.strategies.kalcb.late_incremental_v1 import (
    aligned_fit, population_map, reserve_fit, admitted, COMPACT_COLUMNS, RULE)
from backtests.strategies.kalcb.late_signal_v1 import affine_predict


def fixture(n=100):
    days=[str(date(2025,1,1)+timedelta(days=i)) for i in range(n)]
    records=[];labels={};rows={};eligible={}
    for i,day in enumerate(days):
        for j in range(3):
            key=day+f'/S{j}/minute60/q0.25';x=float(10+(i+j)%7)
            row=dict(opportunity_id=key,trade_date=day,symbol=f'S{j}',
                feature_available_at=day+'T10:00:00+09:00',action={'ready':True},
                features={k:float((i+j+z)%9) for z,k in enumerate(COMPACT_COLUMNS)})
            row['features']['action_notional_equity']=.05+ .01*j
            records.append(dict(opportunity_id=key,valid_from=day,feature_available_at=row['feature_available_at'],
                baseline=x,combined=x-1,capacity_pass=True,failure_probability=.0001,calibration_supported=True))
            labels[key]=dict(label_valid=True,label_available_at=day+'T15:35:00+09:00',
                net_value_equity_bps=(x-5)*row['features']['action_notional_equity'])
            rows[key]=row;eligible[key]=True
    return days,records,labels,rows,eligible


def test_correct_units_and_prior():
    d,r,l,rows,e=fixture()
    proof=population_map(r,l,d,d[80],'baseline',e,rows)
    assert proof['issuer_days']==237
    assert proof['slope']==pytest.approx(1)
    assert affine_predict(10,proof)==pytest.approx(10-5*237/337)
    reserve=reserve_fit(r,l,d,d[80],'baseline',e,rows)
    assert reserve['reserve_bps']==pytest.approx(5)
    assert reserve['sessions']==79


def test_future_embargo_and_immature_poison_ignored():
    d,r,l,rows,e=fixture();first=d[80];immature=r[8]['opportunity_id']
    l[immature]['label_available_at']=first+'T10:00:00+09:00'
    before=[fn(r,l,d,first,'baseline',e,rows) for fn in (population_map,reserve_fit)]
    for record in r:
        if record['opportunity_id'][:10]>=d[79] or record['opportunity_id']==immature:
            l[record['opportunity_id']]['net_value_equity_bps']=float('nan')
    after=[fn(r,l,d,first,'baseline',e,rows) for fn in (population_map,reserve_fit)]
    assert before==after
    assert before[0]['observed_through']==d[78]


def test_unknown_and_incumbent_outcomes_cannot_calibrate():
    d,r,l,rows,e=fixture()
    for i,record in enumerate(r):
        key=record['opportunity_id']
        if i<180:
            if i%2:e.pop(key)
            else:e[key]=False
            l[key]['net_value_equity_bps']=float('nan')
    assert population_map(r,l,d,d[80],'baseline',e,rows)['issuer_days']==57
    assert reserve_fit(r,l,d,d[80],'baseline',e,rows)['issuer_days']==57


def test_positive_tail_and_gates_not_future_winners():
    d,r,l,rows,e=fixture()
    for i,record in enumerate(r):
        if i<210:
            record['combined']=-1
            l[record['opportunity_id']]['net_value_equity_bps']=float('nan')
    result=reserve_fit(r,l,d,d[80],'baseline',e,rows)
    assert result['issuer_days']==27 and result['sessions']==9
    for record in r[210:216]:record['failure_probability']=.001
    assert not reserve_fit(r,l,d,d[80],'baseline',e,rows)['supported']


def test_cluster_reserve_does_not_count_issuer_days_as_independent_sessions():
    from scipy.stats import t
    d,r,l,rows,e=fixture()
    for record in r:
        key=record['opportunity_id'];i=d.index(key[:10]);error=float(i%5)
        l[key]['net_value_equity_bps']=(record['baseline']-error)*rows[key]['features']['action_notional_equity']
    result=reserve_fit(r,l,d,d[80],'baseline',e,rows)
    errors=np.array([i%5 for i in range(79)])
    assert result['reserve_bps']==pytest.approx(errors.mean()+t.ppf(.95,78)*errors.std(ddof=1)/math.sqrt(79))


@pytest.mark.parametrize('bad',['duplicate','forward','quantity','clock','nonfinite','denominator'])
def test_invalid_training_evidence_rejected(bad):
    d,r,l,rows,e=fixture();key=r[3]['opportunity_id']
    if bad=='duplicate':r.append(deepcopy(r[3]))
    elif bad=='forward':r[3]['valid_from']=d[10]
    elif bad=='quantity':r[3]['opportunity_id']=key.replace('q0.25','q0.5')
    elif bad=='clock':l[key]['label_available_at']=d[1]+'T08:00:00+09:00'
    elif bad=='nonfinite':l[key]['net_value_equity_bps']=float('nan')
    else:rows[key]['features']['action_notional_equity']=0
    with pytest.raises(ValueError):reserve_fit(r,l,d,d[80],'baseline',e,rows)


def test_notional_model_original_clock_and_maturity():
    d,r,l,rows,e=fixture();data=list(rows.values())
    with pytest.raises(ValueError,match='Insufficient mature'):aligned_fit(data,l,d,d[60],d[69])
    before=aligned_fit(data,l,d,d[80],d[89]);bad=deepcopy(l)
    for record in r:
        if record['opportunity_id'][:10]>=d[79]:bad[record['opportunity_id']]['net_value_equity_bps']=1e100
    after=aligned_fit(data,bad,d,d[80],d[89])
    assert before[2]==after[2]
    np.testing.assert_array_equal([v for _,v in before[1]],[v for _,v in after[1]])
    assert before[2]['train_through']==d[78] and len(before[0].ridge.coef_)==16


def test_admission_requires_both_supported_positive_net_above_reserve():
    score=dict(calibration_supported=True,population_eligible=True,failure_probability=.0009,capacity_pass=True,
        notional_forecast={'baseline':6.,'combined':5.},reserve_bps={'baseline':5.,'combined':4.})
    assert admitted(score)
    for field,value in [('calibration_supported',False),('population_eligible',False),('capacity_pass',False),('failure_probability',.001)]:
        assert not admitted({**score,field:value})
    assert not admitted({**score,'reserve_bps':{'baseline':6.,'combined':4.}})
    assert not admitted({**score,'calibration_supported':False,'reserve_bps':{'baseline':None,'combined':None}})


def test_full_account_validation_body_is_identical():
    from backtests.strategies.kalcb import late_artifact_validation_v2 as old, late_incremental_validation_v1 as new
    def function(module,name):
        return ast.dump(next(n for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name))
    for name in ('validate_scope','validate_case','validate_native'):
        assert function(old,name)==function(new,name)


def test_recipe_grid_and_single_action():
    from scripts.kalcb_late_incremental_study_v1 import recipes
    rr=recipes();assert len(rr)==8
    assert sum(x['quantity']==0 for x in rr.values())==2
    assert all(x['quantity'] in (0,.25) and x['flat_only'] is False and x['config']=={} for x in rr.values())


def test_dependency_does_not_advance_while_exact_native_or_assessment_active(monkeypatch,tmp_path):
    import scripts.kalcb_late_incremental_study_v1 as study
    p={'dependency':{'native_root':str(tmp_path),'assessment_root':str(tmp_path),'native_supervisor':{},'assessment_supervisor':{}}}
    monkeypatch.setattr(study,'check_gate',lambda *a:False)
    assert study.dependency_gate(p) is False
    (tmp_path/'status.json').write_text(json.dumps({'stage':'assessing'}))
    monkeypatch.setattr(study,'check_gate',lambda *a:True)
    monkeypatch.setattr(study,'exact_process',lambda *a:True)
    assert study.dependency_gate(p) is False
    monkeypatch.setattr(study,'exact_process',lambda *a:False)
    with pytest.raises(RuntimeError,match='without completion'):study.dependency_gate(p)


def test_frozen_bootstrap_imports_no_app_before_isolation():
    p=Path('scripts/kalcb_late_incremental_producer_v1.py');tree=ast.parse(p.read_text())
    imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    names=[(n.module if isinstance(n,ast.ImportFrom) else n.names[0].name).split('.')[0] for n in imports]
    assert set(names)<={'argparse','hashlib','importlib','json','pathlib','sys'}
    text=p.read_text();assert text.index('sys.path[:]=')<text.index('spec.loader.exec_module')


def policy_fixture(tmp_path):
    from backtests.strategies.kalcb.retention_research_io_v1 import write,bound,digest
    recipe=dict(prior=100,parent=100,signal='reserve',quantity=.25,flat_only=False,config={})
    write(tmp_path/'registered.json',dict(recipes={'candidate':recipe},incremental_rule=RULE))
    write(tmp_path/'source.json',dict(immutable=True))
    key='2026-04-01/ABC/minute60/q0.25'
    score=dict(opportunity_id=key,prior=100,signal_kind='reserve',valid_from='2026-03-10',
        calibration_supported=True,population_eligible=True,failure_probability=.0001,capacity_pass=True,
        notional_forecast={'baseline':30.,'combined':25.},reserve_bps={'baseline':10.,'combined':8.},
        baseline=3.,combined=2.5)
    name='100_reserve.jsonl'; member='later_model_'+name
    (tmp_path/member).write_text(json.dumps(score)+'\n')
    write(tmp_path/'models.json',dict(complete=True,files={name:digest(tmp_path/member)}))
    source=bound(tmp_path/'source.json')
    write(tmp_path/'later_lineage.json',dict(registration=bound(tmp_path/'registered.json'),
        preparation=source,models=bound(tmp_path/'models.json'),parent_policy=source,label_manifests=[source,source],
        recipe=recipe,parent_snapshots_preserved=True,incremental_rule=RULE,
        action_unchanged='same native completed60-minute quarter action, management, costs and source generation'))
    row=dict(trade_date='2026-04-01',action_id=key,valid_from=score['valid_from'],quantity_scale=.25,
        flat_only=False,score=2.5,notional_forecast=score['notional_forecast'],reserve_bps=score['reserve_bps'])
    (tmp_path/'later60.new.jsonl').write_text(json.dumps(row)+'\n')
    for f in ('later60.snapshots.jsonl','later60.admission.jsonl','chronology.json'):
        (tmp_path/f).write_text('{}')
    policy=dict(version='kalcb-first-hour-incremental-v1',observation_minutes=60,
        timed_snapshot_file='later60.snapshots.jsonl',timed_admission_file='later60.admission.jsonl',
        contract=dict(budget=8,timed_budget=13,new_budget=1,quantity_cap=.5,maximum_failure_probability=.001,
            value_threshold=0,additional_value_hurdle_bps_on_notional=48.,production_promotion=False),
        later_admission_rule=RULE,files={p.name:digest(p) for p in tmp_path.iterdir() if p.is_file()})
    return policy,row


@pytest.mark.parametrize('poison',['reserve','quantity','gate','formula','unsupported','model_hash'])
def test_strict_incremental_contract_rejects_poison_even_rebound(poison,tmp_path):
    from backtests.strategies.kalcb.late_incremental_validation_v1 import validate_first_hour_contract
    from backtests.strategies.kalcb.retention_research_io_v1 import digest
    policy,row=policy_fixture(tmp_path);file=tmp_path/'policies_manifest.json'
    validate_first_hour_contract(file,policy)
    if poison=='reserve':row['reserve_bps']={'baseline':0.,'combined':0.}
    elif poison=='quantity':row['quantity_scale']=.5
    elif poison=='gate':policy['contract']['maximum_failure_probability']=.1
    elif poison=='formula':policy['later_admission_rule']={**RULE,'reserve':'zero'}
    elif poison=='unsupported':
        p=tmp_path/'later_model_100_reserve.jsonl';r=json.loads(p.read_text());r['calibration_supported']=False
        p.write_text(json.dumps(r)+'\n')
    else:policy['files']['later_model_100_reserve.jsonl']='0'*64
    (tmp_path/'later60.new.jsonl').write_text(json.dumps(row)+'\n')
    policy['files']['later60.new.jsonl']=digest(tmp_path/'later60.new.jsonl')
    with pytest.raises((ValueError,AssertionError)):validate_first_hour_contract(file,policy)

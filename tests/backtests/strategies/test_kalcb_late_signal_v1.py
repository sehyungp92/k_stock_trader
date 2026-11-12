from copy import deepcopy
from datetime import date,timedelta
import numpy as np
import pytest

from backtests.strategies.kalcb.late_signal_v1 import affine_fit,affine_predict,compact_fit,COMPACT_COLUMNS


def fixtures(n=100):
    sessions=[str(date(2025,1,1)+timedelta(days=i)) for i in range(n)]
    records=[];labels={}
    for i,day in enumerate(sessions):
        key=day+'/ABC/minute60/q0.25'
        x=float(i%7)
        records.append(dict(opportunity_id=key,valid_from=day,feature_available_at=day+'T10:00:00+09:00',
            baseline=x,combined=x-1,capacity_pass=True,failure_probability=0.0001))
        labels[key]=dict(label_available_at=day+'T15:35:00+09:00',label_valid=True,net_value_equity_bps=x-5)
    return sessions,records,labels


def test_embargo_future_and_immature_poison_does_not_change_affine():
    sessions,records,labels=fixtures(); first=sessions[70]
    labels[records[5]['opportunity_id']]['label_available_at']=first+'T10:00:00+09:00'
    before=affine_fit(records,labels,sessions,first,'baseline')
    poisoned=deepcopy(labels)
    for i in [5,*range(69,100)]:poisoned[records[i]['opportunity_id']]['net_value_equity_bps']=float('nan')
    assert before==affine_fit(records,poisoned,sessions,first,'baseline')
    assert before['observed_through']==sessions[68]
    assert before['issuer_days']==68


def test_identity_and_negative_bias():
    sessions,records,labels=fixtures()
    fit=affine_fit(records,labels,sessions,sessions[80],'baseline')
    assert fit['slope']==pytest.approx(1.)
    assert affine_predict(5.,fit)==pytest.approx(5.-5.*79/179)
    for r in records:labels[r['opportunity_id']]['net_value_equity_bps']=r['baseline']
    fit=affine_fit(records,labels,sessions,sessions[80],'baseline')
    assert affine_predict(9.,fit)==pytest.approx(9.)


def test_slope_cannot_invert_rank_or_amplify_tail():
    sessions,records,labels=fixtures()
    for sign in (-100.,100.):
        for r in records:labels[r['opportunity_id']]['net_value_equity_bps']=sign*r['baseline']
        fit=affine_fit(records,labels,sessions,sessions[80],'baseline')
        assert 0<=fit['slope']<=1
        assert 0<=affine_predict(10,fit)-affine_predict(0,fit)<=10.00001


def test_unsupported_calibration_abstains():
    sessions,records,labels=fixtures()
    fit=affine_fit(records,labels,sessions,sessions[10],'baseline')
    assert not fit['supported']
    with pytest.raises(ValueError,match='abstain'):affine_predict(20,fit)


@pytest.mark.parametrize('bad', ['duplicate','quantity','forward','clock','nonfinite'])
def test_rejects_invalid_calibration(bad):
    sessions,records,labels=fixtures()
    if bad=='duplicate':records.append(deepcopy(records[3]))
    elif bad=='quantity':records[3]['opportunity_id']=records[3]['opportunity_id'].replace('q0.25','q0.5')
    elif bad=='forward':records[3]['valid_from']=sessions[4]
    elif bad=='clock':labels[records[3]['opportunity_id']]['label_available_at']=sessions[3]+'T08:00:00+09:00'
    else:records[3]['baseline']=float('inf')
    with pytest.raises(ValueError):affine_fit(records,labels,sessions,sessions[80],'baseline')


def test_ineligible_history_is_not_calibration_support():
    sessions,records,labels=fixtures()
    for r in records[:60]:r['failure_probability']=.002
    fit=affine_fit(records,labels,sessions,sessions[80],'baseline')
    assert fit['issuer_days']==19 and not fit['supported']


def test_compact_fit_keeps_original_maturity_and_future_invariance():
    sessions,records,labels=fixtures()
    rows=[dict(opportunity_id=r['opportunity_id'],trade_date=r['opportunity_id'][:10],action={'ready':True},
        features={k:float((i+j)%9) for j,k in enumerate(COMPACT_COLUMNS)}) for i,r in enumerate(records)]
    with pytest.raises(ValueError,match='Insufficient mature'):compact_fit(rows,labels,sessions,sessions[60],sessions[69])
    before=compact_fit(rows,labels,sessions,sessions[80],sessions[89])
    poisoned=deepcopy(labels)
    for r in records[79:]:poisoned[r['opportunity_id']]['net_value_equity_bps']=1e100
    after=compact_fit(rows,poisoned,sessions,sessions[80],sessions[89])
    assert before[2]==after[2]
    np.testing.assert_array_equal([v for _,v in before[1]],[v for _,v in after[1]])
    assert before[2]['train_through']==sessions[78]
    assert before[2]['train_rows']==79
    assert len(before[0].ridge.coef_)==16

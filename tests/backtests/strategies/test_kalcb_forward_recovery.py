"""Training and calibration clocks used by the full-window recovery study."""
from copy import deepcopy
from datetime import date,timedelta
import numpy as np
from backtests.strategies.kalcb import action_value as av,continuation_value as cv
from backtests.strategies.kalcb.value_calibration import fit_support_offsets
from scripts.kalcb_quantity_calibration import corrections
from backtests.strategies.kalcb.executed_value import outer_windows

def sample():
    sessions=[str(date(2025,1,1)+timedelta(days=i)) for i in range(170)]
    rows=[];labels={}
    for i,day in enumerate(sessions):
        key=day+'/005930/q1'
        features={k:float(np.sin(i*.13+j*.07)) for j,k in enumerate(cv.COLUMNS)}
        features['action_quantity_scale']=1.
        rows.append({'opportunity_id':key,'base_opportunity_id':day+'/005930','trade_date':day,
            'action':{'ready':True},'features':features})
        labels[key]={'label_valid':True,'label_available_at':day+'T15:35:00+09:00','net_value_equity_bps':float(i%7-3)}
    return sessions,rows,labels

def test_training_excludes_embargo_future_and_immature_labels():
    sessions,rows,labels=sample();first=sessions[150]
    labels[rows[147]['opportunity_id']]['label_available_at']=first+'T00:00:00+09:00'
    labels[rows[146]['opportunity_id']]['label_valid']=False
    indices,_=av.training_indices(rows,labels,sessions,first,av.ModelSpec('interaction'))
    assert max(indices)==148
    assert not {146,147,149,150}&set(indices)
    indices,_=av.training_indices(rows,labels,sessions,first,av.ModelSpec('interaction',window='rolling120'))
    assert min(indices)==29 and max(indices)==148 and len(indices)==118

def test_future_label_poison_does_not_change_fitted_model_or_predictions():
    sessions,rows,labels=sample();first=sessions[150];through=sessions[155]
    changed=deepcopy(labels)
    for r in rows[149:]:changed[r['opportunity_id']]['net_value_equity_bps']=1e12
    a,pa,proofa=cv.fit_forecast(rows,labels,sessions,first,through)
    b,pb,proofb=cv.fit_forecast(rows,changed,sessions,first,through)
    assert proofa['training_target_hash']==proofb['training_target_hash']
    np.testing.assert_array_equal(a.regressor.coef_,b.regressor.coef_)
    np.testing.assert_array_equal([v for _,v in pa],[v for _,v in pb])
    assert proofa['train_through']==sessions[148]

def test_calibration_ignores_future_and_embargoed_outcomes():
    sessions,rows,labels=sample();first=sessions[150]
    observations=[{'opportunity_id':r['opportunity_id'],'trade_date':r['trade_date'],
        'baseline_value_bps':1.,'combined_value_bps':1.,'failure_probability':0.,'capacity_ok':True,'quantity_scale':1.} for r in rows]
    grid={s:deepcopy(labels) for s in ('baseline','combined')};changed=deepcopy(grid)
    for scenario in changed:
        for r in rows[149:]:changed[scenario][r['opportunity_id']]['net_value_equity_bps']=1e12
    for fit in (corrections,fit_support_offsets):
        offsets,proof=fit(observations,grid,sessions,first)
        other,otherproof=fit(observations,changed,sessions,first)
        assert offsets==other and proof==otherproof
        assert proof['observed_through']==sessions[148]
        assert proof['latest_label_available_at']<first+'T00:00:00+09:00'

def test_scoring_split_does_not_restart_the_training_clock():
    sessions=[str(date(2025,1,1)+timedelta(days=i)) for i in range(286)]
    first_oos=sessions[217]
    windows=[(a,b) for a,b in outer_windows(sessions,min_train=60,block=20) if b>=first_oos]
    assert windows[0]==(sessions[201],sessions[220])
    assert windows[1][0]==sessions[221]
    # The first OOS decision inherits the current IS fold; it does not wait
    # for 60 new OOS sessions or restart its schedule at the reporting split.
    assert windows[0][0]<first_oos<=windows[0][1]

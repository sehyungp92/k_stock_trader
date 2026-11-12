from copy import deepcopy
from datetime import date,timedelta
import numpy as np
import pytest
from backtests.strategies.kalcb import campaign_value as ca,continuation_value as cv

def sample():
    start=date(2025,1,1)
    daily=[dict(date=(start+timedelta(days=i)).isoformat(),open=100,high=102,low=98,
        close=100+(i%3)*.1,volume=1000+i) for i in range(90)]
    day=(start+timedelta(days=90)).isoformat()
    row={'symbol':'A','trade_date':day,'feature_available_at':day+'T09:30:00+09:00',
        'features':{'daily_atr_pct':.04},'opening_bars':[
            dict(timestamp=day+f'T09:{i*5:02}:00+09:00',open=102,high=104,low=101,
                close=103,volume=200) for i in range(6)]}
    return row,daily

def test_campaign_ignores_future_daily_and_uncompleted_opening():
    row,daily=sample();expected=ca.features(row,daily)
    assert expected[1]['available']
    daily.append(dict(daily[-1],date=row['trade_date'],close=1e12,high=1e12,volume=1e12))
    row['opening_bars'].append(dict(row['opening_bars'][-1],
        timestamp=row['trade_date']+'T09:30:00+09:00',close=1e12,high=1e12))
    assert ca.features(row,daily)==expected
    assert expected[1]['box_end']<row['trade_date']
    assert set(expected[0])==set(ca.COLUMNS)

def test_legacy_default_state_model_is_numerically_unchanged():
    rng=np.random.default_rng(7);x=rng.normal(size=(200,len(cv.COLUMNS)))
    x[:,cv.COLUMNS.index('action_quantity_scale')]=rng.choice([.25,.5,1],200)
    y=rng.normal(size=200)
    a=cv.StatePayoff().fit(x,y,np.ones(200))
    b=cv.StatePayoff(columns=cv.COLUMNS).fit(x,y,np.ones(200))
    np.testing.assert_array_equal(a.predict(x),b.predict(x))
    # Old saved models do not contain schema attributes.
    del b.columns;del b.features.columns;del b.features.interactions
    np.testing.assert_array_equal(a.predict(x),b.predict(x))

@pytest.mark.parametrize('interactions',[cv.INTERACTIONS,cv.INTERACTIONS+ca.INTERACTIONS])
def test_extended_model_uses_only_training_prefix_and_zero_quantity(interactions):
    from tests.backtests.strategies.test_kalcb_continuation_value import fixture
    from threadpoolctl import threadpool_limits
    sessions,rows,labels=fixture();columns=cv.COLUMNS+ca.COLUMNS
    for i,r in enumerate(rows):r['features'].update({k:float(i%13) for k in ca.COLUMNS})
    with threadpool_limits(limits=1):
        model,pred,proof=cv.fit_forecast(rows,labels,sessions,sessions[70],sessions[79],columns=columns,interactions=interactions)
        for r in rows:
            if r['trade_date']>=sessions[69]:labels[r['opportunity_id']]['net_value_equity_bps']=1e12
            if r['trade_date']>sessions[79]:r['features'].update({k:1e12 for k in ca.COLUMNS})
        _,second,other=cv.fit_forecast(rows,labels,sessions,sessions[70],sessions[79],columns=columns,interactions=interactions)
    np.testing.assert_array_equal([p for _,p in pred],[p for _,p in second])
    assert proof['training_target_hash']==other['training_target_hash']
    x=cv.matrix([r for r,_ in pred],columns);x[:,columns.index('action_quantity_scale')]=0
    np.testing.assert_array_equal(model.predict(x),np.zeros(len(x)))

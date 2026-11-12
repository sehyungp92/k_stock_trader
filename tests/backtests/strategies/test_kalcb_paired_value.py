from copy import deepcopy
from datetime import date,timedelta
import numpy as np
from backtests.strategies.kalcb import paired_value as pv,continuation_value as cv


def panel():
    days=[str(date(2025,1,1)+timedelta(days=i)) for i in range(15)]
    rows=[];labels={}
    for i,d in enumerate(days):
        for q in pv.QUANTITIES:
            key=f'{d}/A/q{q:g}'
            rows.append({'trade_date':d,'base_opportunity_id':d+'/A','opportunity_id':key,'action':{'ready':True},
                'features':{**{k:0. for k in cv.COLUMNS},'action_quantity_scale':q,'first30_ret':i/1000}})
            labels[key]={'label_valid':True,'label_available_at':d+'T16:00:00+09:00','net_value_equity_bps':q*10-q*q*12}
    return days,rows,labels


def test_paired_size_can_prefer_interior_quantity_and_excludes_future_outcomes():
    days,rows,labels=panel()
    _,pred,proof=pv.fit_forecast(rows,labels,days,days[10],days[-1],min_sessions=5)
    assert proof['train_stock_days']==9
    first=[v for _,v in pred[:3]]
    assert first[1]>first[0]>first[2]
    poison=deepcopy(labels)
    for k,v in poison.items():
        if k[:10]>=days[9]:v['net_value_equity_bps']=1e12
    _,again,p2=pv.fit_forecast(rows,poison,days,days[10],days[-1],min_sessions=5)
    assert proof==p2
    np.testing.assert_array_equal([v for _,v in pred],[v for _,v in again])


def test_one_unmature_quantity_excludes_entire_training_pair():
    days,rows,labels=panel()
    labels[days[0]+'/A/q1']['label_available_at']=days[-1]+'T16:00:00+09:00'
    _,_,proof=pv.fit_forecast(rows,labels,days,days[10],days[-1],min_sessions=5)
    assert proof['train_stock_days']==8

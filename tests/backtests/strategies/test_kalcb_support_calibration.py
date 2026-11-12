from datetime import date,timedelta
from copy import deepcopy
from backtests.strategies.kalcb.value_calibration import fit_support_offsets


def fixture():
    days=[(date(2025,1,1)+timedelta(days=i)).isoformat() for i in range(50)]
    rows=[];labels={'baseline':{},'combined':{}}
    for day in days:
        for symbol,forecast,actual in [('A',5,15),('B',-5,-15)]:
            key=day+'/'+symbol+'/q1'
            rows.append(dict(trade_date=day,opportunity_id=key,quantity_scale=1,
                baseline_value_bps=forecast,combined_value_bps=forecast,failure_probability=0,capacity_ok=True))
            for scenario in labels:labels[scenario][key]=dict(label_valid=True,label_available_at=day+'T15:35:00+09:00',net_value_equity_bps=actual)
    return days,rows,labels


def test_opposite_support_offsets_and_future_poison_invariance():
    days,rows,labels=fixture();first=days[40]
    expected,proof=fit_support_offsets(rows,labels,days,first)
    assert expected[('baseline',1,'positive')]==390/139
    assert expected[('combined',1,'nonpositive')]==-390/139
    for r in rows:
        if r['trade_date']>=days[39]:
            for scenario in labels:labels[scenario][r['opportunity_id']]['net_value_equity_bps']=1e12
    assert fit_support_offsets(rows,labels,days,first)==(expected,proof)


def test_maturity_risk_capacity_and_independent_session_support():
    days,rows,labels=fixture()
    for r in rows:
        if r['trade_date']<days[33]:r['capacity_ok']=False
    offsets,proof=fit_support_offsets(rows,labels,days,days[40],minimum_count=1)
    assert not any(offsets.values())  # six sessions do not establish support
    days,rows,labels=fixture()
    for r in rows:
        if '/A/' in r['opportunity_id']:r['failure_probability']=.001
        else:
            for scenario in labels:labels[scenario][r['opportunity_id']]['label_available_at']=days[45]+'T15:35:00+09:00'
    offsets,proof=fit_support_offsets(rows,labels,days,days[40])
    assert not any(offsets.values()) and proof['latest_label_available_at'] is None

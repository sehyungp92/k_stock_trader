from copy import deepcopy
from datetime import datetime,timedelta
import numpy as np
import pytest
from backtests.strategies.kalcb import continuation_value as cv


def fixture():
    rng=np.random.default_rng(7)
    sessions=[(datetime(2025,1,1)+timedelta(days=i)).date().isoformat() for i in range(100)]
    rows=[];labels={}
    for i,day in enumerate(sessions):
        f={k:float(rng.normal()) for k in cv.COLUMNS}
        for q in (.25,.5,1):
            key=day+'/A/q'+str(q)
            rows.append({'trade_date':day,'opportunity_id':key,'base_opportunity_id':day+'/A',
                'action':{'ready':True},'features':{**f,'action_quantity_scale':q}})
            labels[key]={'label_valid':True,'label_available_at':day+'T15:35:00+09:00',
                'net_value_equity_bps':q*((i%9)-4)*10,'buy_qty':10}
    return sessions,rows,labels


@pytest.mark.parametrize('kind',['interaction','mixture'])
def test_models_are_past_only_and_zero_size_has_zero_value(kind):
    from threadpoolctl import threadpool_limits
    sessions,rows,labels=fixture()
    for r in rows:
        if r['trade_date']==sessions[10]:
            labels[r['opportunity_id']]['label_available_at']=sessions[80]+'T15:35:00+09:00'
    with threadpool_limits(limits=1):
        model,preds,proof=cv.fit_forecast(rows,labels,sessions,sessions[70],sessions[79],kind)
        for r in rows:
            if r['trade_date']>=sessions[69] or r['trade_date']==sessions[10]:
                labels[r['opportunity_id']]['net_value_equity_bps']=1e12
            if r['trade_date']>sessions[79]:
                for key in cv.COLUMNS:
                    if key!='action_quantity_scale':r['features'][key]=1e12
        _,poisoned,proof2=cv.fit_forecast(rows,labels,sessions,sessions[70],sessions[79],kind)
    np.testing.assert_array_equal([p for _,p in preds],[p for _,p in poisoned])
    assert proof['training_target_hash']==proof2['training_target_hash']
    x=cv.matrix([r for r,_ in preds]);x[:,cv.COLUMNS.index('action_quantity_scale')]=0
    np.testing.assert_array_equal(model.predict(x),np.zeros(len(x)))
    assert proof['train_stock_days']==68


def test_prefix_and_prior_features_cannot_observe_future_bars():
    bars=[{'timestamp':f'2025-08-07T09:{i*5:02}:00+09:00','open':100+i,'high':102+i,
        'low':99+i,'close':101+i,'volume':1000+i} for i in range(6)]
    expected=cv.opening_path(bars,'2025-08-07T09:30:00+09:00')
    bars.append({**bars[-1],'timestamp':'2025-08-07T09:30:00+09:00','close':1e9,'volume':1e12})
    assert cv.opening_path(bars,'2025-08-07T09:30:00+09:00')==expected
    prior=[{**bars[0],'timestamp':'2025-08-06T14:00:00+09:00'},
           {**bars[1],'timestamp':'2025-08-06T15:30:00+09:00'}]
    assert cv.prior_path(prior+bars,'2025-08-07')==cv.prior_path(prior,'2025-08-07')
    assert cv.prior_path(bars,'2025-08-07')['prior_path_available']==0


def test_state_basis_is_fixed_after_training():
    _,rows,_=fixture();x=cv.matrix(rows[:120]);transform=cv.StateMap().fit(x)
    expected=transform.transform(x[:2]);changed=x[:2].copy();changed[1,:]=1e12
    np.testing.assert_array_equal(expected[0],transform.transform(changed)[0])


@pytest.mark.parametrize('grid',[False,True])
def test_compact_loader_authenticates_even_discarded_evidence(tmp_path,grid):
    import json
    from backtests.strategies.kalcb import executed_value as ev
    from scripts.kalcb_continuation_study import compact_labels
    request={'source_generation_hash':'source'}
    ev.write_json(tmp_path/'request.json',request)
    ev.write_json(tmp_path/'completion.json',{'complete':True,**request})
    row={'opportunity_id':'2025-01-01/A','label_valid':True,
         'label_available_at':'2025-01-01T15:35:00+09:00',
         'net_value_equity_bps':3.5,'discarded_diagnostics':{'reason':7}}
    path=tmp_path/'outcomes.jsonl'
    path.write_text(json.dumps(row)+'\n')
    files={'outcomes.jsonl':ev.digest(path)}
    if grid:
        (tmp_path/'events.gz').write_bytes(b'opaque authenticated evidence')
        files['events.gz']=ev.digest(tmp_path/'events.gz')
    ev.write_json(tmp_path/('grid_manifest.json' if grid else 'labels_manifest.json'),{
        'prepared_sha256':'prepared','scenario':'baseline',
        'request_sha256':ev.digest(tmp_path/'request.json'),
        'files' if grid else 'label_files':files})
    labels,_=compact_labels([tmp_path],'prepared',grid=grid,wanted={row['opportunity_id']})
    assert labels['baseline'][row['opportunity_id']]=={k:row[k] for k in (
        'label_valid','label_available_at','net_value_equity_bps')}
    # The compact view must not allow alterations to omitted diagnostic fields.
    row['discarded_diagnostics']['reason']=8
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='Changed label evidence'):
        compact_labels([tmp_path],'prepared',grid=grid)


def test_blend_rejects_incompatible_prediction_clocks_and_risk_models():
    from scripts.kalcb_continuation_blend import combine
    a={'opportunity_id':'A','valid_from':'2025-08-07','failure_probability':.0001,
       'baseline_value_bps':20.,'combined_value_bps':5.}
    b={**a,'baseline_value_bps':10.,'combined_value_bps':-7.}
    assert combine(a,b)['baseline_value_bps']==15.
    assert combine(a,b)['combined_value_bps']==-1.
    with pytest.raises(ValueError,match='clock'):
        combine(a,{**b,'valid_from':'2025-09-07'})
    with pytest.raises(ValueError,match='risk'):
        combine(a,{**b,'failure_probability':.1})


def test_more_frequent_value_refits_cannot_use_future_risk_fits():
    from scripts.kalcb_continuation_study import check_risk_clock
    windows={'2025-08-07':'2025-09-04','2025-09-05':'2025-10-02'}
    check_risk_clock({'valid_from':'2025-08-07'},windows,'2025-08-14','2025-08-18')
    with pytest.raises(ValueError,match='Risk clock'):
        check_risk_clock({'valid_from':'2025-09-05'},windows,'2025-08-14','2025-08-18')
    with pytest.raises(ValueError,match='Risk clock'):
        check_risk_clock({'valid_from':'2025-08-07'},windows,'2025-08-14','2025-09-10')

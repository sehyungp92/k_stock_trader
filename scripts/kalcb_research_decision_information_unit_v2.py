"""Synthetic maturity/prefix checks; no empirical fit or financial evaluation."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
from copy import deepcopy
import importlib.util,json
import numpy as np
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1')
def main():
    source=Path(__file__).with_name('kalcb_research_decision_information_v1.py')
    spec=importlib.util.spec_from_file_location('c18_information_unit',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    out=ROOT/'information_unit_v2';out.mkdir(exist_ok=False)
    m.save(out/'registration.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),program=m.bound(__file__),analysis=m.bound(source),market_evaluations=0))
    calendar=[(datetime(2025,1,1)+timedelta(days=i)).date().isoformat() for i in range(130)];data=[]
    for i,day in enumerate(calendar):
        for symbol,offset in [('A',0),('B',.1)]:
            x=(i%7)+offset
            data.append(dict(day=day,symbol=symbol,features=[x,x*.2,None,x*.3,1,0,2],target=x*2-3,
                label_mature_at=day+'T15:30:00+09:00'))
    first,models,_=m.chronological(data,calendar)
    assert models[0]['fit_day']==calendar[61] and models[0]['train_through']==calendar[59] and models[0]['train_sessions']==60
    altered=deepcopy(data)
    for r in altered:
        if r['day']>=calendar[80]:r['target']+=1000000
    later,_,_=m.chronological(altered,calendar)
    assert [r['prediction'] for r in first if r['day']<=calendar[80]]==[r['prediction'] for r in later if r['day']<=calendar[80]]
    assert all(datetime.fromisoformat(x['latest_label_maturity'])<datetime.fromisoformat(x['fit_day']+'T00:00:00+09:00') for x in models)
    y=np.array([-1.,1.,-1.,1.]);p=np.array([.1,.9,.1,.9]);w=np.ones(4)
    assert m.weighted_auc(y,p,w)==1 and m.weighted_auc(y,np.ones(4),w)==.5
    multi=np.array([[1,1],[2,0],[0,2]])
    assert np.all(m.bootstrap_auc(y,p,w,np.array([0,0,1,1]),multi)==1)
    clock=datetime.fromisoformat('2026-04-01T10:00:00+09:00')
    buy=dict(symbol='A',side='BUY',qty=100,price=100.,timestamp='2026-04-01T09:30:00+09:00')
    prefix=[dict(event='fill',fill=buy,order={'order_id':'b'},costs={'commission_bps':0.,'tax_bps_on_sell':0.},cash_before=1e8,cash_after=1e8-10000),
        dict(event='closing_marks',prices={'A':105.})]
    order=dict(side='SELL',qty=100,reason='failed_followthrough',metadata={'order_role':'EXIT','activation_phase':'decision',
        'decision_available_at':clock.isoformat(),'entry_price':100.,'management_risk_per_share':2.,'hold_bars':6,'mfe_r':4.,'mae_r':-1.})
    base=dict(fills=[buy],complete_native_journal=prefix+[dict(event='submit',order=order)])
    continuation=dict(fills=[buy],complete_native_journal=deepcopy(prefix)+[dict(event='opening_marks',prices={'A':105.})])
    label=dict(label_valid=True,filled_entry_orders=1,first_fill='2026-04-01T09:30:00+09:00',symbol='A',raw_score=3.,
        label_mature_at='2026-04-01T15:30:00+09:00',net_cash_krw=500.)
    extended={**label,'net_cash_krw':1000.};bars=[(clock-timedelta(minutes=5),106.,104.,105.,1000.)]
    contrast,reason=m.holding_pair(base,continuation,label,extended,bars,.1)
    assert reason=='verified' and contrast['target_cash']==500. and contrast['common_cash']==1e8-10000
    assert contrast['features'][4]==2.5 and contrast['features'][6]==1.
    bad=deepcopy(continuation);bad['complete_native_journal'][1]['prices']['A']=90.
    assert m.holding_pair(base,bad,label,extended,bars,.1)[1]=='economic_prefix_not_identical'
    m.save(out/'receipt.json',dict(passed=True,market_evaluations=0,checks=['60mature_plus_one_embargo','future_labels_do_not_change_earlier_predictions',
        'training_only_missing_transforms','weighted_auc_ties_and_group_resampling','holding_common_cash_prefix_reconstructed',
        'earlier_profit_not_credited_to_continuation','mismatched_prefix_excluded','holding_features_at_completed_decision_clock']))
    print('C18 information synthetic tests passed',flush=True)
if __name__=='__main__':main()

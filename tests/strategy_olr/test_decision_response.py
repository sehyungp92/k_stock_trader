from copy import deepcopy
from datetime import datetime,timezone
import pytest
from backtests.strategies.olr.decision_response import reconstruct,available_records,paired_cells

def base():
    r=dict(status='matured',pending_orders=0,residual_quantity=0,filled_quantity=1,filled_notional=10.,total_fees=2.,net_pnl=3.,net_bps=3000.,
           source_sessions=['a:2025-09-01'],fixed_target_risk={'max_drawdown_krw':7.})
    j=[dict(event='opening_marks',timestamp='2025-09-01T09:00:00+09:00',prices={'a':10})]
    def fill(side,price,cash_before,cash_after,id,stamp):return dict(event='fill',cash_before=cash_before,cash_after=cash_after,
        fill=dict(symbol='a',qty=1,price=price,side=side,timestamp=stamp,metadata={'fee':1.,'execution_id':id}))
    j.append(fill('BUY',10,100_000_000,99_999_989,'1','2025-09-01T09:00:00+09:00'))
    j.append(dict(event='closing_marks',timestamp='2025-09-01T09:00:00+09:00',prices={'a':20}))
    j.append(dict(event='opening_marks',timestamp='2025-09-01T09:01:00+09:00',prices={'a':15}))
    j.append(fill('SELL',15,99_999_989,100_000_003,'2','2025-09-01T09:01:00+09:00'))
    # Peak9 to terminal3 is a6 cash-unit giveback, including the sale fee.
    r['fixed_target_risk']['max_drawdown_krw']=6.
    return r,j

def test_independent_cash_fees_inventory_and_full_event_giveback():
    r,j=base();z=reconstruct(r,j)
    assert z['policy_cash_pnl']==3 and z['full_event_dd_krw']==6
    assert len(z['path'])==5 and z['path'][1][0]==z['path'][0][0]

def test_censored_inventory_never_becomes_zero_or_matured_alpha():
    r,j=base();j=j[:-1];r.update(status='right_censored',pending_orders=1,residual_quantity=1,total_fees=1,net_pnl=None,net_bps=None)
    r['fixed_target_risk']['max_drawdown_krw']=5.
    z=reconstruct(r,j);assert z['policy_cash_pnl'] is None and z['known_at'] is None

def test_supported_no_fill_is_known_cash_zero_but_unknown_alpha():
    r,j=base();r.update(status='unfilled',filled_quantity=0,filled_notional=0,total_fees=0,net_pnl=None,net_bps=None)
    r['fixed_target_risk']['max_drawdown_krw']=0.
    z=reconstruct(r,[j[0]]);assert z['policy_cash_pnl']==0 and z['alpha_net_bps'] is None

def test_unknown_source_cannot_be_imputed_to_zero_cash():
    r,j=base();r.update(status='unsupported_source',filled_quantity=0,filled_notional=0,total_fees=0,net_pnl=None,net_bps=None)
    r['fixed_target_risk']['max_drawdown_krw']=0.
    assert reconstruct(r,[j[0]])['policy_cash_pnl'] is None

def test_cash_tamper_and_execution_duplication_rejected():
    r,j=base();x=deepcopy(j);x[1]['cash_after']+=1
    with pytest.raises(ValueError):reconstruct(r,x)
    x=deepcopy(j);x.insert(2,x[1])
    with pytest.raises(ValueError):reconstruct(r,x)

def test_future_labels_and_zero_cash_pairing():
    r,j=base();x=reconstruct(r,j);x.update(day='2025-09-01',symbol='a',budget=20,scenario='stress')
    assert not available_records([x],'2025-09-01T15:30:00+09:00')
    assert available_records([x],'2025-09-02T09:00:00+09:00')==(x,)
    p=paired_cells([x],[('2025-09-01','2025-09-30')],[20],['stress'])
    assert p[0]['cells'][0]['date_mean_net_pnl']==3

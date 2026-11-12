import copy
import pytest
from scripts.kalcb_checkpoint_contrast_review_v2 import account,stamp


def fixture():
    times=['2026-01-05T09:30:00+09:00','2026-01-05T09:35:00+09:00']
    raw={('A',t,'5m'):(100,112,95,105 if i==0 else 110,1000,True) for i,t in enumerate(times)}
    fills=[]
    for i,(side,price) in enumerate([('BUY',100.),('SELL',110.)]):
        fills.append({'symbol':'A','timestamp':times[i],'side':side,'qty':10,'price':price,
            'metadata':{'execution_id':str(i),'execution_bar':dict(zip(('open','high','low','close','volume','is_completed'),raw['A',times[i],'5m']))}})
    # 1bps buy commission; 1bps commission + 20bps tax on sell.
    cash=100000000-1000-.1+1100-2.31
    d={'fills':fills,'costs':{'commission_bps':1.,'tax_bps_on_sell':20.,'slippage_bps':0.},
       'equity_curve':[100000000.,100000049.9,cash],'timestamps':times,'terminal_cash':cash,
       'independent_engine_audit':{'audit_status':'unverified'}}
    v={'fees_and_tax_krw':2.41,'terminal_inventory':0,'net_value_krw':97.59,'full_portfolio_audit_status':'unverified'}
    return d,v,lambda day:raw


def test_independent_full_cash_fee_raw_mtm_reconciliation():
    d,v,raw=fixture();result=account(d,v,raw)
    assert result['marks']==2 and result['fills']==2
    assert result['fees']==pytest.approx(2.41)


@pytest.mark.parametrize('poison',['mtm','fees','inventory','duplicate','raw','initial','qualification'])
def test_independent_review_rejects_poison(poison):
    d,v,raw=fixture()
    if poison=='mtm':d['equity_curve'][1]+=20
    elif poison=='fees':v['fees_and_tax_krw']+=1
    elif poison=='inventory':d['fills'][1]['qty']+=1
    elif poison=='duplicate':d['fills'][1]['metadata']['execution_id']='0'
    elif poison=='raw':d['fills'][0]['metadata']['execution_bar']['close']+=1
    elif poison=='initial':d['equity_curve'][0]+=1
    else:d['independent_engine_audit']['audit_status']='passed'
    with pytest.raises(ValueError):account(d,v,raw)


def test_predecision_inventory_must_match_prior_actual_fills():
    d,v,raw=fixture()
    d['checkpoint']={'state':{'clock':'2026-01-05T09:35:00+09:00','cash':99998999.9,'positions':[{'symbol':'A','qty':9}]}}
    with pytest.raises(ValueError,match='Prestate inventory'):account(d,v,raw)


def test_inherited_impossible_fill_is_retained_but_never_a_valid_payoff_label():
    d,v,raw=fixture();original=raw('2026-01-05');changed={k:(*r[:4],1,True) for k,r in original.items()}
    for f in d['fills']:f['metadata']['execution_bar']['volume']=1
    v.update(independent_execution_pass=False,label_valid=False,net_value_krw=None,full_portfolio_audit_status='failed',
        physical_fill_failures=[{'symbol':f['symbol'],'timestamp':f['timestamp'],'quantity':10,'realized_bar_volume':1,'reason':'shared_bar_volume_exceeded'} for f in d['fills']])
    d['independent_engine_audit']['audit_status']='failed'
    result=account(d,v,lambda day:changed)
    assert not result['execution_valid'] and len(result['physical_failures_retained'])==2
    v['label_valid']=True
    with pytest.raises(ValueError,match='admitted'):account(d,v,lambda day:changed)

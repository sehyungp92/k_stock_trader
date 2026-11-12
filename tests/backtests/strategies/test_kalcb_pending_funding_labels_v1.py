from dataclasses import replace
from datetime import timedelta,time
import ast
from pathlib import Path
import pytest
from backtests.strategies.kalcb.pending_funding_labels_v1 import exact_snapshot,episode
from tests.backtests.strategies.test_kalcb_reclaim_labels_v1 import native_fixture


def setup():
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    from strategy_kalcb.core.preview import preview_entry
    snap,prefix,cfg,at=native_fixture();cfg=replace(cfg,flatten_time=time(14,45),exit_hard_stop_enabled=False)
    qty=preview_entry(prefix,snap,cfg,KALCBPortfolioView(1000000,equity=1000000),available_at=at)['intents'][0]['quantity']
    rows=prefix+[replace(prefix[-1],timestamp=prefix[0].timestamp+timedelta(minutes=5*i),
        open=104.5,high=105,low=104.25,close=104.75,volume=100000) for i in range(18,78)]
    return snap,prefix,rows,cfg,at,qty


@pytest.mark.parametrize('scenario',['baseline','combined'])
def test_real_native_changed_quantity_next_bar_cost_cash_and_independent_audit(scenario):
    snap,prefix,rows,cfg,at,qty=setup();target=max(1,qty//3)
    spec={'episode_id':'fixture','symbol':snap.candidates[0].symbol,'day':str(snap.trade_date),
        'scenario':scenario,'available_at':at.isoformat(),'initial_equity':1000000,'parent_qty':qty,
        'target_qty':target,'reference_price':prefix[-1].close,'action_kind':'capped','aliases':[]}
    index={(b.symbol,b.timestamp.isoformat(),'5m'):(b.open,b.high,b.low,b.close,b.volume,True) for b in rows}
    value,detail=episode(spec,snap,rows,cfg,None,[str(snap.trade_date)],{},lambda p,d:index)
    assert value['status']=='filled' and value['independent_execution_pass']
    assert value['planned_quantity']==target and value['buy_qty']==target
    assert all(f['timestamp']>=at for f in detail['fills'] if f['side']=='BUY')
    assert value['planned_notional_krw']==target*prefix[-1].close
    assert value['net_value_krw']==pytest.approx(detail['terminal_cash']-1000000)
    assert value['full_portfolio_audit_status']=='unverified'
    assert not value['full_native_portfolio_qualification']


def test_original_quantity_requires_exact_native_reconstruction():
    snap,prefix,_,cfg,at,qty=setup()
    with pytest.raises(ValueError,match='parent quantity'):
        exact_snapshot(snap,snap.candidates[0].symbol,qty//2,qty+1,cfg,prefix,1000000,at)


def test_zero_is_not_a_physical_exposure():
    snap,prefix,_,cfg,at,qty=setup()
    with pytest.raises(ValueError,match='quantity request'):
        exact_snapshot(snap,snap.candidates[0].symbol,0,qty,cfg,prefix,1000000,at)


def test_bootstrap_has_no_application_imports_at_module_level():
    tree=ast.parse(Path('scripts/kalcb_pending_funding_labels_v1.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[node.module] if isinstance(node,ast.ImportFrom) else [n.name for n in node.names]
            assert all(n.split('.')[0] not in ('backtests','scripts','strategy_kalcb','kis_core') for n in names)


@pytest.mark.parametrize('poison',[None,'bytes','contract'])
def test_smoke_reuse_checks_bytes_and_exact_action_contract(tmp_path,poison):
    import gzip,json,os
    from backtests.strategies.kalcb.pending_funding_labels_v1 import reuse_smoke
    from backtests.strategies.kalcb.retention_research_io_v1 import bound,write
    source=tmp_path/'smoke/e';source.mkdir(parents=True);target=tmp_path/'labels/e';target.mkdir(parents=True)
    spec={'episode_id':'e','target_qty':7};write(source/'label.json',{'planned_quantity':7})
    with gzip.open(source/'events.json.gz','wt',encoding='utf-8') as f:json.dump({'spec':spec},f)
    write(tmp_path/'smoke/completion.json',{'outputs':[{'label':bound(source/'label.json'),'events':bound(source/'events.json.gz')}]})
    if poison=='bytes':write(source/'label.json',{'planned_quantity':99})
    if poison=='contract':spec={**spec,'target_qty':8}
    if poison:
        with pytest.raises(ValueError):reuse_smoke(tmp_path,spec,target)
    else:
        result,_=reuse_smoke(tmp_path,spec,target)
        assert result['planned_quantity']==7
        assert os.path.samefile(source/'events.json.gz',target/'events.json.gz')

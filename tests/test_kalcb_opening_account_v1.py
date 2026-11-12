from dataclasses import replace
from datetime import timedelta
import numpy as np
import pytest

from strategy_kalcb.core.account_allocation_v1 import entry_allocation, allocate_quantity
from backtests.strategies.kalcb.opening_book_risk_v1 import allocate, downside, HistoricalPaths
from backtests.strategies.kalcb import account_decision_v1 as accounts


def intents():
    return [dict(symbol='a', qty=1000, price=100., score=2.),
            dict(symbol='b', qty=1000, price=100., score=1.)]


def paths():
    p = np.zeros((60, 71, 2))
    p[:, :, 0] = -.02
    p[:, :, 1] = -.10
    return p


def test_disabled_and_scoped_integer_action_metadata():
    meta = {'entry_route':'first30_open_anchor'}
    assert allocate_quantity('a',100,meta) == 100 and len(meta) == 1
    book = {'a':{'parent_qty':100,'qty':25}}
    with entry_allocation(book):
        assert allocate_quantity('a',100,meta) == 25
        assert meta['account_allocation_v1']['qty'] == 25
        assert allocate_quantity('a',100,{'entry_route':'checkpoint'}) == 100
        with pytest.raises(ValueError,match='absent'):allocate_quantity('b',100,meta)
        with pytest.raises(ValueError,match='changed'):allocate_quantity('a',101,meta)
    assert allocate_quantity('a',100,{}) == 100


@pytest.mark.parametrize('qty',[-1,101,2.5,True])
def test_invalid_allocations_fail_before_submission(qty):
    with entry_allocation({'a':{'parent_qty':100,'qty':qty}}):
        with pytest.raises(ValueError):allocate_quantity('a',100,{'entry_route':'first30_open_anchor'})


def test_joint_path_budget_preserves_more_valuable_lower_downside_and_matches_control():
    result = allocate(intents(),paths(),equity=100000, buying_power=200000,budget=.06)
    assert result['joint'][0] == 1000
    assert 0 < result['joint'][1] < 500
    assert result['joint_risk'] <= .06+1e-8
    assert abs(result['uniform_risk']-result['joint_risk']) < .00013
    assert result['joint_notional'] > result['uniform_notional']


def test_unconstrained_book_is_exact_identity_and_funding_includes_costs():
    out = allocate(intents(),paths(),equity=100000,buying_power=300000,budget=.5)
    assert out['joint'] == out['uniform'] == [1000,1000] and not out['changed']
    constrained = allocate(intents(),paths(),equity=100000,buying_power=100000,budget=.5,cost_fraction=.01)
    assert constrained['joint_notional']*1.01 <= 100000+.01


def test_synchronized_common_shock_not_independent_marginal_low():
    p = np.zeros((60,2,2));p[:,0,:]=[-.1,.1];p[:,1,:]=[.1,-.1]
    assert downside(p,[1,1]) == 0
    assert downside(p,[1,0]) == pytest.approx(.1)


def test_history_embargo_future_poison_and_missing_bar_fail_closed(tmp_path):
    import sqlite3
    from datetime import date,time
    db=tmp_path/'source.sqlite';c=sqlite3.connect(db)
    c.execute('create table bars(symbol text,ts text,day text,o real,c real)')
    for i in range(63):
        day=str(date(2025,1,1)+timedelta(days=i))
        for m in range(71):
            stamp=f'{day}T{9+(25+5*m)//60:02d}:{(25+5*m)%60:02d}:00+09:00'
            price=100 if m<=1 else 99 if i<60 else 1000000
            c.execute('insert into bars values(?,?,?,?,?)',('a',stamp,day,price,price))
    c.commit()
    hours=lambda day:(time(9),time(15,30))
    reader=HistoricalPaths([db],hours);p,proof=reader.matrix('2025-03-03',['a'])
    assert proof['history_through']=='2025-03-01' and proof['embargo_session']=='2025-03-02'
    assert p.max()==0. and p.min()==pytest.approx(-.01)
    reader.close()
    c.execute("delete from bars where day='2025-02-20' and substr(ts,12,5)='10:00'");c.commit();c.close()
    reader=HistoricalPaths([db],hours);p,proof=reader.matrix('2025-03-03',['a']);reader.close()
    assert p is not None and len(proof['unobserved_clocks'])==1
    c=sqlite3.connect(db);c.execute("delete from bars where day='2025-02-20' and substr(ts,12,5)='09:30'");c.commit();c.close()
    reader=HistoricalPaths([db],hours);p,proof=reader.matrix('2025-03-03',['a']);reader.close()
    assert p is None and proof['reason']=='missing_historical_open_reference'


def test_generic_fork_retains_history_and_rejects_changed_parent(tmp_path):
    from tests.test_kalcb_pending_entry_account_pilot_v1 import example
    c,_=example('baseline');aa,bb=c['parent']
    boundary=accounts.capture(aa,bb,bb.timestamps[-1]+timedelta(minutes=5),tmp_path)
    ca,cb=accounts.fork(aa,bb,boundary)
    accounts.assert_identity((aa,bb),(ca,cb))
    cb.cash+=1
    with pytest.raises(ValueError,match='differs'):accounts.assert_identity((aa,bb),(ca,cb))
    aa.state.meta['poison']=1
    with pytest.raises(ValueError,match='changed'):accounts.fork(aa,bb,boundary)


def test_immutable_prefix_prevents_nested_poison_and_shares_only_bytes():
    from copy import deepcopy
    from types import MappingProxyType
    from backtests.strategies.kalcb.account_history_v1 import AccountHistory
    parent=AccountHistory([{'nested':[1]},MappingProxyType({'frozen':[3]})])
    child=deepcopy(parent)
    assert parent._prefix is child._prefix and not child._tail
    child[0]['nested'].append(2)
    child[1]['frozen'].append(4)
    assert parent[0]=={'nested':[1]} and parent[1]=={'frozen':[3]}
    child.append({'own':5})
    assert len(child)==3 and len(parent)==2 and child[2:]==[{'own':5}]
    assert bool(parent) and parent[:1]==[{'nested':[1]}]
    import json
    assert json.loads(json.dumps(parent))==[{'nested':[1]},{'frozen':[3]}]
    with pytest.raises(TypeError):child[0]={'bad':True}


def test_path_value_includes_inherited_peak_and_counts_lifetimes_not_partials():
    from types import SimpleNamespace
    boundary=accounts.AccountBoundary('x','clock',100.,120.,1,0,0)
    fills=[SimpleNamespace(side='BUY',symbol='a',order_id='one',qty=1,price=2.) for _ in range(3)]
    broker=SimpleNamespace(equity_curve=[100.,100.,110.,90.,130.], timestamps=['t0','t1','t2','t3'],
        fills=fills, positions={},orders=[])
    value=accounts.suffix_value(boundary,broker)
    assert value['max_drawdown_from_inherited_peak']==.25
    assert value['funded_entry_orders']==1 and value['net_change_krw']==30


def test_native_core_patch_disabled_identity_and_actual_allocated_order(monkeypatch):
    import inspect
    from strategy_kalcb.core import logic
    from backtests.strategies.kalcb.opening_account_io_v1 import patch_core
    from tests.backtests.strategies.test_kalcb_checkpoint_contrast_labels_v1 import fixture
    from scripts.kalcb_executed_value_study import run_episode
    row,bars,cfg,_,_=fixture()
    text=patch_core(inspect.getsource(logic._evaluate_entry))
    namespace=dict(logic.__dict__);exec(text,namespace)
    original=run_episode(row,bars,cfg,'baseline','original',None,[row['trade_date']])
    monkeypatch.setattr(logic,'_evaluate_entry',namespace['_evaluate_entry'])
    disabled=run_episode(row,bars,cfg,'baseline','original',None,[row['trade_date']])
    assert original==disabled
    qty=row['action']['quantity']
    with entry_allocation({row['symbol']:{'parent_qty':qty,'qty':qty//2}}):
        result,detail=run_episode(row,bars,cfg,'baseline','allocated',None,[row['trade_date']])
    orders=[r['order'] for r in detail['orders'] if r['event']=='submit' and r['order']['side']=='BUY']
    assert len(orders)==1 and orders[0]['qty']==qty//2
    assert orders[0]['metadata']['risk_reserved_entry_qty']==qty//2
    assert result['label_valid'] and not result['terminal_inventory'] and not result['terminal_pending']


def test_objective_rejects_oos_gain_bought_with_material_is_loss():
    from backtests.strategies.kalcb.opening_account_plugin_v1 import objective,SCENARIOS
    parent={f'{p}_{s}_{m}':v for p in ('is','oos') for s in SCENARIOS
            for m,v in [('return',50. if p=='is' else 5.),('dd',10.),('material',100)]}
    candidate={k:v+(2 if k.startswith('oos_') and k.endswith('_return') else 0) for k,v in parent.items()}
    assert objective(candidate,parent)['passed']
    candidate['is_cost_return']=48.
    assert not objective(candidate,parent)['passed']


def test_native_whole_account_preview_forks_and_sequential_action(monkeypatch,tmp_path):
    import inspect
    from strategy_kalcb.core import logic
    from backtests.strategies.kalcb.opening_account_io_v1 import patch_core
    from backtests.strategies.kalcb.opening_account_replay_v1 import OpeningAccountReplay
    from backtests.strategies.kalcb import executed_value as ev
    from tests.backtests.strategies.test_kalcb_checkpoint_contrast_labels_v1 import fixture
    from scripts.kalcb_executed_value_study import run_episode
    row,bars,cfg,_,_=fixture()
    cfg=replace(cfg,frontier_shadow_enabled=False,heat_cap_r=.01)
    namespace=dict(logic.__dict__);exec(patch_core(inspect.getsource(logic._evaluate_entry)),namespace)
    monkeypatch.setattr(logic,'_evaluate_entry',namespace['_evaluate_entry'])
    snapshot=ev.snapshot_for
    monkeypatch.setattr(ev,'snapshot_for',lambda day,rows,**kw:snapshot(day,[(r,2.) for r,_ in rows],**kw))
    def controller(mode,pairs=0):
        obj=OpeningAccountReplay(mode,[],pair_limit=pairs,pair_store=tmp_path/'prefixes')
        obj.paths.matrix=lambda day,symbols:(np.full((60,71,len(symbols)),-.5),{'reason':'fixture'})
        return obj
    hold=controller('control',1);hold.install()
    try:
        result,detail=run_episode(row,bars,cfg,'baseline','native-fixture',None,[row['trade_date']])
        hold.finish_pair()
    finally:hold.restore()
    assert hold.pairs[0]['exact_disabled_identity']
    assert hold.records[0]['changed']
    joint=controller('joint');joint.install()
    try:
        jr,jd=run_episode(row,bars,cfg,'baseline','native-fixture',None,[row['trade_date']])
    finally:joint.restore()
    assert jr['label_valid'] and not jr['terminal_pending'] and not jr['terminal_inventory']
    assert sum(f['qty'] for f in jd['fills'] if f['side']=='BUY') < sum(f['qty'] for f in detail['fills'] if f['side']=='BUY')
    expected=hold.pairs[0]['contrasts']['joint']['marginal_terminal_wealth_krw']
    assert jr['net_value_krw']-result['net_value_krw']==pytest.approx(expected)

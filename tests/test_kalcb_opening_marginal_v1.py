from datetime import datetime
from dataclasses import replace
import inspect
import pytest
from backtests.strategies.kalcb.opening_marginal_replay_v1 import omit_opening,apply_omission,OpeningMarginalReplay


def test_veto_scope_retries_and_context_restoration():
    identity=lambda s,q,m:q;hits=[];stamp=datetime(2025,8,8,9,25)
    with omit_opening('a',stamp,hits):
        assert apply_omission(identity,'a',100,{'entry_route':'first30_open_anchor'})==0
        assert apply_omission(identity,'a',90,{'entry_route':'first30_open_anchor'})==0
        assert apply_omission(identity,'b',100,{'entry_route':'first30_open_anchor'})==100
        assert apply_omission(identity,'a',100,{'entry_route':'pullback'})==100
        with pytest.raises(RuntimeError):
            with omit_opening('b',stamp,[]):raise RuntimeError('test')
        assert apply_omission(identity,'a',80,{'entry_route':'first30_open_anchor'})==0
    assert apply_omission(identity,'a',100,{'entry_route':'first30_open_anchor'})==100
    assert [x['requested_qty'] for x in hits]==[100,90,80]


def test_real_native_route_omission_and_complete_account_identity(monkeypatch,tmp_path):
    from strategy_kalcb.core import logic
    from backtests.strategies.kalcb.opening_account_io_v1 import patch_core
    from tests.backtests.strategies.test_kalcb_checkpoint_contrast_labels_v1 import fixture
    from scripts.kalcb_executed_value_study import run_episode
    row,bars,cfg,_,_=fixture()
    cfg=replace(cfg,frontier_shadow_enabled=False)
    namespace=dict(logic.__dict__);exec(patch_core(inspect.getsource(logic._evaluate_entry)),namespace)
    monkeypatch.setattr(logic,'_evaluate_entry',namespace['_evaluate_entry'])
    control=run_episode(row,bars,cfg,'baseline','fixture',None,[row['trade_date']])
    controller=OpeningMarginalReplay(tmp_path);controller.install()
    try:
        actual=run_episode(row,bars,cfg,'baseline','fixture',None,[row['trade_date']]);controller.finish()
    finally:controller.restore()
    assert actual==control
    branches=[r for r in controller.pairs if r['omitted_symbol']]
    assert len(branches)==1
    assert branches[0]['delta_wealth_krw']==pytest.approx(-branches[0]['donor_first30_control_pnl_krw'])
    assert branches[0]['other_opportunity_effect_krw']==pytest.approx(0,abs=.001)


def test_unregistered_population_stops_real_native_replay(monkeypatch,tmp_path):
    from strategy_kalcb.core import logic
    from backtests.strategies.kalcb.opening_account_io_v1 import patch_core
    from tests.backtests.strategies.test_kalcb_checkpoint_contrast_labels_v1 import fixture
    from scripts.kalcb_executed_value_study import run_episode
    row,bars,cfg,_,_=fixture();cfg=replace(cfg,frontier_shadow_enabled=False);namespace=dict(logic.__dict__)
    exec(patch_core(inspect.getsource(logic._evaluate_entry)),namespace)
    monkeypatch.setattr(logic,'_evaluate_entry',namespace['_evaluate_entry'])
    controller=OpeningMarginalReplay(tmp_path,[]);controller.install()
    try:
        with pytest.raises(ValueError,match='Unregistered native opening population'):
            run_episode(row,bars,cfg,'baseline','fixture',None,[row['trade_date']])
    finally:controller.restore()

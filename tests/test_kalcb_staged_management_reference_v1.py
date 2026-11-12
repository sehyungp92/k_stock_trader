from dataclasses import replace
import json
import pytest
from tests.test_kalcb_staged_commitment_v1 import setup,fill_initial,completion,event,CORE,CFG,AT
from strategy_kalcb.core import staged_management_reference_v1 as ref
from strategy_kalcb.core.serializers import snapshot_state,restore_state

ref.install_reference()

def test_cost_basis_changes_but_original_management_reference_and_age_survive():
    s,a=setup();s.meta['staged_management_reference_enabled']=True;fill_initial(s)
    p=s.symbol_state(a.symbol).position;p.hold_bars=8;completion(s)
    CORE.on_kalcb_fill(s,event('completion',4,104),CFG)
    assert p.entry_price==pytest.approx((500+416)/9)
    assert p.metadata[ref.META]==100 and p.unrealized_r(102)==1
    assert p.entry_time==AT and p.hold_bars==8 and p.risk_per_share==2
    assert p.metadata['entry_fees']==2

def test_disabled_reference_retains_actual_cost_R_exactly():
    s,a=setup();fill_initial(s);completion(s);CORE.on_kalcb_fill(s,event('completion',4,104),CFG)
    p=s.symbol_state(a.symbol).position
    assert ref.META not in p.metadata
    assert p.unrealized_r(102)==pytest.approx((102-p.entry_price)/2)

def test_reference_persists_across_restart_without_changing_broker_cost_field():
    s,a=setup();s.meta['staged_management_reference_enabled']=True;fill_initial(s);completion(s)
    s=restore_state(json.loads(json.dumps(snapshot_state(s))))
    CORE.on_kalcb_fill(s,event('completion',4,104),CFG);p=s.symbol_state(a.symbol).position
    assert p.entry_price>100 and p.unrealized_r(102)==1
    p.update_mark(high=106,low=97)
    assert p.mfe_r()==3 and p.mae_r()==-1.5

def test_invalid_reference_fails_and_unmodified_position_delegates():
    s,a=setup();fill_initial(s);p=s.symbol_state(a.symbol).position
    p.metadata[ref.META]=float('nan')
    with pytest.raises(ValueError):p.unrealized_r(102)

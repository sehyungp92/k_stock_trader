from copy import deepcopy
import pytest
from scripts.kalcb_pending_funding_qualification_v1 import require_controls,complete_grid


def grid():
    return [dict(name=n,period=p,scenario=s,reused=s in ('baseline','combined'),
                 metrics={'audit_status':'passed'},identity_passed=n=='control')
            for n in ('control','pending_100') for p in ('is','oos')
            for s in ('baseline','participation','cost','combined')]


def test_controls_need_all_stresses_and_both_periods():
    rows=grid()[:8];require_controls(rows)
    for i in range(8):
        with pytest.raises(ValueError):require_controls(rows[:i]+rows[i+1:])


def test_duplicate_cannot_replace_missing_case():
    rows=grid();rows[-1]=deepcopy(rows[-2])
    with pytest.raises(ValueError,match='duplicate'):complete_grid(rows)


def test_failed_source_or_bridge_cannot_gate_variant():
    rows=grid();rows[2]['identity_passed']=False
    with pytest.raises(ValueError):require_controls(rows)
    rows=grid();rows[3]['metrics']['audit_status']='unverified'
    with pytest.raises(ValueError):require_controls(rows)


def test_fresh_count_and_any_candidate_audit_are_enforced():
    rows=grid();complete_grid(rows)
    rows[-1]['reused']=False
    with pytest.raises(ValueError):complete_grid(rows)
    rows=grid();rows[-1]['metrics']['audit_status']='failed'
    with pytest.raises(ValueError):complete_grid(rows)

from copy import deepcopy
from dataclasses import replace,asdict
from datetime import timedelta,datetime
import json
import pytest

from backtests.strategies.kalcb import account_fork_v2 as states
from backtests.strategies.kalcb import holding_action_v1 as actions


def test_immutable_merkle_history_and_source_maps():
    h=states.MerkleHistory([{'nested':[1]}]);c=deepcopy(h)
    assert h.commitment()==c.commitment() and h._prefix is c._prefix
    c[0]['nested'].append(2)
    assert h[0]=={'nested':[1]}
    c.append({'b':2});assert c.commitment()!=h.commitment()
    m=states.FrozenValueMap({'key':{'nested':[1]}})
    assert deepcopy(m) is m
    m['key']['nested'].append(2);assert m['key']=={'nested':[1]}


def test_contract_rejects_standalone_or_wrong_quantity_value():
    c=actions.ValueContract();c.validate(asdict(c))
    for key in ('target','quantity_basis','continuation','maturity'):
        raw=asdict(c);raw[key]='wrong'
        with pytest.raises(ValueError):c.validate(raw)


def fixture_run(tmp_path,monkeypatch,mode='labels',limit=2,poison=None):
    from tests.backtests.strategies.test_kalcb_checkpoint_contrast_labels_v1 import fixture
    from scripts.kalcb_executed_value_study import run_episode
    from backtests.strategies.kalcb.holding_account_replay_v1 import HoldingAccountReplay
    from backtests.strategies.kalcb.pending_entry_account_pilot_v1 import state_receipt
    monkeypatch.setattr('backtests.strategies.kalcb.executed_value.REFERENCE_EQUITY',1_000_000.)
    row,bars,cfg,_,_=fixture();cfg=replace(cfg,frontier_shadow_enabled=False)
    obj=HoldingAccountReplay(tmp_path,mode=mode,limit=limit)
    obj.install();obj.set_source(bars)
    if poison:
        original=actions.apply
        def bad(aa,bb,obs,arm):
            obs=deepcopy(obs)
            if poison=='clock':obs['available_at']=obs['signal_bar']
            elif poison=='anchor':obs['management_reference']+=1
            elif poison=='quantity':obs['qty']+=1
            return original(aa,bb,obs,arm)
        monkeypatch.setattr('backtests.strategies.kalcb.holding_account_replay_v1.apply',bad)
    try:
        result,detail=run_episode(row,bars,cfg,'combined','holding-fixture',None,[row['trade_date']])
        obj.finish()
        aa,bb=obj.parent
        before=state_receipt(aa,bb);boundary,_=states.capture(aa,bb,bb.timestamps[-1]+timedelta(minutes=5))
        ca,cb=states.fork(aa,bb,boundary)
        assert before==state_receipt(aa,bb)==state_receipt(ca,cb)
        ca.state.meta['independent']=True;cb.cash+=1
        assert 'independent' not in aa.state.meta and cb.cash!=bb.cash
        return obj,result,detail
    finally:obj.restore()


def test_native_labels_exact_parent_clock_cost_and_account(tmp_path,monkeypatch):
    obj,result,detail=fixture_run(tmp_path,monkeypatch)
    assert obj.rows==2 and obj.forks==4 and len(obj.checked_days)==1
    rows=[json.loads(x) for x in (tmp_path/'labels.jsonl').read_text().splitlines()]
    assert len(rows)==2 and result['label_valid']
    for r in rows:
        assert r['observation']['qty']>0 and set(r['observation']['features'])==set(actions.FEATURES)
        for arm,c in r['contrasts'].items():
            assert datetime.fromisoformat(c['first_fill'])>=datetime.fromisoformat(r['observation']['available_at'])
            assert c['actual_action_qty']==r['observation']['qty']//2 if arm=='half' else c['actual_action_qty']==r['observation']['qty']
    # Observation layer leaves every actual trade and native mark unchanged.
    other=tmp_path/'disabled';other.mkdir()
    _,base,bd=fixture_run(other,monkeypatch,mode='disabled')
    assert result==base and detail==bd


@pytest.mark.parametrize('poison',['clock','anchor','quantity'])
def test_native_rejects_stale_and_anticipatory_actions(tmp_path,monkeypatch,poison):
    with pytest.raises(ValueError):fixture_run(tmp_path,monkeypatch,poison=poison)


@pytest.mark.parametrize('poison',[None,'fee','price','early','duplicate'])
def test_independent_saved_branch_reconstruction(tmp_path,monkeypatch,poison):
    from scripts.kalcb_holding_account_review_v1 import check_ledger,gz
    fixture_run(tmp_path,monkeypatch,limit=1)
    r=json.loads((tmp_path/'labels.jsonl').read_text().splitlines()[0])
    b=r['contrasts']['half']['ledger'];d=deepcopy(gz(b['path'],b['sha256']))
    s=r['source'];source=gz(s['path'],s['sha256'])
    if poison=='fee':d['fills'][0]['metadata']['fee']+=1
    elif poison=='price':d['fills'][0]['price']+=1
    elif poison=='early':d['fills'][0]['metadata']['decision_available_at']='2026-01-05T15:30:00+09:00'
    elif poison=='duplicate':d['fills'].append(deepcopy(d['fills'][0]))
    costs={'commission_bps':2.,'tax_bps_on_sell':18.,'slippage_bps':13.}
    if poison:
        with pytest.raises(ValueError):check_ledger(r,d,source,costs,.005)
    else:assert check_ledger(r,d,source,costs,.005)['marks']>0

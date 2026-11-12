from dataclasses import replace
from unittest.mock import patch
import pytest
from backtests.strategies.kalcb import pending_entry_account_pilot_v1 as p


def example(scenario):
    from tests.backtests.strategies.test_kalcb_checkpoint_contrast_labels_v1 import fixture
    from scripts.kalcb_executed_value_study import run_episode
    from backtests.engine import replay
    row,bars,cfg,_,_=fixture();q=row['action']['quantity'];ref=bars[5].close
    # Historical parent explicitly disables frontier shadow; reject it instead
    # of dropping an active shadow from a purported exact account clone.
    cfg=replace(cfg,frontier_shadow_enabled=False)
    bars[5]=replace(bars[5],volume=max(200,int(q/.04)))
    bars[6]=replace(bars[6],open=ref,high=ref,low=ref*.99,close=ref*.99,volume=max(200,int(q/.04)))
    indexes={row['trade_date']:{(b.symbol,b.timestamp.isoformat(),'5m'):
        (b.open,b.high,b.low,b.close,b.volume,True) for b in bars}}
    kernel=replay.drive_timestamp_batch; captured={}
    def drive(broker,batch,adapters,**kw):
        result=kernel(broker,batch,adapters,**kw)
        if 'pair' in captured:
            for aa,bb in captured['pair']['arms'].values():kernel(bb,batch,(aa,))
        stamp=batch[0].timestamp
        if (stamp.hour,stamp.minute)==(9,30):
            before=p.state_receipt(adapters[0],broker)['sha256']
            assert any(r['reason']=='cancel_requested' for r in p.eligibility(adapters[0],broker,stamp))
            assert before==p.state_receipt(adapters[0],broker)['sha256']
            captured['pair']=p.fork_pair(adapters[0],broker,stamp)
        captured['parent']=(adapters[0],broker)
        return result
    with patch.object(replay,'drive_timestamp_batch',drive):
        run_episode(row,bars,cfg,scenario,'same-account-fixture',None,[row['trade_date']])
    return captured,indexes


@pytest.mark.parametrize('scenario',['baseline','combined'])
def test_full_state_native_continuation_and_independent_account(scenario,tmp_path):
    c,indexes=example(scenario);pair=c['pair']
    assert p.state_receipt(*pair['arms']['hold'])==p.state_receipt(*c['parent'])
    result=p.save_pair(tmp_path,scenario,1,pair,c['parent'],indexes)
    assert result['exact_disabled_continuation'] and result['original_later_buy_qty']>0
    assert result['full_portfolio_audit_status']=={'hold':'unverified','cancel':'unverified'}
    for aa,bb in pair['arms'].values():
        check=p.account_detail(aa,bb,indexes)
        assert check['independent_account']['raw_close_marks']==78


def test_every_mutable_adapter_and_broker_field_copied():
    c,_=example('baseline');aa,bb=c['parent'];ca,cb=p.clone_account(aa,bb)
    aa.extra_nested={'test':[1]};ca,cb=p.clone_account(aa,bb)
    ca.extra_nested['test'].append(2);cb.journal.append({'event':'poison'})
    ca.state.meta['independent']=True
    assert aa.extra_nested=={'test':[1]} and bb.journal[-1].get('event')!='poison'
    assert 'independent' not in aa.state.meta


def test_source_poison_is_rejected():
    c,indexes=example('combined');key=next(iter(indexes[next(iter(indexes))]))
    indexes[next(iter(indexes))][key]=(0,0,0,0,0,True)
    with pytest.raises(ValueError,match='raw source mismatch'):p.account_detail(*c['parent'],indexes)


def test_disabled_account_poison_is_rejected(tmp_path):
    c,indexes=example('baseline');c['pair']['arms']['hold'][1].cash+=1
    with pytest.raises(ValueError,match='Disabled native continuation'):
        p.save_pair(tmp_path,'baseline',1,c['pair'],c['parent'],indexes)


@pytest.mark.parametrize('poison',[None,'fee','price','early','duplicate'])
def test_independent_saved_ledger_reader_and_poison(poison):
    import json
    from datetime import datetime
    from scripts.kalcb_pending_entry_account_review_v1 import validate
    c,indexes=example('combined')
    d=json.loads(json.dumps(p.account_detail(*c['pair']['arms']['cancel'],indexes),default=str))
    def bars(symbol,day):
        return {datetime.fromisoformat(t):v[:5] for (s,t,tf),v in indexes[day].items() if s==symbol}
    if poison is None:
        assert validate(d,bars)['marks']==78
        return
    f=d['fills'][0]
    if poison=='fee':f['metadata']['fee']+=1
    elif poison=='price':f['price']+=1
    elif poison=='early':f['metadata']['decision_available_at']='2026-01-05T10:00:00+09:00'
    else:d['fills'].append(dict(f))
    with pytest.raises(ValueError):validate(d,bars)

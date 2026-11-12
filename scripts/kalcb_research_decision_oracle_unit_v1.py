"""Native-interface synthetic tests of oracle-only hooks, no market replay."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
from copy import deepcopy
import sys,json,gzip,importlib.util,hashlib
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1')
REPO=Path(__file__).resolve().parents[1]
def bound(p):
    with Path(p).open('rb') as f:return dict(path=str(p),sha256=hashlib.file_digest(f,'sha256').hexdigest())
def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,default=str)
def main():
    out=ROOT/'oracle_unit_v1';out.mkdir(exist_ok=False)
    path=REPO/'backtests/strategies/kalcb/research_decision_oracle_v1.py'
    save(out/'registration.json',dict(program=bound(__file__),helper=bound(path),protocol=bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0043-counterfactual-protocol.md'),market_evaluations=0))
    code=ROOT/'native_v1/is/inputs/files';sys.path.insert(0,str(code))
    spec=importlib.util.spec_from_file_location('c18_oracle_unit_helper',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    from scripts import kalcb_executed_value_replay as replay
    from backtests.engine import integrity
    before=(replay.verify_policy,replay.timed_policy,integrity.audit_broker)
    off=m.OracleProjection('off','is','baseline',{},out);off.install();off.restore()
    assert before==(replay.verify_policy,replay.timed_policy,integrity.audit_broker)
    with gzip.open(ROOT/'cohorts_v1/episodes.jsonl.gz','rt') as f:
        original=next(json.loads(l) for l in f if json.loads(l)['candidate'])
    obj=m.OracleProjection.__new__(m.OracleProjection)
    obj.arm='SE';obj.period='is';obj.cohorts={};obj.menus={};obj.labels={};obj.choices=[];obj.membership={};obj.entry_records={};obj.exit_records=[];obj.reg={'cohorts':{'sha256':'synthetic'}}
    for i,(cash,score,core) in enumerate([(300,-4,False),(250,7,False),(200,2,True)],1):
        key='toy/'+str(i);r=deepcopy(original);r.update(episode_id=key,trade_date='2026-04-01',symbol=str(i).zfill(6),cohort='opening')
        r['candidate']['symbol']=r['symbol'];r['candidate']['trade_date']='2026-04-01';r['feature_available_at']='2026-04-01T09:30:00+09:00'
        r['original_admission']={'record':{'forecast_origin':'frozen_core'}} if core else None
        obj.cohorts[key]=r;obj.menus[key]=dict(net_cash_krw=cash,raw_score=score,horizon_minutes=60,quantity_scale=.25,first_fill='2026-04-01T09:30:00+09:00',buy_quantity=10)
    chosen=obj.choose('2026-04-01','opening',8)
    assert [x[0] for x in chosen]==['toy/1','toy/3'] and obj.choices[0]['new_support_slots']==1
    snap=obj.snapshot('2026-04-01',chosen)
    assert [c.selection_score for c in snap.candidates]==[-4,2]
    assert all(not any(k.startswith('oracle') or k in ('net_cash_krw','label_cash','horizon_minutes') for k in c.metadata) for c in snap.candidates)
    position=SimpleNamespace(entry_time=datetime.fromisoformat('2026-04-01T09:35:00+09:00'),symbol='000001',entry_order_id='new-native-order',qty_open=17)
    assert obj.horizon(None,position,None)==60
    assert obj.exit_records[0]['actual_entry']!=obj.exit_records[0]['label_entry'] and obj.exit_records[0]['actual_qty']==17
    # Original quantity and fill time are observations, never account mutations.
    assert position.qty_open==17 and position.entry_time.minute==35
    assert obj.choose('2026-04-01','opening',1)[0][0]=='toy/1'
    save(out/'receipt.json',dict(passed=True,market_evaluations=0,checks=['off_function_identity','one_support_slot_preserved',
        'native_snapshot_preserves_causal_score_not_hindsight_payoff','future_book_separate_from_candidate_metadata',
        'actual_account_quantity_and_entry_clock_not_overwritten','deterministic_finite_choice'],helper=bound(path)))
    print('C18 oracle synthetic interface tests passed',flush=True)
if __name__=='__main__':main()

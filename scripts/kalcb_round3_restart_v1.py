"""Fresh-process frozen-native checkpoint continuation through the next session."""
import argparse,hashlib,json,pickle,sys
from pathlib import Path
from datetime import date,datetime
from itertools import groupby
p=argparse.ArgumentParser();p.add_argument('--receipt',required=True);p.add_argument('--label',choices=['pending','inventory'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
receipt=json.loads(Path(a.receipt).read_text());folder=Path(receipt['folder']);item=receipt['item'];code=Path(item['manifest']['path']).parent/'files'
sys.path.insert(0,str(code))
from backtests.strategies.kalcb.round3_integrity_v1 import neutral,rows
from strategy_kalcb.core.serializers import restore_state
from backtests.engine.sim_broker import SimBroker
from backtests.engine.replay import drive_timestamp_batch
from backtests.strategies.kalcb.runner import KALCBReplayAdapter
from backtests.strategies.kalcb.funded_horizon_v1 import HorizonReplay
from scripts.kalcb_executed_value_study import load_plan,connect,load_day
from scripts.kalcb_executed_value_replay import observation_prefix,verify_policy
from backtests.auto.shared.cache_keys import source_generation_hash
from scripts.kalcb_pending_funding_worker_v1 import origins
manifest=json.loads(Path(item['manifest']['path']).read_text());before=origins(code,manifest)
assert source_generation_hash()==item['source_generation_hash']
checkpoint=folder/(a.label+'.checkpoint.pkl');expected=json.loads((folder/'hydration.json').read_text())
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==next(x['checkpoint_sha256'] for x in expected if x['label']==a.label)
with checkpoint.open('rb') as f:packet=pickle.load(f) # Own authenticated local checkpoint only.
broker=SimBroker.__new__(SimBroker);broker.__dict__.update(packet['broker']);broker.source_events=hashlib.sha256()
for batch in broker.expected_batches:
    for bar in batch:broker.source_events.update(json.dumps(bar,sort_keys=True,default=str).encode())
assert broker.source_events.hexdigest()==packet['source_hash']
adapter=KALCBReplayAdapter.__new__(KALCBReplayAdapter);adapter.__dict__.update(packet['adapter']);adapter.state=restore_state(packet['core'])
last=datetime.fromisoformat(broker.expected_batches[-1][0]['timestamp']);plan=load_plan(Path(item['plan']['path']));conn=connect(plan)
days=[r[0] for r in conn.execute('SELECT DISTINCT day FROM bars WHERE day>=? ORDER BY day',(str(last.date()),))][:2]
policy,snaps=verify_policy(Path(item['policy']['path']).parent,'interaction_cost_margin')
controller=HorizonReplay('veto',json.loads((code/'research/horizon_policy/map.json').read_text()));controller.install();decisions=[]
try:
    for day in days:
        key=date.fromisoformat(day);symbols={c.symbol for c in adapter.snapshots[key].candidates}
        for update in adapter.timed_snapshots.get(key,()):symbols.update(c.symbol for c in update.candidates)
        symbols.update(s for _,s in broker.positions);symbols.update(o.symbol for o in broker.orders)
        if not symbols:symbols={'000100'}
        execution_symbols=set(symbols);symbols.update(policy.get('observation_symbols',()))
        bars=load_day(conn,day,plan['input_sha256'][plan['database']],symbols)
        bars=observation_prefix(bars,key,execution_symbols,policy['observation_minutes'])
        for at,batch in groupby(bars,key=lambda b:b.timestamp):
            if at<=last:continue
            decisions.extend(e.to_json_dict() for e in drive_timestamp_batch(broker,tuple(batch),(adapter,)))
finally:controller.restore();conn.close()
actual_journal=neutral(broker.journal);expected_journal=[x['record'] for x in rows(folder/'broker_journal.jsonl.gz')][:len(actual_journal)]
assert actual_journal==expected_journal,'Fresh-process journal suffix differs'
expected_decisions=[x for x in rows(folder/'decisions.jsonl.gz') if last<datetime.fromisoformat(x['timestamp']) and x['timestamp'][:10]<=days[-1]]
assert neutral(decisions)==expected_decisions,'Fresh-process decision suffix differs'
expected_equity=json.loads((folder/'equity.json').read_text())[:len(broker.timestamps)]
assert neutral(list(zip(broker.timestamps,broker.equity_curve)))==expected_equity,'Fresh-process equity differs'
assert source_generation_hash()==item['source_generation_hash'];after=origins(code,manifest)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:json.dump(dict(passed=True,fresh_process=True,label=a.label,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    through=days[-1],journal_rows=len(actual_journal),suffix_decisions=len(decisions),marks=len(broker.timestamps),source=item['source_generation_hash'],
    origins_before=before,origins_after=after,scope='Frozen research replay, pending and inventory restore through next session; live service restart not qualified'),f,indent=2,default=str)
print('Fresh-process continuation passed '+a.label+' '+item['period'])

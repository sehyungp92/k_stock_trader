"""Inherited native timing tests plus delayed decisions and unchanged children."""
from pathlib import Path
exec(compile(Path(__file__).with_name('kalcb_round3_native_unit_v1.py').read_text(),__file__,'exec'))
from backtests.strategies.kalcb.round3_execution_options_v1 import ExecutionOptions
probe=ExecutionOptions(Path(sys.argv[3]),delay_minutes=5)
probe.install()
try:
    a=Adapter();b=SimBroker(1e8,BrokerCosts(2,18,3))
    drive_timestamp_batch(b,(bar(6),),(a,))
    assert b.orders[0].metadata['decision_available_at']==bar(8).timestamp.isoformat()
    drive_timestamp_batch(b,(bar(7,close=100,low=94),),(a,))
    assert not b.fills
    drive_timestamp_batch(b,(bar(8,close=100,low=94),),(a,))
    assert len(b.fills)==2 and all(f.timestamp==bar(8).timestamp for f in b.fills)
    assert b.fills[1].metadata['execution_phase']=='intrabar' and probe.delayed==1
finally:probe.restore()
print(json.dumps(dict(passed=True,delay_minutes=5,ordinary_completed_decision_delayed=True,active_child_unchanged=True)))

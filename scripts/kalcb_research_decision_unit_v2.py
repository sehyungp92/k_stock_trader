"""C18 actual native timing and selector-boundary tests, no market outcomes."""
from pathlib import Path
import sys,json
exec(compile(Path(__file__).with_name('kalcb_round3_native_unit_v1.py').read_text(),__file__,'exec'))
from backtests.strategies.kalcb.research_decision_probe_v1 import DecisionStudyOptions
from strategy_common.actions import SubmitExit

checks=[]
for target in ['off','entry','exit']:
    probe=DecisionStudyOptions(Path(sys.argv[3]),target=target);probe.install()
    try:
        a=Adapter();b=SimBroker(1e8,BrokerCosts(2,18,3))
        drive_timestamp_batch(b,(bar(6),),(a,))
        expected=8 if target=='entry' else 7
        assert b.orders[0].metadata['decision_available_at']==bar(expected).timestamp.isoformat()
        drive_timestamp_batch(b,(bar(7,low=94),),(a,))
        if target=='entry':
            assert not b.fills
            drive_timestamp_batch(b,(bar(8,low=94),),(a,))
        assert len(b.fills)==2 and b.fills[1].metadata['execution_phase']=='intrabar'
        assert probe.delayed==(1 if target=='entry' else 0)
        checks.append(target+'_entry_and_protective_precedence')
    finally:probe.restore()

class ExitAdapter:
    snapshots={};frontier_shadow=None
    def __init__(self):self.entry=False;self.exit=False
    def on_bar(self,b,broker):
        if not self.entry:
            broker.submit(SubmitEntry('KALCB',b.symbol,10,'MARKET',None,95,'probe',
                {'fill_timing':'next_5m_open','hard_stop_enabled':False}),b.timestamp);self.entry=True
        elif broker.positions and not self.exit:
            broker.submit(SubmitExit('KALCB',b.symbol,10,'MARKET',None,'probe_exit',{}),b.timestamp);self.exit=True
        return []
    def on_fills(self,broker):return []

for target in ['off','exit']:
    probe=DecisionStudyOptions(Path(sys.argv[3]),target=target);probe.install()
    try:
        a=ExitAdapter();b=SimBroker(1e8,BrokerCosts(2,18,3))
        for i in range(6,10):drive_timestamp_batch(b,(bar(i,low=99),),(a,))
        assert len(b.fills)==2 and b.fills[0].timestamp==bar(7).timestamp
        assert b.fills[1].timestamp==bar(9 if target=='exit' else 8).timestamp
        checks.append(target+'_discretionary_exit_clock')
    finally:probe.restore()
try:DecisionStudyOptions(Path(sys.argv[3]),target='entry',delay_minutes=5)
except ValueError:checks.append('reject_stacked_delay')
else:raise AssertionError('Double delay accepted')
import importlib.util
contract_path=Path('E:/CodexResearch/k_stock_trader/kalcb_round3_20261017/attempt_2/round_run_v2/controller/files/backtests/strategies/kalcb/round3_contract_v1.py')
spec=importlib.util.spec_from_file_location('c16_receipt_check',contract_path)
contract_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract_module)
verify_acceptance=contract_module.verify_acceptance
try:verify_acceptance({'passed':False,'eligible_for_scoring':True})
except (ValueError,KeyError):checks.append('invalid_receipt_rejected')
else:raise AssertionError('Invalid evidence accepted')
print(json.dumps(dict(passed=True,C18_checks=checks)))

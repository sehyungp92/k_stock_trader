"""Exercise the actual shared orchestration with mocked diagnostic work only."""
from pathlib import Path
import importlib.util,json,time

ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1/timing_run_v2')
spec=importlib.util.spec_from_file_location('c18_controller',ROOT/'controller.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
out=ROOT/'framework_preflight';out.mkdir(exist_ok=False)
plugin=m.DiagnosticPlugin(ROOT/'registration.json',2)
seen=[]
def fake(name,callback):
    seen.append(name)
    plugin.records[name]={'diagnostic_only':True,'passed':True,'eligible_for_scoring':False,'return_pp':1e99}
plugin.evaluate=fake
try:plugin.canonicalize_mutations({**plugin.initial_mutations,'bad':True})
except ValueError:pass
else:raise AssertionError('Diagnostic mutation accepted')
batch=plugin.create_evaluate_batch(1,plugin.initial_mutations)
assert batch([m.Experiment('off',{})],plugin.initial_mutations)==[]
def invalid(name,callback):raise ValueError('Deliberately invalid native receipt')
plugin.evaluate=invalid
try:batch([m.Experiment('off',{})],plugin.initial_mutations)
except ValueError:pass
else:raise AssertionError('Invalid evaluation entered selection')
plugin.evaluate=fake;seen.clear()
manager=m.RoundManager('stock','kalcb_research_decision_preflight',base_dir=out/'framework')
runner=m.PhaseRunner(plugin,out/'framework/diagnostic',round_name='C18 interface unit, mocked work only',max_rounds=1,
    min_delta=0,max_retries=0,max_diagnostic_retries=0,round_manager=manager,round_num=1)
runner.run_all_phases()
assert set(seen)=={'off','entry','exit'}
states=list((out/'framework').rglob('phase_state.json'));assert states
for p in states:
    state=json.loads(p.read_text());assert state['cumulative_mutations']==plugin.initial_mutations
with (out/'receipt.json').open('x') as f:json.dump(dict(passed=True,financial_evaluations=0,
    checks=['mutations_rejected','invalid_receipt_propagated','arbitrary_high_diagnostic_return_withheld',
            'shared_two_phase_orchestration','retained_283_settings_unchanged'],seen=seen),f,indent=2)
print('C18 shared framework unit passed; no financial evaluations')

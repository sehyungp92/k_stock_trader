"""Synthetic shared-framework and registration checks; no market evaluations."""
from pathlib import Path
import importlib.util,json,hashlib
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1/labels_run_v2')
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('c18_labels',ROOT/'programs/kalcb_research_decision_labels_run_v2.py')
reg=m.read(ROOT/'registration.json');base=load('c18_base',reg['base_controller']['path'])
out=ROOT/'framework_preflight';out.mkdir(exist_ok=False)
m.write(out/'registration.json',dict(program=m.bound(__file__),source_registration=m.bound(ROOT/'registration.json'),market_evaluations=0))
plugin=m.install_plugin(base)(ROOT/'registration.json');seen=[]
def fake(name,callback):
    seen.append(name);plugin.records[name]=dict(passed=True,eligible_for_scoring=False,return_pp=1e99)
plugin.evaluate=fake
batch=plugin.create_evaluate_batch(1,plugin.initial_mutations)
assert batch([base.Experiment('pilot',{})],plugin.initial_mutations)==[]
def invalid(name,callback):raise ValueError('Deliberately invalid label receipt')
plugin.evaluate=invalid
try:batch([base.Experiment('pilot',{})],plugin.initial_mutations)
except ValueError:pass
else:raise AssertionError('Invalid evidence admitted')
try:plugin.canonicalize_mutations({**plugin.initial_mutations,'unregistered':1})
except ValueError:pass
else:raise AssertionError('Trading policy mutation allowed')
plugin.evaluate=fake;seen.clear()
manager=base.RoundManager('stock','kalcb_labels_preflight',base_dir=out/'framework')
runner=base.PhaseRunner(plugin,out/'framework/labels',round_name='C18 labels synthetic only',max_rounds=1,min_delta=0,
    max_retries=0,max_diagnostic_retries=0,round_manager=manager,round_num=1)
runner.run_all_phases();assert seen==['pilot','full']
for f in (out/'framework').rglob('phase_state.json'):assert m.read(f)['cumulative_mutations']==plugin.initial_mutations
for b in reg['programs']:m.verify(b)
m.write(out/'receipt.json',dict(passed=True,market_evaluations=0,checks=['actual_shared_two_phase_dispatch','invalid_evidence_propagates',
    'arbitrary_high_diagnostic_score_withheld','trading_mutations_rejected','frozen_programs_match','retained_settings_unchanged']))
print('C18 label framework synthetic unit passed',flush=True)

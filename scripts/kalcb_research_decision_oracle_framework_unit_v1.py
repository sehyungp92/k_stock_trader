"""Exercise actual three-phase C18 dispatch with synthetic receipts only."""
from pathlib import Path
import importlib.util,json
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1/oracle_run_v1')
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
    reg=json.loads((ROOT/'registration.json').read_text());m=load('c18_oracle_controller',reg['controller_program']['path']);base=load('c18_shared_base',reg['base_controller']['path'])
    out=ROOT/'framework_preflight';out.mkdir(exist_ok=False)
    m.write(out/'registration.json',dict(program=m.bound(__file__),controller_registration=m.bound(ROOT/'registration.json'),market_evaluations=0))
    plugin=m.plugin_class(base)(ROOT/'registration.json');seen=[]
    def fake(name,callback):seen.append(name);plugin.records[name]=dict(passed=True,eligible_for_scoring=False,return_pp=1e99)
    plugin.evaluate=fake;batch=plugin.create_evaluate_batch(1,plugin.initial_mutations)
    assert batch([base.Experiment('off',{})],plugin.initial_mutations)==[]
    def invalid(name,callback):raise ValueError('Deliberately invalid oracle receipt')
    plugin.evaluate=invalid
    try:batch([base.Experiment('off',{})],plugin.initial_mutations)
    except ValueError:pass
    else:raise AssertionError('Invalid receipt admitted')
    try:plugin.canonicalize_mutations({**plugin.initial_mutations,'unregistered':1})
    except ValueError:pass
    else:raise AssertionError('Trading mutation admitted')
    plugin.evaluate=fake;seen.clear()
    manager=base.RoundManager('stock','kalcb_oracle_preflight',base_dir=out/'framework')
    runner=base.PhaseRunner(plugin,out/'framework/oracles',round_name='C18 oracle synthetic dispatch',max_rounds=1,min_delta=0,
        max_retries=0,max_diagnostic_retries=0,round_manager=manager,round_num=1)
    runner.run_all_phases();assert seen==['off','smoke_S','smoke_E','smoke_SE','S','E','SE']
    for f in (out/'framework').rglob('phase_state.json'):assert m.read(f)['cumulative_mutations']==plugin.initial_mutations
    m.write(out/'receipt.json',dict(passed=True,market_evaluations=0,checks=['actual_shared_three_phase_dispatch','invalid_evidence_rejected','diagnostic_returns_not_selected','trading_mutations_rejected','retained_settings_unchanged']))
    print('C18 oracle framework synthetic checks passed',flush=True)
if __name__=='__main__':main()

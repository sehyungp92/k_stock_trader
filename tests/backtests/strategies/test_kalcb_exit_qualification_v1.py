from dataclasses import asdict,replace
from datetime import timedelta
import ast
from pathlib import Path
import pytest
from scripts.kalcb_exit_qualification_study_v1 import recipes
from backtests.strategies.kalcb.retention_research_io_v1 import read,write
from backtests.strategies.kalcb.research_process_v1 import completed_dependency
from backtests.strategies.kalcb.exit_qualification_plugin_v1 import ExitQualificationPlugin
from backtests.strategies.kalcb.fixed_allocation_reference_v1 import FixedAllocationReference
from backtests.strategies.kalcb.mechanism_observer_v2 import MechanismObserver
from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.auto.shared.round_manager import RoundManager


def test_finite_qualification_preserves_existing_neighbor_and_no_new_parameter_grid():
    plans={m:recipes(m) for m in ('compounding','identity','fixed')}
    assert {r['hold'] for r in plans['compounding'].values()}=={0,6,8}
    assert all(r['prior']==100 for p in plans.values() for r in p.values())
    assert sum(2*len(r['scenarios']) for p in plans.values() for r in p.values())==40
    reused=2*2*len(plans['compounding'])
    assert reused==12 and 40-reused==28
    assert len(plans['identity'])*2*2==8


def fixture_dependency(tmp_path):
    spec={'control':{'quantity':0},'other':{'quantity':.25}}
    rows=[dict(name=n,period=p,scenario=s,is_identity=n=='control',metrics={'audit_status':'passed'})
        for n in spec for p in ('is','oos') for s in ('baseline','combined')]
    # Complete design is checked independently of the configured identity count.
    write(tmp_path/'status.json',{'stage':'complete'});write(tmp_path/'protocol.json',{'recipes':spec})
    write(tmp_path/'progress.json',rows);write(tmp_path/'stage_completion.json',{'complete':True,'native_cases':8})
    return rows


def test_dependency_keeps_running_exact_process_even_with_stale_completion(monkeypatch,tmp_path):
    fixture_dependency(tmp_path)
    monkeypatch.setattr('backtests.strategies.kalcb.research_process_v1.exact_process',lambda _:True)
    assert not completed_dependency(tmp_path,{},8)


@pytest.mark.parametrize('poison',['duplicate','missing','audit','bridges'])
def test_dependency_rejects_unqualified_or_aliased_counts(monkeypatch,tmp_path,poison):
    rows=fixture_dependency(tmp_path)
    monkeypatch.setattr('backtests.strategies.kalcb.research_process_v1.exact_process',lambda _:False)
    if poison=='duplicate':rows[-1]=rows[0]
    if poison=='missing':rows.pop()
    if poison=='audit':rows[0]['metrics']['audit_status']='failed'
    write(tmp_path/'progress.json',rows)
    with pytest.raises(ValueError):completed_dependency(tmp_path,{},8)


def test_shared_framework_qualifies_all_cells_without_promoting_synthetic_winner(tmp_path):
    write(tmp_path/'study.json',dict(recipes=recipes('compounding'),mode='compounding',code=str(tmp_path),
        source_generation_hash='fixture',bound_inputs=[],source_programs=[]))
    calls=[]
    def native(name,period,scenario):
        calls.append((name,period,scenario));folder=tmp_path/'cases'/name/period/scenario
        r=dict(return_pct=1. if name=='control' else 100.,drawdown_pct=2.,trades=3,active_days=2,audit_status='passed')
        write(folder/'result.json',r);write(folder/'independent_account_verification.json',{'passed':True});return r
    plugin=ExitQualificationPlugin(tmp_path/'study.json',native)
    manager=RoundManager('stock',plugin.name,base_dir=tmp_path/'framework')
    PhaseRunner(plugin,manager.get_round_dir(1),round_name='Qualification fixture',round_manager=manager,
        round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
    assert len(calls)==len(set(calls))==24
    ledger=read(tmp_path/'qualification_ledger.json')
    assert ledger['study_complete'] is False
    assert all(r['metrics']['diagnostic_score']==0 for r in ledger['trials'])
    assert plugin.compute_final_metrics({'recipe':'control'})['is_baseline_return']==1.


def test_failed_account_cannot_be_recorded_as_candidate(tmp_path):
    write(tmp_path/'study.json',dict(recipes=recipes('fixed'),mode='fixed',code=str(tmp_path),
        source_generation_hash='fixture',bound_inputs=[],source_programs=[]))
    for p in ('is','oos'):
        for s in ('baseline','combined'):
            folder=tmp_path/'cases/prior100_failure8'/p/s
            write(folder/'result.json',dict(return_pct=100.,drawdown_pct=0.,trades=5,active_days=5))
            write(folder/'independent_account_verification.json',{'passed':False})
    plugin=ExitQualificationPlugin(tmp_path/'study.json',lambda *args:None)
    with pytest.raises(ValueError,match='Unverified account'):plugin.record('prior100_failure8')
    assert not (tmp_path/'cases/prior100_failure8/record.json').exists()


@pytest.mark.parametrize('enabled',[False,True])
def test_funded_overlay_and_observer_keep_real_gap_exit_and_cash(monkeypatch,enabled):
    from tests.backtests.strategies.test_kalcb_mechanism_v2 import setup
    from backtests.engine.sim_broker import SimBroker,BrokerCosts
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter
    from strategy_common.market import MarketBar
    from strategy_common.actions import SubmitEntry,SubmitExit
    from strategy_kalcb.core import logic
    from tests.strategy_kalcb.test_core_logic import _snapshot
    state,s,p,b,cfg=setup(monkeypatch,8);p.hold_bars=8
    cfg=replace(cfg,frontier_shadow_enabled=False)
    broker=SimBroker(1_000_000.,BrokerCosts())
    at=p.entry_time-timedelta(minutes=5)
    broker.process_bar(MarketBar('A',at,'5m',100,101,99,100,100000))
    broker.submit(SubmitEntry('KALCB','A',100,'MARKET',None,None,'entry'),at)
    broker.process_bar(MarketBar('A',p.entry_time,'5m',100,101,99,100,100000));p.entry_price=broker.fills[-1].price
    original_cash=broker.cash
    overlay=FixedAllocationReference(2_000_000.,enabled=enabled);observer=MechanismObserver(())
    overlay.install();observer.install()
    try:
        adapter=KALCBReplayAdapter(cfg,{b.timestamp.date():_snapshot(b.timestamp.date())},initial_equity=1_000_000.,costs=BrokerCosts());adapter.state=state
        view=adapter._portfolio_view(broker)
        assert broker.cash==original_cash
        assert view.equity==overlay.original(adapter,broker).equity
        exits=[a for a in logic._manage_position(state,s,b,cfg).actions if isinstance(a,SubmitExit)]
        assert len(exits)==1 and exits[0].reason=='opening_thesis_failure'
        broker.process_bar(b);broker.submit(exits[0],b.timestamp)
        assert observer.summarize([asdict(f) for f in broker.fills])['direct_filled_positions']==0
        broker.process_bar(MarketBar('A',b.timestamp+timedelta(minutes=5),'5m',80,81,79,80,100000))
        assert observer.summarize([asdict(f) for f in broker.fills])['direct_filled_positions']==1
        cash=1_000_000.
        for fill in broker.fills:cash+=(1 if fill.side=='SELL' else -1)*fill.qty*fill.price-fill.metadata['fee']
        assert broker.cash==pytest.approx(cash)
        assert broker.fills[-1].price<80
        assert overlay.evidence()['external_capital_flows']==0
    finally:observer.restore();overlay.restore()


def test_new_qualification_does_not_import_an_orchestration_runner():
    source=Path('scripts/kalcb_exit_qualification_study_v1.py').read_text(encoding='utf-8')
    imports=[n.module for n in ast.walk(ast.parse(source)) if isinstance(n,ast.ImportFrom)]
    assert not any(m and m.startswith('scripts.') for m in imports)
    assert 'backtests.auto.shared.phase_runner' in imports


def test_publisher_binds_distinct_identity_scenarios_and_rejects_changed_cache(tmp_path,monkeypatch):
    from tests.backtests.strategies.test_kalcb_fixed_allocation_packaging_v1 import fixture
    from backtests.strategies.kalcb.exit_allocation_io_v1 import freeze_allocation,OVERLAY,WORKER
    from backtests.strategies.kalcb.fixed_allocation_research_io_v1 import WORKER as OLD_WORKER
    out,old,programs=fixture(tmp_path,monkeypatch)
    programs[WORKER]=programs.pop(OLD_WORKER)
    write(out/'protocol.json',dict(runtime_root=str(tmp_path/'runtime'),registered='fixture'))
    monkeypatch.setattr('backtests.strategies.kalcb.exit_allocation_io_v1.source_hash',lambda p:'new')
    refs=[]
    for s in ('baseline','combined'):
        ref=tmp_path/s;refs.append(ref)
        for f in ('result.json','daily.json','equity.json','fills.json'):write(ref/f,{'scenario':s})
        write(ref/'independent_account_verification.json',{'passed':True})
    first=freeze_allocation(out,'control_baseline','is',old,programs,set(),mode='identity',bridge=refs[0])
    second=freeze_allocation(out,'control_combined','is',old,programs,set(),mode='identity',bridge=refs[1])
    assert first['folder']!=second['folder'] and first['policy']['sha256']==second['policy']['sha256']==old['policy']['sha256']
    assert freeze_allocation(out,'control_baseline','is',old,programs,set(),mode='identity',bridge=refs[0])==first
    with pytest.raises(ValueError,match='scenario bridge'):
        freeze_allocation(out,'control_baseline','is',old,programs,set(),mode='identity',bridge=refs[1])

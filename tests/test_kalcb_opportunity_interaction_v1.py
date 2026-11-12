"""Readiness: real frozen core interaction and complete saved-account interfaces."""
import json,os,subprocess,sys,shutil
from pathlib import Path
from dataclasses import replace
import pytest
from backtests.strategies.kalcb.opportunity_interaction_io_v1 import configuration
from scripts.kalcb_opportunity_interaction_v1 import declared_cases,activation,assess,bridge_gate
from backtests.strategies.kalcb.retention_research_io_v1 import read,write

BASE=Path('E:/CodexResearch/k_stock_trader')
REPO=Path(__file__).resolve().parents[1]


def test_only_declared_cap_changes_no_risk_leverage_or_lifecycle_mutation():
    p=read(BASE/'kalcb_retained_funding_20261016/v1/runtime/is/config.json')
    for cap in (0,2):
        new=configuration(p,cap)
        assert new['risk_max_gross_equity_pct']==cap and new['intraday_leverage']==p['intraday_leverage']
        assert {k:v for k,v in new.items() if k!='risk_max_gross_equity_pct'}=={k:v for k,v in p.items() if k!='risk_max_gross_equity_pct'}
    for cap in (1,3):
        with pytest.raises(ValueError):configuration(p,cap)


def test_matrix_has_all_four_interaction_stresses_and_gates(tmp_path):
    controls={a:{p+'/'+s:'x' for p in ('is','oos') for s in ('baseline','participation','cost','combined')} for a in ('cap','veto')}
    cases=declared_cases(controls)
    assert len(cases)==12 and sum(c['kind']=='bridge' for c in cases)==4
    assert len({c['id'] for c in cases})==12
    with pytest.raises(ValueError,match='All four'):bridge_gate(tmp_path,cases,[])


def frozen(function):
    code=BASE/'kalcb_retained_funding_20261016/v1/runtime/is/inputs/files'
    source='''import sys,runpy
from pathlib import Path
code,repo,test,function=sys.argv[1:]
sys.path[:0]=[code,repo]
import tests
tests.__path__=[str(Path(repo)/'tests')]
# Only this readiness module needs new orchestration imports; frozen core/runner
# remain first on their packages, and canonical core must never be exercised.
import scripts,backtests.strategies.kalcb
scripts.__path__.append(str(Path(repo)/'scripts'))
backtests.strategies.kalcb.__path__.append(str(Path(repo)/'backtests/strategies/kalcb'))
import strategy_kalcb.core.logic as logic
assert Path(logic.__file__).resolve().is_relative_to(Path(code).resolve())
runpy.run_path(test)[function]()
'''
    r=subprocess.run([sys.executable,'-B','-c',source,str(code),str(REPO),str(Path(__file__).resolve()),function],
        cwd=code,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
    assert r.returncode==0,r.stdout+'\n'+r.stderr


def test_actual_frozen_cap_and_veto_change_joint_pending_commitments():frozen('native_interaction')


def native_interaction():
    from copy import deepcopy
    from tests.test_kalcb_retained_funding_v1 import native_fixture
    from backtests.engine import replay
    from backtests.engine.sim_broker import SimBroker,BrokerCosts
    from backtests.engine.capacity import BarParticipationScenario
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter
    from backtests.strategies.kalcb.executed_value import snapshot_for
    from backtests.strategies.kalcb.funded_horizon_v1 import HorizonReplay
    row,bars,cfg=native_fixture()
    cfg=replace(cfg,intraday_leverage=2.,max_position_notional_pct=.6,risk_per_trade_pct=.5,heat_cap_r=100.,max_positions=8,max_per_sector=8)
    rows=[]
    for i in range(4):
        r=deepcopy(row);r['symbol']=f'{5930+i:06d}';r['candidate']['symbol']=r['symbol']
        r['candidate']['average_30m_volume']=100_000_000.
        r['candidate']['metadata']['action_quantity_scale']=1.;r['opportunity_id']=r['trade_date']+'/'+r['symbol']+'/q1'
        rows.append(r)
    signal=dict(day=row['trade_date'],origin='2025-01-01',available_at=row['feature_available_at'],predictions={'baseline':-30.,'combined':-40.})
    outputs={}
    for cap,mode in ((0,'disabled'),(2,'disabled'),(0,'veto'),(2,'veto')):
        config=replace(cfg,risk_max_gross_equity_pct=cap);costs=BrokerCosts(cfg.commission_bps,cfg.tax_bps_on_sell,3.)
        broker=SimBroker(100_000_000,costs,buying_power_leverage=2.,capacity_policy=BarParticipationScenario(.01,'baseline'))
        snap=snapshot_for(row['trade_date'],[(r,0) for r in rows],source='fixture')
        adapter=KALCBReplayAdapter(config,{snap.trade_date:snap},initial_equity=100_000_000,costs=costs)
        c=HorizonReplay(mode,{row['trade_date']+'/'+rows[0]['symbol']:signal});driver=replay.drive_timestamp_batch;c.install()
        try:
            for b in bars[:7]:replay.drive_timestamp_batch(broker,tuple(replace(b,symbol=r['symbol']) for r in rows),(adapter,))
        finally:c.restore()
        assert replay.drive_timestamp_batch is driver
        orders={r['order']['symbol']:r['order'] for r in broker.journal if r['event']=='submit' and r['order']['side']=='BUY'}
        outputs[(cap,mode)]=orders
        if mode=='veto':assert rows[0]['symbol'] not in orders
        if cap:
            assert sum(o['qty']*o['metadata']['entry_price_ref'] for o in orders.values())<=200_000_000
            for o in orders.values():
                m=o['metadata'];proof=m['pending_funding_reservation']
                assert o['qty']==proof['capped_qty']<=proof['original_qty']
                parent=outputs[(0,mode)][o['symbol']]['metadata']
                for k in ('entry_price_ref','stop_price','risk_per_share'):assert m[k]==parent[k]
    caponly=outputs[(2,'disabled')];joint=outputs[(2,'veto')]
    assert any(o['metadata']['pending_funding_reservation']['capped_qty']<o['metadata']['pending_funding_reservation']['original_qty'] for o in caponly.values()),{s:o['qty'] for s,o in caponly.items()}
    assert any(o['qty']>caponly.get(s,{}).get('qty',0) for s,o in joint.items())


def test_actual_saved_cap_activation_reconciles_orders_and_decisions(tmp_path):
    parent=BASE/'kalcb_late_entry_20261011/v28_pending_funding_qualification/cases/pending_100/oos/cost'
    for f in ('orders.jsonl.gz','decisions.jsonl.gz'):shutil.copyfile(parent/f,tmp_path/f)
    write(tmp_path/'horizon_actions.json',[])
    result=activation(tmp_path,dict(cap=2,mode='disabled'),{})
    assert result['passed'] and result['cap_decisions']>0 and result['trimmed']>0


def test_complete_real_account_assessment_and_trace_interface(tmp_path):
    prior=BASE/'holding_account_20261013/study_v2/cases/control/oos/baseline'
    veto=BASE/'kalcb_structural_preparation_20261016/funded_horizon_v1/cases/veto/oos/baseline'
    cap=BASE/'kalcb_late_entry_20261011/v25_pending_funding_screen/cases/pending_100/oos/baseline'
    case=dict(id='combined_oos_baseline',period='oos',scenario='baseline',cap=2,mode='veto',kind='treatment')
    # Copy only consumed account files into an isolated fixture, preserving inputs.
    dest=tmp_path/'cases'/case['id'];dest.mkdir(parents=True)
    for f in ('result.json','daily.json','equity.json','fills.json','trades.json','orders.jsonl.gz','decisions.jsonl.gz','independent_account_verification.json'):
        shutil.copyfile(veto/f,dest/f)
    write(tmp_path/'protocol.json',dict(controls={a:{'oos/baseline':str(f)} for a,f in [('prior',prior),('veto',veto),('cap',cap)]}))
    write(tmp_path/'completed_cases.json',[dict(case=case,verdict={'qualified':True})])
    assess(tmp_path);r=read(tmp_path/'assessment.json');v=r['cells']['oos/baseline']
    assert v['comparisons']['veto']['net_delta_krw']==0
    assert v['interaction_net_krw']==-v['cap_minus_prior']['net_delta_krw']
    trace=read(tmp_path/'traces/combined_oos_baseline_versus_veto.json')
    assert trace['changed_issuer_days']==0 and trace['first_order_divergence'] is None
    assert not r['criteria']['all_cells_qualified'] and not r['criteria']['economic_objective_achieved']

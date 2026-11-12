"""C13 readiness: frozen native composition and complete actual-account assessment."""
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
from scripts import kalcb_fixed_veto_v1 as r


def test_contract_matrix_preserves_capitals_and_only_registered_interaction():
    p=r.read(r.C10/'protocol.json')
    controls={a:{s:dict(folder='x') for s in r.SCENARIOS} for a in ('fixed','veto')}
    cases=r.cases(p['capitals'],controls)
    assert len(cases)==6 and len({c['id'] for c in cases})==6
    assert [(c['mode'],c['horizon_mode']) for c in cases[:2]]==[('fixed','disabled'),('compound','veto')]
    assert all(c['mode']=='fixed' and c['horizon_mode']=='veto' and c['initial_equity']==p['capitals'][c['scenario']] for c in cases[2:])


def test_rebinding_changes_only_declared_contract_context_and_bound_paths(tmp_path):
    item=r.read(r.C10/'registration.json')['capital_profiles']['bridge_veto']
    old=r.read(item['plan']['path']);context=r.read(old['audit_configuration'])
    r.write(tmp_path/'protocol.json',{'test':'binding'})
    case=dict(id='joint_baseline',scenario='baseline',initial_equity=175412823.94974506,mode='fixed',horizon_mode='veto',kind='treatment')
    p=r.plan_for(old,context,case,tmp_path,tmp_path/'plans/joint_baseline')
    assert p['capital_sensitivity']['reference']==100000000 and p['capital_sensitivity']['mode']=='fixed'
    assert {k:v for k,v in p.items() if k not in ('capital_sensitivity','audit_configuration','input_sha256')}=={k:v for k,v in old.items() if k not in ('capital_sensitivity','audit_configuration','input_sha256')}
    assert r.read(p['audit_configuration'])['initial_equity']==case['initial_equity']
    assert old['audit_configuration'] not in p['input_sha256']
    assert old['capital_sensitivity']['protocol']['path'] not in p['input_sha256']


def test_native_frozen_composition_changes_orders_without_mutating_cash_or_protection():
    code=r.C10/'capital/inputs/files'
    program='''import sys,runpy
from pathlib import Path
code,repo,test=sys.argv[1:]
sys.path[:0]=[code,repo]
import tests;tests.__path__=[str(Path(repo)/'tests')]
import scripts,backtests.strategies.kalcb
scripts.__path__.append(str(Path(repo)/'scripts'))
backtests.strategies.kalcb.__path__.append(str(Path(repo)/'backtests/strategies/kalcb'))
import strategy_kalcb.core.logic as logic
assert Path(logic.__file__).resolve().is_relative_to(Path(code).resolve())
runpy.run_path(test)['native_composition']()
'''
    result=subprocess.run([sys.executable,'-B','-c',program,str(code),str(r.REPO),str(Path(__file__).resolve())],cwd=code,capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    assert result.returncode==0,result.stdout+'\n'+result.stderr


def native_composition():
    from tests.test_kalcb_retained_funding_v1 import native_fixture
    from backtests.engine import replay
    from backtests.engine.sim_broker import SimBroker,BrokerCosts
    from backtests.engine.capacity import BarParticipationScenario
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter
    from backtests.strategies.kalcb.executed_value import snapshot_for
    from backtests.strategies.kalcb.fixed_allocation_reference_v1 import FixedAllocationReference
    from backtests.strategies.kalcb.funded_horizon_v1 import HorizonReplay
    row,bars,cfg=native_fixture()
    cfg=replace(cfg,intraday_leverage=2.,max_position_notional_pct=.6,risk_per_trade_pct=.1,heat_cap_r=100.,max_positions=8,max_per_sector=8,risk_max_gross_equity_pct=0.)
    rows=[]
    for i in range(2):
        x=deepcopy(row);x['symbol']=f'{5930+i:06d}';x['candidate']['symbol']=x['symbol'];x['candidate']['average_30m_volume']=100_000_000.
        x['candidate']['metadata']['action_quantity_scale']=1.;x['opportunity_id']=x['trade_date']+'/'+x['symbol']+'/q1';rows.append(x)
    signal=dict(day=row['trade_date'],origin='2025-01-01',available_at=row['feature_available_at'],predictions={'baseline':-30.,'combined':-40.})
    outputs={}
    for fixed,mode in ((False,'disabled'),(True,'disabled'),(False,'veto'),(True,'veto')):
        costs=BrokerCosts(cfg.commission_bps,cfg.tax_bps_on_sell,3.)
        broker=SimBroker(175_000_000.,costs,buying_power_leverage=2.,capacity_policy=BarParticipationScenario(.01,'baseline'))
        snap=snapshot_for(row['trade_date'],[(x,0) for x in rows],source='fixture')
        adapter=KALCBReplayAdapter(cfg,{snap.trade_date:snap},initial_equity=175_000_000.,costs=costs)
        allocation=FixedAllocationReference(100_000_000.,enabled=fixed);controller=HorizonReplay(mode,{row['trade_date']+'/'+rows[0]['symbol']:signal})
        original=KALCBReplayAdapter._portfolio_view;allocation.install();controller.install()
        try:
            for bar in bars[:7]:replay.drive_timestamp_batch(broker,tuple(replace(bar,symbol=x['symbol']) for x in rows),(adapter,))
        finally:controller.restore();allocation.restore()
        assert KALCBReplayAdapter._portfolio_view is original
        orders={j['order']['symbol']:j['order'] for j in broker.journal if j['event']=='submit' and j['order']['side']=='BUY'}
        outputs[(fixed,mode)]=orders
        assert allocation.rows[0]['actual_equity']==175_000_000.
        assert allocation.evidence()['external_capital_flows']==allocation.evidence()['ledger_cash_mutations']==0
        if mode=='veto':assert rows[0]['symbol'] not in orders
    symbol=rows[1]['symbol']
    assert outputs[(True,'veto')][symbol]['qty']<outputs[(False,'veto')][symbol]['qty']
    for key in ('entry_price_ref','stop_price','risk_per_share'):
        assert outputs[(True,'veto')][symbol]['metadata'][key]==outputs[(False,'veto')][symbol]['metadata'][key]


def test_full_assessor_real_accounts_and_unqualified_control_exclusion(tmp_path):
    from backtests.strategies.kalcb.retention_research_io_v1 import link
    a=r.read(r.C6/'assessment.json');p=r.read(r.C10/'protocol.json');controls={arm:{} for arm in ('prior','fixed','veto')};ledger=[]
    for s in r.SCENARIOS:
        fixed=a['case_ledger']['fixed_'+s];prior=a['case_ledger']['compound_'+s]
        controls['fixed'][s]=dict(folder=fixed['folder'],qualified=True)
        for arm in ('prior','veto'):controls[arm][s]=dict(folder=prior['folder'],qualified=prior['qualified'],verdict={'qualified':prior['qualified']})
        case=dict(id='joint_'+s,scenario=s,initial_equity=p['capitals'][s],kind='treatment')
        ledger.append(dict(case=case,verdict={'qualified':True}))
        for name in ('result.json','daily.json','equity.json','fills.json','trades.json','orders.jsonl.gz','decisions.jsonl.gz','independent_account_verification.json'):
            link(Path(fixed['folder'])/name,tmp_path/'cases'/case['id']/name)
    r.write(tmp_path/'protocol.json',dict(controls=controls));r.write(tmp_path/'completed_cases.json',ledger)
    r.assess(tmp_path);out=r.read(tmp_path/'assessment.json')
    assert len(out['cells'])==4 and out['criteria']['median_net_vs_fixed']==0
    assert set(out['cells']['participation']['comparisons'])=={'fixed'}
    assert set(out['cells']['participation']['unqualified_controls'])=={'prior','veto'}
    assert not out['criteria']['economic_objective_achieved']

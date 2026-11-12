"""C15 real frozen-core timing, order and account interface readiness."""
from pathlib import Path
import json,os,subprocess,sys
import pytest
REPO=Path(__file__).resolve().parents[1]
C=Path('E:/CodexResearch/k_stock_trader/kalcb_structural_preparation_20261016/funded_horizon_v1')


def frozen(function,*args):
    code=C/'runtime/is/inputs/files'
    program='''import sys,runpy,json,importlib.util
from pathlib import Path
code,repo,test,function,args=sys.argv[1:]
sys.path[:0]=[code,repo]
import tests;tests.__path__=[str(Path(repo)/'tests')]
name='backtests.strategies.kalcb.reconsideration_v1'
spec=importlib.util.spec_from_file_location(name,Path(repo)/'backtests/strategies/kalcb/reconsideration_v1.py')
module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
import strategy_kalcb.core.logic as logic
assert Path(logic.__file__).resolve().is_relative_to(Path(code).resolve())
import backtests.strategies.kalcb
backtests.strategies.kalcb.__path__.append(str(Path(repo)/'backtests/strategies/kalcb'))
runpy.run_path(test)[function](*json.loads(args))
'''
    result=subprocess.run([sys.executable,'-B','-c',program,str(code),str(REPO),str(Path(__file__).resolve()),function,json.dumps(args)],
        cwd=code,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
    assert result.returncode==0,result.stdout+'\n'+result.stderr


@pytest.mark.parametrize('case',['disabled','confirmed','unconfirmed','positive_signal','pending_lapse','missing_prefix'])
def test_real_native_lifecycle(case):frozen('native_lifecycle',case)


def native_lifecycle(case):
    from dataclasses import replace
    from tests.test_kalcb_retained_funding_v1 import native_fixture
    from backtests.strategies.kalcb.reconsideration_v1 import ReconsiderationReplay
    from backtests.strategies.kalcb.funded_horizon_v1 import HorizonReplay
    from backtests.engine import replay
    from backtests.engine.sim_broker import SimBroker,BrokerCosts
    from backtests.engine.capacity import BarParticipationScenario
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter
    from backtests.strategies.kalcb.executed_value import snapshot_for
    from strategy_kalcb.core import logic,account_allocation_v1
    row,bars,cfg=native_fixture();high=max(b.high for b in bars[:6])
    cfg=replace(cfg,exit_stop_mode='fixed_pct',exit_stop_pct=.0045,entry_require_complete_prefix=True)
    price=high+2
    bars[6]=replace(bars[6],open=high,high=price+1,low=high-1,close=price if case!='unconfirmed' else high)
    if case=='pending_lapse':
        # Native opening capacity uses the previous completed interval, not future realised volume.
        bars[6]=replace(bars[6],volume=100)
        bars[7]=replace(bars[7],open=price,high=price+1,low=high-1,close=high,volume=100)
    if case=='missing_prefix':bars.pop(3)
    signal=dict(day=row['trade_date'],origin='2025-01-01',available_at=row['feature_available_at'],
                predictions={'baseline':-30.,'combined':-40.})
    if case=='positive_signal':signal['predictions']={'baseline':1.,'combined':1.}
    signals={row['trade_date']+'/'+row['symbol']:signal}
    def run(controller):
        costs=BrokerCosts(cfg.commission_bps,cfg.tax_bps_on_sell,3.)
        broker=SimBroker(100_000_000.,costs,buying_power_leverage=cfg.intraday_leverage,capacity_policy=BarParticipationScenario(.01,'baseline'))
        snapshot=snapshot_for(row['trade_date'],[(row,0)],source='c15-fixture')
        adapter=KALCBReplayAdapter(cfg,{snapshot.trade_date:snapshot},initial_equity=100_000_000.,costs=costs)
        driver=replay.drive_timestamp_batch;quantity=account_allocation_v1.allocate_quantity;evaluate=logic._evaluate_entry
        controller.install()
        try:
            for b in bars:replay.drive_timestamp_batch(broker,(b,),(adapter,))
        finally:controller.restore()
        assert replay.drive_timestamp_batch is driver and account_allocation_v1.allocate_quantity is quantity and logic._evaluate_entry is evaluate
        return broker,adapter
    controller=ReconsiderationReplay('disabled' if case=='disabled' else 'reconsider',signals)
    broker,adapter=run(controller)
    orders=[x['order'] for x in broker.journal if x['event']=='submit' and x['order']['side']=='BUY']
    if case=='disabled':
        control,_=run(HorizonReplay('veto',signals))
        assert broker.journal==control.journal and broker.fills==control.fills and not controller.renewals
    elif case in ('confirmed','pending_lapse'):
        assert len(orders)==1,controller.renewals
        order=orders[0];meta=order['metadata']
        assert meta['decision_available_at'].endswith('09:35:00+09:00')
        assert meta['signal_bar'].endswith('09:30:00+09:00')
        assert meta['entry_price_ref']==price and meta['research_reconsideration']['opening_high']==high
        assert meta['stop_price']==logic.round_price_for_krx(price*(1-.0045),'protective_stop')
        assert meta['risk_per_share']==max(price-meta['stop_price'],1.)
        assert len([x for x in controller.renewals if x['kind']=='renewal_considered'])==1
        assert all(f.timestamp.isoformat()>=meta['decision_available_at'] for f in broker.fills)
        if case=='pending_lapse':
            assert any(x['kind']=='renewal_pending_cancel' for x in controller.renewals),str([
                (x['event'],x.get('timestamp'),x.get('reason'),x.get('order',{}).get('side'),x.get('order',{}).get('qty'))
                for x in broker.journal if x['event'] not in ('opening_marks','closing_marks')])
            assert any(x['event']=='cancel' for x in broker.journal)
        assert not broker.positions and not broker.orders
    elif case=='unconfirmed':assert not orders and not any(x.get('confirmed') for x in controller.renewals)
    elif case=='positive_signal':
        control,_=run(HorizonReplay('veto',signals))
        assert broker.journal==control.journal and not controller.renewals
    else:assert not orders


def test_six_cases_no_parameter_search():
    from scripts import kalcb_reconsideration_v1 as r
    controls={p+'/'+s:'control' for p in ('is','oos') for s in ('baseline','combined')}
    cases=r.cases(controls)
    assert len(cases)==6 and sum(c['mode']=='disabled' for c in cases)==2
    assert {c['id'] for c in cases[2:]}=={'reconsider_'+p+'_'+s for p in ('is','oos') for s in ('baseline','combined')}


def test_complete_real_account_assessor_and_validator(tmp_path):
    from scripts import kalcb_reconsideration_v1 as r
    from backtests.strategies.kalcb.retention_research_io_v1 import link
    controls={p+'/'+s:str(r.C/'cases/veto'/p/s) for p in ('is','oos') for s in ('baseline','combined')}
    ledger=[]
    for c in r.cases(controls)[2:]:
        source=Path(controls[c['period']+'/'+c['scenario']]);item=r.read(source/'input_profile.json')
        assert r.classify_standard(source,item,c['scenario'])['qualified']
        assert r.diag.funding_review(source)['verified']
        for name in ('result.json','daily.json','equity.json','fills.json','trades.json','orders.jsonl.gz','decisions.jsonl.gz','independent_account_verification.json'):
            link(source/name,tmp_path/'cases'/c['id']/name)
        ledger.append(dict(case=c,verdict={'qualified':True}))
    r.write(tmp_path/'protocol.json',dict(controls=controls,prior_controls=controls));r.write(tmp_path/'completed_cases.json',ledger)
    r.assess(tmp_path);out=r.read(tmp_path/'assessment.json')
    assert len(out['cells'])==4 and all(v['net_delta_krw']==0 for v in out['cells'].values())
    assert not out['economic_objective_achieved'] and not out['principal_qualification']

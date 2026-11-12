from copy import deepcopy
from datetime import datetime,timedelta
from pathlib import Path
from types import SimpleNamespace,ModuleType
import gzip,json,sqlite3
import pytest
from strategy_olr.config import OLRConfig
from strategy_olr.core.state import OLRState
from strategy_common.market import MarketBar
from strategy_olr.core.liquidity_v6 import CausalLiquidityPanel
from strategy_olr.core.liquidity_runtime_v6 import KEY,day_context,override_forecast,observed_prefix
from scripts.olr_structural_patches_v6 import patch

def fixture():
    days=['2026-01-05','2026-01-06','2026-01-07','2026-01-08']
    rows=[dict(day=d,symbol='000100',prefix_volume=10000,prefix_rows=66,prefix_complete=True,
        prefix_available_at=d+'T14:30:00+09:00',auction_volume=2000,
        auction_available_at=d+'T15:35:00+09:00',auction_disposition='source_qualified_auction',source_hash='fixture') for d in days]
    panel=CausalLiquidityPanel(rows,days,['000100'])
    bar=MarketBar('000100',datetime.fromisoformat(days[-1]+'T14:30:00+09:00'),'5m',100.,101.,99.,100.,10000.)
    cfg=OLRConfig(capacity_exit_quantile=.5)
    state=OLRState();state.meta[KEY]=day_context(panel,days[-1],'panel-hash')
    return panel,bar,cfg,state

def test_policy_observation_feedback_removed_at_request_boundary():
    panel,bar,cfg,a=fixture();b=deepcopy(a)
    a.meta['capacity_history_v1']={'000100':{'auction':[], 'continuous':[]}}
    b.meta['capacity_history_v1']={'000100':{'auction':[['2026-01-07T15:30:00+09:00',999999]],'continuous':[]}}
    x=override_forecast(a,bar,cfg,{},[]);z=override_forecast(b,bar,cfg,{},[])
    assert x==z and x[1]==int(2000*cfg.capacity_participation)

def test_entry_limit_remains_binding():
    _,bar,cfg,state=fixture()
    result=override_forecast(state,bar,cfg,{'entry':{'quantity':3}},[3])
    assert result[0]==3 and result[1]>3

def test_stale_or_wrong_identity_rejected():
    _,bar,cfg,state=fixture();state.meta[KEY]['forecasts']['000100']['symbol']='000200'
    with pytest.raises(ValueError):override_forecast(state,bar,cfg,{},[])

def test_original_capacity_has_exact_off_parity_after_patch():
    import strategy_olr.core.capacity as old
    source=Path(old.__file__).read_text(encoding='utf-8')
    module=ModuleType('strategy_olr.core.test_capacity_v6');module.__package__='strategy_olr.core'
    if 'override_forecast(state, bar, config, evidence, limits)' not in source:
        source=patch('strategy_olr/core/capacity.py',source)
    exec(compile(source,'patched_capacity.py','exec'),module.__dict__)
    _,bar,cfg,state=fixture();state.meta.pop(KEY)
    for n in range(5):
        state.meta['capacity_history_v1']={'000100':dict(auction=[[(bar.timestamp-timedelta(days=i+1)).isoformat(),i*2000+500] for i in range(n)],continuous=[])}
        assert old.forecast(deepcopy(state),bar,cfg)==module.forecast(deepcopy(state),bar,cfg)
    panel,bar,cfg,state=fixture()
    assert module.forecast(state,bar,cfg)[1]==20

def test_actual_prefix_restores_without_future_rows():
    panel,bar,_,_=fixture();rows=observed_prefix(panel,'2026-01-07')
    assert len(rows)==3 and all(r['day']<'2026-01-08' for r in rows)
    assert all('source_hash' not in r for r in rows)

def test_independent_request_audit_detects_quantity_tampering(tmp_path):
    from backtests.engine.olr_liquidity_audit_v6 import audit_requests
    panel,bar,cfg,state=fixture();_,qty,ev=override_forecast(state,bar,cfg,{'asof':(bar.timestamp+timedelta(minutes=5)).isoformat()},[])
    data={'rows':list(panel.rows.values())};p=tmp_path/'decisions.gz'
    def save():
        with gzip.open(p,'wt',encoding='utf-8') as f:f.write(json.dumps({'metadata':{'capacity_forecast':ev,'capacity_approved_qty':qty}})+'\n')
    save();assert audit_requests(p,data,'broad_median','panel-hash',cfg.capacity_participation)['passed']
    ev['exit']['quantity']+=1;save()
    with pytest.raises(ValueError):audit_requests(p,data,'broad_median','panel-hash',cfg.capacity_participation)

def test_source_projection_checks_prefix_and_original_closing_clock():
    from backtests.engine.olr_liquidity_audit_v6 import verify_projection
    from kis_core.trading_calendar import get_trading_calendar
    day='2026-01-08';names=[str(i).zfill(6) for i in range(103)]
    db=sqlite3.connect(':memory:');db.execute('CREATE TABLE bars(symbol TEXT,day TEXT,ts TEXT,o REAL,h REAL,l REAL,c REAL,v REAL)')
    rows=[]
    for symbol in names:
        at=datetime.fromisoformat(day+'T09:00:00+09:00')
        for i in range(66):
            db.execute('INSERT INTO bars VALUES(?,?,?,?,?,?,?,?)',(symbol,day,(at+timedelta(minutes=5*i)).isoformat(),100,101,99,100,1000))
        db.execute('INSERT INTO bars VALUES(?,?,?,?,?,?,?,?)',(symbol,day,day+'T15:30:00+09:00',100,100,100,100,2000))
        rows.append(dict(day=day,symbol=symbol,prefix_volume=66000,prefix_rows=66,prefix_complete=True,
            prefix_available_at=day+'T14:30:00+09:00',auction_volume=2000,auction_available_at=day+'T15:35:00+09:00',
            auction_disposition='source_qualified_auction',source_hash='fixture'))
    panel=dict(rows=rows,sessions=[day],universe=names,coarse_database_sha256='fixture')
    assert verify_projection(db,panel,'fixture')['rows_recomputed']==103
    panel['rows'][0]['auction_available_at']=day+'T15:30:00+09:00'
    with pytest.raises(ValueError):verify_projection(db,panel,'fixture')

"""Independent replay-boundary proof using authenticated source observations."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from datetime import date,datetime,time,timedelta


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main(root,window):
    protocol=read(root/'protocol.json');info=read(Path(protocol['generation_directories'][window])/window/'ready.json')
    files=Path(info['files']);sys.path.insert(0,str(files))
    from dataclasses import replace
    from strategy_olr.config import OLRConfig
    from strategy_olr.core.state import OLRState
    from strategy_olr.core.capacity import seed_history,observe,forecast,phase
    from strategy_common.market import MarketBar
    from strategy_common.clock import KST
    from backtests.auto.shared.cache_keys import stable_signature
    request=read(files/info['study']['olr_request']);base=read(root/'candidate_plan.json')['baseline']
    cfg=OLRConfig.from_mapping(request['config'],{k:v for k,v in base.items() if not k.startswith('capacity_')})
    overrides={k:v for k,v in base.items() if k.startswith('capacity_')};overrides['capacity_broad_auction_history']=True
    cfg=replace(cfg,**overrides)
    snapshots=read(files/info['study']['olr_snapshots']);asof=date.fromisoformat(info['study']['configuration']['date_range']['start'])
    cutoff=datetime.combine(asof,time(0),tzinfo=KST)
    database=files/info['study']['database'];assert digest(database)==info['study']['database_hash']
    with sqlite3.connect('file:'+database.as_posix()+'?mode=ro',uri=True) as connection:
        dates=[r[0] for r in connection.execute('SELECT DISTINCT day FROM bars WHERE day<? ORDER BY day',(str(asof),))]
        rows=connection.execute('SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day<? AND o=h AND h=l AND l=c ORDER BY ts,symbol',(str(asof),))
        bars=[MarketBar(s,datetime.fromisoformat(t),'5m',*values,source='kis_krx_parquet',source_fingerprint=info['study']['database_hash']) for s,t,*values in rows]
        import yaml
        u=Path(info['study']['configuration']['universe'])
        u=next(p for p in (files/u,files/'config'/u) if p.exists())
        manifest=read(info['study']['manifest'])
        assert digest(u)==manifest['files'][u.relative_to(files).as_posix()]
        declared={str(v).zfill(6) for v in yaml.safe_load(u.read_text())['symbols']}
        auctions=[b for b in bars if b.symbol in declared and phase(b)=='auction']
    def state():return OLRState(snapshot_hash='validation',source_fingerprint=info['study']['database_hash'])
    uninterrupted=state();seeded=state();split=state()
    for b in auctions:observe(uninterrupted,b,cfg)
    seed_history(seeded,auctions,cfg,cutoff)
    # A genuine split before IS: warm-start once, then stream all IS sessions.
    is_cutoff=datetime(2025,8,7,tzinfo=KST)
    prefix=[b for b in auctions if b.timestamp<is_cutoff]
    seed_history(split,prefix,cfg,is_cutoff)
    for b in auctions:
        if b.timestamp>=is_cutoff:observe(split,b,cfg)
    expected={}
    for b in auctions:expected.setdefault(b.symbol,[]).append([b.timestamp.isoformat(),float(b.volume)])
    expected={k:{'auction':v[-20:],'continuous':[]} for k,v in expected.items()}
    assert expected==uninterrupted.meta['capacity_history_v1']==seeded.meta['capacity_history_v1']==split.meta['capacity_history_v1']
    # Repeated restore must be idempotent.
    seed_history(seeded,auctions,cfg,cutoff);assert seeded.meta['capacity_history_v1']==expected
    available=0
    for symbol in expected:
        probe=MarketBar(symbol,datetime.combine(asof,time(14,30),tzinfo=KST),'5m',100,101,99,100,10000,source_fingerprint='validation')
        one=forecast(uninterrupted,probe,cfg);two=forecast(seeded,probe,cfg)
        assert one==two==forecast(split,probe,cfg)
        samples=one[2]['exit']['samples'];assert all(date.fromisoformat(t[:10])<asof for t,_ in samples)
        valid=sorted(v for t,v in expected[symbol]['auction'] if datetime.fromisoformat(t)>=probe.timestamp-timedelta(days=60))
        if len(valid)>=cfg.capacity_min_exit_observations:
            from math import floor
            p=(len(valid)-1)*(cfg.capacity_exit_quantile or .5);lo=floor(p)
            assert one[1]==floor((valid[lo]+(valid[min(lo+1,len(valid)-1)]-valid[lo])*(p-lo))*cfg.capacity_participation)
            available+=1
    # Future-day and missing-provenance observations must fail closed.
    for bad in (replace(auctions[-1],timestamp=cutoff.replace(hour=15,minute=30)),replace(auctions[-1],source_fingerprint='')):
        try:seed_history(state(),[bad],cfg,cutoff)
        except ValueError:pass
        else:raise AssertionError('Invalid seed accepted')
    original=read(Path(__file__).resolve().parents[1]/'data/backtests/analysis/olr_round1_oos_review_20261010/evidence'/window/'scored_snapshots.json')
    economics=lambda row:[{k:v for k,v in c.items() if k not in ('metadata','source_fingerprint','source_artifact_hash')} for c in row['candidates']]
    old={r['trade_date']:economics(r) for r in original}
    assert old=={r['trade_date']:economics(r) for r in snapshots if r['trade_date'] in old}
    result={'status':'passed','window':window,'asof':str(asof),'database_sha256':info['study']['database_hash'],
        'validator_sha256':digest(__file__),'source_generation_hash':info['study']['source_generation_hash'],
        'prior_sessions':len(dates),'prior_first':dates[0],'prior_last':dates[-1],
        'is_sessions_in_prior':sum('2025-08-07'<=d<='2026-03-31' for d in dates),
        'prior_auction_rows':len(auctions),'symbols':len(expected),'symbols_with_usable_exit_forecast':available,
        'seeded_state_hash':stable_signature(expected),'seeded_state':expected,
        'full_stream_equals_august_restart_equals_scoring_restart':True,'idempotent_restore':True,
        'future_seed_rejected':True,'unsourced_seed_rejected':True,'default_feature_parity_sessions':len(old),
        'observation_policy':'Declared source-qualified universe completed auctions; 20 last events per stock; native 60-calendar-day forecast window.',
        'account_boundary':'Starting cash and portfolio are independently reset; market history is continuous.'}
    (root/f'boundary_validation_{window}.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print({k:v for k,v in result.items() if k!='seeded_state'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--window',required=True);a=p.parse_args();main(a.root,a.window)

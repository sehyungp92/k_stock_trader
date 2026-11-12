"""Causal boundary and backward-parity checks for the optional history panel."""
from __future__ import annotations
import argparse
from datetime import date,datetime,timedelta
import hashlib
import json
import importlib.util
from pathlib import Path
import sys


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main(root,window,repo):
    info=read(root/'protocol.json');generation=Path(info['generation_directories'][window])/window
    ready=read(generation/'ready.json');files=Path(ready['files']);sys.path.insert(0,str(files))
    spec=importlib.util.spec_from_file_location('olr_panel_validation_streaming',repo/'scripts/olr_snapshot_streaming.py')
    streaming=importlib.util.module_from_spec(spec);spec.loader.exec_module(streaming)
    helper,read_payload,native_panel_cache=streaming.helper,streaming.read_payload,streaming.native_panel_cache
    from backtests.strategies.olr import research_sweep as native
    from strategy_olr import research
    from strategy_olr.config import OLRConfig
    from strategy_common.sector_intraday import build_sector_intraday_panel,AFTERNOON_CUTOFF
    from strategy_common.clock import KST
    from strategy_common.market import MarketBar
    from strategy_olr.models import OLRDailySnapshot,OLRDailyCandidate
    from types import SimpleNamespace
    h=helper();source=read(root/'cache_sources'/window/'source.json');directory=Path(source['cache'])
    index=read(directory/'index.json');assert h.digest(directory/'index.json')==source['index_sha256']
    static=read_payload(directory/'static.pickle',index['files']['static.pickle'],h.digest)
    cfg=OLRConfig.from_mapping(read(files/ready['study']['olr_request'])['config'],read(root/'candidate_plan.json')['baseline'])
    days=tuple(date.fromisoformat(x) for x in index['identity']['base']['context_dates'])
    dataset=SimpleNamespace(config=read(files/ready['study']['olr_request'])['config'],trading_dates=days,sector_map=static['sectors'],sector_intraday_panel_1430=static['panel'])
    original=read(repo/'data/backtests/analysis/olr_round1_oos_review_20261010/evidence'/window/'scored_snapshots.json')
    old={r['trade_date']:r for r in original}
    economic=lambda snap:[{k:v for k,v in c.items() if k not in ('metadata','source_fingerprint','source_artifact_hash')} for c in snap['candidates']]
    changes=[];checked=0;actual=[]
    corrected={r['trade_date']:r for r in read(files/ready['study']['olr_snapshots'])}
    for ds in old:
        d=date.fromisoformat(ds);name=ds+'.pickle'
        snap,daily_panel,current_panel,bars=read_payload(directory/name,index['files'][name],h.digest)
        with native_panel_cache(research,d,snap,daily_panel,current_panel,static['sectors'],AFTERNOON_CUTOFF):
            morning=native.snapshots_for_experiment(dataset,read(root/'candidate_plan.json')['baseline'],research_snapshots={d:snap})
            unchanged=research.afternoon_selection_from_snapshot(morning[d],bars,cfg,static['sectors'])
            if economic(unchanged.to_json_dict())!=economic(old[ds]):changes.append(ds)
            contexts=research.build_afternoon_contexts(morning[d],bars,cfg,static['sectors'],sector_intraday_panel=static['panel'])
            for c in corrected[ds]['candidates']:
                reference=static['panel'].feature_for(d,c['symbol'],sector=c['sector'])
                assert contexts[c['symbol']].intraday_sector_rel_volume==reference.rel_volume
                actual.append({'day':ds,'symbol':c['symbol'],'relative_volume':reference.rel_volume})
        checked+=1
    assert not changes,changes
    # Direct formula test: old-session volume affects the current feature;
    # observations at/after cutoff and later dates cannot change it.
    first=date(2026,3,2);target=date(2026,4,1);future=target+timedelta(days=1)
    symbol='005930';sector={symbol:'SEMICONDUCTORS'}
    prior=[first+timedelta(days=i) for i in range(25)]
    def bars(d,volume):
        return tuple(MarketBar(symbol,datetime(d.year,d.month,d.day,9,0,tzinfo=KST)+timedelta(minutes=5*i),'5m',100.,102.,99.,101.,volume,is_completed=True) for i in range(66))
    source_bars={(d,symbol):bars(d,100.) for d in prior};source_bars[target,symbol]=bars(target,200.)
    panel=build_sector_intraday_panel(source_bars,sector,trade_dates=tuple(prior+[target]),cutoff=AFTERNOON_CUTOFF)
    value=panel.feature_for(target,symbol,sector=sector[symbol])
    assert value.member_score!=0
    # Single-member sector shrinkage can dilute sector relative volume. The
    # member's exact 2x volume is checked via the public panel constructor's
    # internal volume normalization as well as feature invariance.
    from strategy_common.sector_intraday import _relative_cutoff_volume_map
    volumes={(d,symbol):6600. for d in prior};volumes[target,symbol]=13200.
    assert _relative_cutoff_volume_map(volumes)[target,symbol]==2.
    contaminated=dict(source_bars);contaminated[target,symbol]=source_bars[target,symbol]+(MarketBar(symbol,datetime(2026,4,1,14,30,tzinfo=KST),'5m',100.,2000.,1.,1900.,1e12,is_completed=True),)
    contaminated[future,symbol]=bars(future,1e10)
    after=build_sector_intraday_panel(contaminated,sector,trade_dates=tuple(prior+[target,future]),cutoff=AFTERNOON_CUTOFF)
    assert after.feature_for(target,symbol,sector=sector[symbol])==value
    out={'status':'passed','window':window,'default_behavior_economic_parity_sessions':checked,'default_behavior_changes':changes,
        'corrected_selected_feature_checks':len(actual),'corrected_selected_nonunit_relative_volumes':sum(abs(x['relative_volume']-1)>1e-12 for x in actual),
        'future_and_post_cutoff_invariance':True,'prior_volume_normalization_2x_pass':True,'actual_features':actual}
    (root/f'feature_validation_{window}.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print({k:v for k,v in out.items() if k!='actual_features'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--window',required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args();main(a.root,a.window,a.repo)

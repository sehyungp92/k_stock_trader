"""Build frozen-rule OLR snapshots using prior daily and completed afternoon bars."""
from collections import defaultdict
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from scripts.portfolio_targeted_validation import ROOT,read,bound
from backtests.strategies.kalcb.executed_value import write_json,digest


def build(manifest,database,request,days):
    import yaml
    from backtests.core.input_manifest import InputManifest
    from backtests.strategies.olr import research_sweep as native
    from strategy_olr.config import OLRConfig
    from strategy_olr.research import afternoon_selection_from_snapshot
    from strategy_common.sector_intraday import build_sector_intraday_panel,AFTERNOON_CUTOFF,load_canonical_sector_map
    from strategy_common.source_quality import complete_feature_bars
    from scripts.kalcb_executed_value_study import connect,load_day
    generation=InputManifest(manifest);contract=generation.payload['evidence']['shared_market_inputs']
    cfg=OLRConfig.from_mapping(request['config'],request['mutations'])
    end=max(days);source=digest(database)
    with generation.activate():
        symbols=tuple(str(s).zfill(6) for s in yaml.safe_load(generation.read(generation.root/contract['universe']))['symbols'])
        root=generation.root/contract['daily_data_root']
        daily=native._load_daily_rows(root,symbols,end);flow=native._load_flow_rows(root,symbols,end)
        foreign=native._load_foreign_flow_rows(root,symbols,end);inst=native._load_institutional_flow_rows(root,symbols,end)
        indices=native._load_index_rows(root,end)
        sectors=load_canonical_sector_map({'sector_map_path':str(generation.root/contract['sector_map_path'])})
    conn=connect({'database':str(database)});bars=defaultdict(list)
    for day in days:
        for bar in load_day(conn,str(day),source,set(symbols)):bars[day,bar.symbol].append(bar)
    conn.close();bars={k:tuple(v) for k,v in bars.items()}
    feature_bars, exclusions = (complete_feature_bars({(d,s):bars.get((d,s),()) for d in days for s in symbols},AFTERNOON_CUTOFF)
                                if cfg.require_complete_feature_prefix else (bars, []))
    panel=build_sector_intraday_panel(feature_bars,sectors,trade_dates=tuple(days),cutoff=AFTERNOON_CUTOFF,symbols=symbols)
    dataset=SimpleNamespace(config=request['config'],source_fingerprint=source,requested_symbols=symbols,symbols=symbols,
        excluded_symbols={},daily_by_symbol=daily,flow_by_symbol=flow,foreign_flow_by_symbol=foreign,
        institutional_flow_by_symbol=inst,index_by_code=indices,sector_map=sectors,trading_dates=tuple(days),
        bars_by_key=bars,sector_intraday_panel_1430=panel)
    research=native.research_snapshots_for_dataset(dataset,request['mutations'])
    morning=native.snapshots_for_experiment(dataset,request['mutations'],research_snapshots=research)
    result={d:afternoon_selection_from_snapshot(snap,bars,cfg,sectors) for d,snap in morning.items()}
    if cfg.require_complete_feature_prefix:
        from strategy_common.source_quality import POLICY
        for day, snapshot in result.items():
            excluded = [item.split(':')[0] for item in exclusions if item.endswith(':' + day.isoformat())]
            if set(excluded) & {c.symbol for c in snapshot.candidates}:
                raise ValueError('Incomplete feature prefix admitted by frozen OLR rule')
            snapshot.metadata.update(feature_prefix_policy=POLICY, feature_prefix_exclusions=excluded)
    return result


def bridge(root):
    from strategy_olr.models import OLRDailySnapshot
    study=read(root/'development/study.json');code=Path(study['code'])
    days=[date(2026,2,d) for d in (6,9,10)]
    snapshots=build(study['manifest'],code/study['database'],read(code/study['olr_request']),days)
    expected={s.trade_date:s for r in read(code/study['olr_snapshots']) if (s:=OLRDailySnapshot.from_json_dict(r)).trade_date in days[1:]}
    errors=[]
    for day in days[1:]:
        def comparable(snap):return [(c.symbol,c.rank,c.selection_score) for c in snap.candidates]
        a,b=comparable(snapshots[day]),comparable(expected[day])
        if len(a)!=len(b) or any(x[:2]!=y[:2] or abs(x[2]-y[2])>1e-8 for x,y in zip(a,b)):
            errors.append({'day':str(day),'new':a,'saved':b})
    write_json(root/'olr_inference_bridge.json',{'pass':not errors,'days':list(map(str,days[1:])),'errors':errors,
        'source_study':bound(root/'development/study.json'),'program':bound(Path(__file__))})
    print({'olr_bridge_pass':not errors,'mismatches':len(errors)},flush=True)
    if errors:raise ValueError('OLR candidate reconstruction differs from selected strategy')


if __name__=='__main__':bridge(ROOT)

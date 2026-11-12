"""Compile matched KALCB research pools with one session of bars in memory.

Uses the existing daily/first30 feature and candidate constructors. The seed OR
guarded-Q3 eligibility rule is applied after each ranker's top-16 shortlist,
before the survivors receive active ranks. Outcome fields never reach snapshots.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import date, datetime
import json
import math
from pathlib import Path
import sqlite3
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--database', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--seed', type=Path, required=True)
    p.add_argument('--q3', type=Path, required=True)
    p.add_argument('--reference-snapshots', type=Path, required=True)
    p.add_argument('--pool', action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.code_root.resolve()))
    from scripts.kalcb_bounded_replay import digest, effective_config, write
    from backtests.auto.shared.cache_keys import source_generation_hash
    from backtests.core.input_manifest import InputManifest
    from backtests.strategies.kalcb import first30_signal_sweep as features
    from backtests.strategies.kalcb.candidate_surfacing_recovery import _pool_route_meta
    from backtests.strategies.kalcb.fixed_trade_plan_phase import _configured_entry_routes, _route_candidate_passes
    from backtests.strategies.kalcb.trade_plan_sweep import compile_core_replay, _candidate_from_context
    from kis_core.trading_calendar import get_trading_calendar
    from strategy_common.market import MarketBar
    from strategy_kalcb.config import KALCBConfig

    source = source_generation_hash()
    generation = InputManifest(args.manifest)
    db_provenance = json.loads(args.database.with_suffix('.provenance.json').read_text())
    if db_provenance['manifest_hash'] != generation.identity or digest(args.database) != db_provenance['database_sha256']:
        raise ValueError('Changed market database')
    args.output.mkdir(parents=True, exist_ok=False)
    seed = json.loads(args.seed.read_text())['mutations']
    q3 = json.loads(args.q3.read_text())['mutations']
    def eligible(row, mutations):
        return any(_route_candidate_passes(route, mutations, _pool_route_meta(row))[0]
                   for route in _configured_entry_routes(mutations))
    pools, counts, hashes = {}, {}, {}
    for item in args.pool:
        name, path = item.split('=', 1)
        path = Path(path)
        hashes[str(path)] = digest(path)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        kept = [r for r in rows if eligible(r, seed) or (eligible(r, q3)
            and float(r.get('stock_sector_daily_ret5_spread') or 0) >= -.036622
            and r.get('sector') != 'FINANCIAL')]
        by_day = defaultdict(list)
        for row in kept:
            by_day[date.fromisoformat(row['trade_date'][:10])].append(row)
        pools[name] = by_day
        counts[name] = {'input':len(rows), 'guarded':len(kept), 'compiled':0, 'missing_context':0}
    snapshots = {}
    for path in args.reference_snapshots.glob('*.json'):
        raw = json.loads(generation.read(path))
        snapshots[date.fromisoformat(raw['trade_date'])] = raw
    config = effective_config(json.loads(args.reference.read_text()))
    cfg = KALCBConfig.from_mapping(config)
    get_trading_calendar()  # Verify this frozen code root's sourced session files.
    conn = sqlite3.connect(f'file:{args.database.as_posix()}?mode=ro', uri=True)
    conn.execute('PRAGMA cache_size=-8192')
    symbols = tuple(r[0] for r in conn.execute('SELECT DISTINCT symbol FROM bars ORDER BY symbol'))
    days = tuple(date.fromisoformat(r[0]) for r in conn.execute('SELECT DISTINCT day FROM bars ORDER BY day'))
    daily_root = generation.root/'data/krx_daily_parquet'
    with generation.activate():
        daily = features._load_daily_rows(daily_root,symbols,days[-1])
        flow = features._load_flow_rows(daily_root,symbols,days[-1])
        foreign = features._load_foreign_flow_rows(daily_root,symbols,days[-1])
        institutional = features._load_institutional_flow_rows(daily_root,symbols,days[-1])
        index = features._load_index_rows(daily_root,days[-1])
        sectors = features.load_sector_map(daily_root)
    dataset = features.KALCBFirst30Dataset(config, db_provenance['database_sha256'],generation.identity,
        generation.root/'data/kis_intraday_parquet', daily_root, '5m', symbols,symbols,
        tuple(daily),(),daily,flow,index,(),{},sectors,foreign,institutional)
    outputs = {name:[] for name in pools}
    parity = {'candidates_checked':0,'mismatches':[]}
    numeric = ('prior_day_high','prior_day_low','prior_day_close','daily_atr','expected_5m_volume',
               'average_30m_volume','accumulation_score','flow_score')
    for i, day in enumerate(days):
        bars = defaultdict(list)
        for symbol, stamp, *ohlcv in conn.execute('SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day=? ORDER BY symbol,ts',(day.isoformat(),)):
            bars[(day,symbol)].append(MarketBar(symbol,datetime.fromisoformat(stamp),'5m',*ohlcv))
        # These caches use tuple object identity; do not retain previous sessions.
        for cache in (features._SORTED_BARS_CACHE,features._FIRST30_BARS_CACHE,features._POST_ENTRY_BARS_CACHE):
            cache.clear()
        one = replace(dataset,trading_dates=(day,),bars_by_key={key:tuple(value) for key,value in bars.items()})
        contexts = features.build_contexts(one).get(day,())
        by_key = {(day,ctx.symbol):ctx for ctx in contexts}
        for old in snapshots.get(day,{}).get('candidates',[]):
            ctx=by_key.get((day,old['symbol']))
            if ctx is None:
                raise ValueError(f'Reference candidate lost: {day}/{old["symbol"]}')
            new = _candidate_from_context(ctx,features.Selection(day,old['symbol'],old['selection_score'],'parity'),one,cfg,1)
            for key in numeric:
                if not math.isclose(getattr(new,key),old[key],rel_tol=1e-10,abs_tol=1e-8):
                    parity['mismatches'].append([day.isoformat(),old['symbol'],key,old[key],getattr(new,key)])
            for key,value in old['metadata'].items():
                if key in new.metadata and isinstance(value,(float,int)) and key not in ('candidate_rank','frontier_rank','first30_score','frontier_selection_score'):
                    if not math.isclose(value,new.metadata[key],rel_tol=1e-10,abs_tol=1e-8):
                        parity['mismatches'].append([day.isoformat(),old['symbol'],key,value,new.metadata[key]])
            parity['candidates_checked'] += 1
        for name, pool in pools.items():
            rows=sorted(pool.get(day,()),key=lambda r:(int(r['pool_rank']),r['symbol']))
            selections=[features.Selection(day,r['symbol'],float(r['causal_ranker_score']),'matched_guarded_pool') for r in rows]
            missing = [s.symbol for s in selections if (day,s.symbol) not in by_key]
            counts[name]['missing_context'] += len(missing)
            extra = {(day,r['symbol']):{'pool_variant':name,'candidate_prefilter_contract':'seed_or_guarded_q3',
                    'forecast_train_through':r.get('forecast_train_through'), 'forecast_label_through':r.get('forecast_label_through')}
                    for r in rows}
            compiled=compile_core_replay(selections,one,by_key,(day,),{day:len(rows)},cfg,
                frontier_by_day={day:tuple(r['symbol'] for r in rows)},
                frontier_scores_by_day={day:{r['symbol']:float(r['causal_ranker_score']) for r in rows}},
                candidate_metadata_by_key=extra)
            for snapshot in compiled.snapshots.values():
                outputs[name].append(snapshot.to_json_dict())
                counts[name]['compiled'] += len(snapshot.candidates)
        if i%20==0:
            print(json.dumps({'session':i+1,'of':len(days),'parity_mismatches':len(parity['mismatches'])}),flush=True)
    conn.close()
    write(args.output/'feature_parity.json',parity)
    if parity['mismatches'] or any(c['missing_context'] for c in counts.values()):
        raise ValueError('Candidate parity or completeness failed; snapshots not released')
    for name, rows in outputs.items():
        write(args.output/(name+'.json'),rows)
    if source_generation_hash()!=source:
        raise RuntimeError('Source changed during compilation')
    write(args.output/'provenance.json',{'source_generation_hash':source,'manifest_hash':generation.identity,
        'database_sha256':db_provenance['database_sha256'],'pool_hashes':hashes,'counts':counts,
        'seed_sha256':digest(args.seed),'q3_sha256':digest(args.q3),'reference_sha256':digest(args.reference),
        'daily_loaded_files':sorted(generation.loaded),'scope':'Matched historical development; not untouched OOS'})
    print(json.dumps(counts),flush=True)


if __name__=='__main__':
    main()

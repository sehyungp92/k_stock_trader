"""Source-bound, non-trading feasibility of an earlier-stage frequency repair.

No returns are inspected. Only previously empty final-selection days are used.
Keep original downstream entry and score rules; screen daily participation-only
rejects for positive own flow and current sector confirmation at 14:30.
"""
from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

from scripts.olr_repair_failure_diagnosis_v5 import REPO, ROOT, OUT, SOURCES, read, bind, sha
from scripts.olr_snapshot_streaming import native_panel_cache


def main(w):
    study = read(ROOT / 'repair_v4/runtime/c4f49be01ed09b8dc' / w / 'baseline/study.json')
    code = Path(study['code'])
    sys.path.insert(0, str(code))
    from backtests.strategies.olr import research_sweep as native
    from strategy_olr import research, execution
    from strategy_olr.config import OLRConfig
    from strategy_common.sector_intraday import AFTERNOON_CUTOFF
    for module in [native, research, execution]:
        assert Path(module.__file__).is_relative_to(code)
        bind(Path(module.__file__))
    request = read(code / study['olr_request'])
    base_cfg = OLRConfig.from_mapping(request['config'], request['mutations'])
    altered = {**request['mutations'], 'olr.research.min_sector_participation': 0.0}
    cfg = OLRConfig.from_mapping(request['config'], altered)
    descriptor = read(ROOT / f'continuous_panel_v1/cache_sources/{w}/source.json')
    cache = Path(descriptor['cache'])
    assert sha(cache / 'index.json') == descriptor['index_sha256']
    index = read(cache / 'index.json')
    def payload(name):
        path = cache / name
        assert sha(path) == index['files'][name]
        bind(path)
        with path.open('rb') as f:
            return pickle.load(f)
    static = payload('static.pickle')
    days = tuple(date.fromisoformat(d) for d in index['identity']['base']['context_dates'])
    dataset = SimpleNamespace(config=request['config'], trading_dates=days, sector_map=static['sectors'], sector_intraday_panel_1430=static['panel'])
    snapshots = read(code / study['olr_snapshots'])
    period = study['configuration']['date_range']
    eligible = [r for r in snapshots if period['start'] <= r['trade_date'] <= period['end'] and not r['candidates']]
    counts = {'empty_days': len(eligible), 'sole_participation_rejects': 0, 'native_afternoon_survivors': 0, 'current_sector_confirmed': 0, 'six_bar_signals': 0}
    rows = []; day_rows = []
    plan = execution.OLREntryPlan(**request['mutations']['olr.trade_plan.entry'])
    for base in eligible:
        day = date.fromisoformat(base['trade_date'])
        snapshot, daily, same, bars = payload(day.isoformat() + '.pickle')
        sole = {s for s, reasons in base['metadata']['rejected_symbols'].items() if set(reasons) == {'sector_participation_below_floor'}}
        counts['sole_participation_rejects'] += len(sole)
        with native_panel_cache(research, day, snapshot, daily, same, static['sectors'], AFTERNOON_CUTOFF):
            original = native.snapshots_for_experiment(dataset, request['mutations'], research_snapshots={day:snapshot})[day]
            original = research.afternoon_selection_from_snapshot(original, bars, base_cfg, static['sectors'], sector_intraday_panel=None)
            assert not original.candidates, (w, day, 'Failed native empty-day reproduction')
            morning = native.snapshots_for_experiment(dataset, altered, research_snapshots={day:snapshot})[day]
            alternative = research.afternoon_selection_from_snapshot(morning, bars, cfg, static['sectors'], sector_intraday_panel=None)
        additions = [c for c in alternative.candidates[:6] if c.symbol in sole]
        counts['native_afternoon_survivors'] += len(additions)
        day_rows.append({'date': str(day), 'sole_rejects': len(sole), 'native_afternoon_survivors': len(additions)})
        for c in additions:
            f = c.metadata['afternoon_features']
            confirmed = f['lagged_flow_5d'] > 0 and f['sector_intraday_ret'] > 0 and f['sector_intraday_breadth'] >= .5 and f['sector_intraday_effective_count'] >= 3
            sig = execution.find_olr_entry_signal(bars.get((day, c.symbol), ()), c, plan, require_fill_bar=False) if confirmed else None
            counts['current_sector_confirmed'] += confirmed
            counts['six_bar_signals'] += sig is not None
            rows.append({'date': str(day), 'symbol': c.symbol, 'rank': c.rank, 'score': c.selection_score,
                         'stock_flow': f['lagged_flow_5d'], 'sector_return': f['sector_intraday_ret'],
                         'sector_breadth': f['sector_intraday_breadth'], 'sector_effective_count': f['sector_intraday_effective_count'],
                         'current_sector_confirmed': confirmed, 'six_bar_signal': sig is not None,
                         'signal_time': bars[day,c.symbol][sig.signal_index].timestamp.isoformat() if sig is not None else None})
    bind(Path(__file__))
    result = {'purpose': __doc__, 'window': w, 'counts': counts, 'rows': rows, 'days': day_rows,
              'limitations': ['Candidates/signals are not fills or added profit.', 'No funding, capacity, turnover, future holding overlap or cash displacement is simulated.',
                              'Removing only participation for reconstruction can affect rankings; proposed causal exception must receive its own exact candidate reproduction.',
                              'Sector taxonomy remains fixed; this is not point-in-time sector membership validation.',
                              'Current sector confirmation thresholds are the prior registered v4 thresholds, not tuned to these new candidates.'],
              'source_sha256': SOURCES}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'frequency_feasibility_{w}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'window': w, 'counts': counts, 'confirmed_rows': [r for r in rows if r['current_sector_confirmed']]}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])

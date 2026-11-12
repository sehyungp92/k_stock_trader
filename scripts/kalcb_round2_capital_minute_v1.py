"""C16 finest retained marks for fixed executed capital-perturbation paths."""
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_round2_assessment_v1 import read, write, bound, csvwrite, SCENARIOS

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('E:/CodexResearch/k_stock_trader/kalcb_round2_assessment_20261017/v1')
RUN = ROOT / 'noise_r2b'
OUT = ROOT / 'capital_minute'
REG = ROOT / 'capital_minute_registration.json'


def register():
    sources = {}
    for folder in sorted((REPO / 'data/kis_intraday_parquet').iterdir()):
        if not folder.is_dir(): continue
        files = list(folder.glob(folder.name + '_1m_*.parquet'))
        if files: sources[folder.name] = bound(max(files, key=lambda p: (p.stem.split('_')[-1], p.stem.split('_')[-2])))
    write(REG, dict(program=bound(__file__), sources=sources,
        protocol=bound(REPO / 'docs/optimisation/kalcb/experiments/kalcb-exp-0041-capital-minute.md')))
    print({'registered_sources': len(sources)})


def main():
    reg = read(REG); assert bound(__file__)['sha256'] == reg['program']['sha256']
    assert read(RUN / 'completion.json')['complete']
    OUT.mkdir(exist_ok=False); wanted = defaultdict(set); cases = []; unavailable = []
    for period in ('is', 'oos'):
        for scenario in SCENARIOS:
            for capital in (95000000, 105000000):
                name = f'capital_{period}_{scenario}_{capital}'; folder = RUN / 'cases' / name
                verdict = read(RUN / (name + '_verdict.json'))
                if not verdict['qualified']:
                    unavailable.append(dict(name=name, reason=verdict['reason'])); continue
                cases.append((period, scenario, capital, folder))
                for t in read(folder / 'trades.json'): wanted[t['symbol']].add(t['entry_fill_time'][:10])
    write(OUT / 'binding.json', dict(registration=bound(REG), accounts=[bound(f / x) for _, _, _, f in cases
        for x in ('fills.json', 'equity.json', 'daily.json', 'trades.json', 'result.json')], unavailable=unavailable))
    quotes = defaultdict(lambda: defaultdict(dict))
    for sym, dates in wanted.items():
        source = reg['sources'][sym]; assert bound(source['path'])['sha256'] == source['sha256']
        df = pd.read_parquet(source['path']); df = df[df.timestamp.dt.strftime('%Y-%m-%d').isin(dates)]
        assert not df.timestamp.duplicated().any(), ('Duplicate minute observations', sym)
        for r in df.itertuples():
            quotes[r.timestamp.strftime('%Y-%m-%d')][r.timestamp.to_pydatetime()][sym] = (float(r.open), float(r.close), int(r.Index))
    results = []
    for period, scenario, capital, folder in cases:
        fills = read(folder / 'fills.json'); eq = read(folder / 'equity.json'); daily = read(folder / 'daily.json')
        assert all(f['metadata']['execution_phase'] == 'open' for f in fills)
        bytime = defaultdict(list)
        for f in fills: bytime[datetime.fromisoformat(f['timestamp'])].append(f)
        native = {datetime.fromisoformat(t) + timedelta(minutes=5): value for t, value in eq}
        cash = float(capital); quantities = Counter(); marks = {}; observed = {}; quote_rows = {}; fields = {}
        peak = float(capital); peak_point = None; worst = 0.; points = None; stale = 0.; errors = []; allmarks = []; seen = 0

        def capture(at, phase, nav):
            return dict(at=str(at), phase=phase, nav=nav, cash=cash, holdings=[dict(symbol=s, qty=q, price=marks[s],
                source=reg['sources'][s], parquet_row=quote_rows[s], field=fields[s], available_at=str(observed[s]),
                age_minutes=(at-observed[s]).total_seconds()/60) for s, q in quantities.items() if q])

        def mark(at, phase):
            nonlocal peak, peak_point, worst, points, stale
            nav = cash + sum(q * marks[s] for s, q in quantities.items() if q)
            stale = max([stale] + [(at-observed[s]).total_seconds()/60 for s, q in quantities.items() if q])
            if nav >= peak: peak = nav; peak_point = capture(at, phase, nav)
            dd = 100 * (1 - nav/peak)
            if dd > worst: worst = dd; points = dict(peak=peak_point, trough=capture(at, phase, nav))
            allmarks.append(dict(at=str(at), phase=phase, nav=nav))
            return nav

        for day in daily:
            d = day['day']; at = datetime.fromisoformat(d+'T09:00:00+09:00'); end = datetime.fromisoformat(d+'T16:40:00+09:00')
            while at <= end:
                here = quotes[d].get(at, {})
                for s, (o, c, index) in here.items():
                    marks[s] = o; observed[s] = at; quote_rows[s] = index; fields[s] = 'open'
                for f in bytime.get(at, []):
                    sign = 1 if f['side'] == 'BUY' else -1
                    cash -= sign*f['qty']*f['price'] + f['metadata']['fee']; quantities[f['symbol']] += sign*f['qty']; seen += 1
                    assert quantities[f['symbol']] >= 0 and (sign < 0 or f['symbol'] in marks)
                mark(at, 'open_after_fill')
                next_at = at + timedelta(minutes=1)
                for s, (o, c, index) in here.items():
                    marks[s] = c; observed[s] = next_at; quote_rows[s] = index; fields[s] = 'close'
                nav = mark(next_at, 'close_before_next_open')
                if next_at in native: errors.append(abs(nav-native[next_at]))
                at = next_at
            assert not any(quantities.values()) and abs(cash-day['equity']) < .02
        assert seen == len(fills) and len(errors) == len(eq) and max(errors) < .02
        cell = f'{period}_{scenario}_{capital}'
        csvwrite(OUT / (cell+'_marks.csv'), allmarks); write(OUT / (cell+'_critical_points.json'), points)
        limit = 12 if scenario == 'baseline' else 15
        result = dict(period=period, scenario=scenario, initial_krw=capital, native_dd_pct=read(folder/'result.json')['drawdown_pct'],
            retained_minute_dd_pct=worst, observed_hard_limit_breach=worst>limit, peak_at=points['peak']['at'], trough_at=points['trough']['at'],
            maximum_held_mark_staleness_minutes=stale, critical_max_staleness_minutes=max([h['age_minutes'] for x in points.values() for h in x['holdings']], default=0),
            checked_native_marks=len(errors), max_native_nav_error_krw=max(errors), strict_finer_source_qualified=False)
        results.append(result); print(result, flush=True)
    csvwrite(OUT / 'summary.csv', results)
    write(OUT / 'completion.json', dict(complete=True, cells=len(results), unavailable=unavailable, new_native_accounts=0,
        all_native_closes_reconciled=True, strict_finer_source_qualified=False))


if __name__ == '__main__':
    if '--register' in sys.argv: register()
    else: main()

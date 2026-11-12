"""C16 decision-time ATR feature repair; preserve old fill-denominator statistics."""
from pathlib import Path
import csv
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_round2_assessment_v1 import read, write, bound, csvwrite, discriminate, SOURCE, SCENARIOS

ROOT = Path('E:/CodexResearch/k_stock_trader/kalcb_round2_assessment_20261017/v1')
OUT = ROOT / 'decision_atr_v1'


def main():
    OUT.mkdir(exist_ok=False)
    write(OUT / 'binding.json', dict(program=bound(__file__),
        protocol=bound(Path(__file__).resolve().parents[1] / 'docs/optimisation/kalcb/experiments/kalcb-exp-0041-feature-repair-r1.md'),
        sources=[bound(SOURCE / 'cases' / f'combined_{p}_{sc}' / 'trades.json') for p in ('is','oos') for sc in SCENARIOS]))
    results = []
    for period in ('is','oos'):
        for scenario in SCENARIOS:
            cell = f'{period}_{scenario}'; trades = read(SOURCE / 'cases' / ('combined_'+cell) / 'trades.json')
            original = list(csv.DictReader((ROOT / 'analysis' / (cell+'_trade_paths.csv')).open()))
            rows = []; missing = 0
            for r, t in zip(original, trades):
                assert r['symbol'] == t['symbol'] and r['entry'] == t['entry_fill_time']
                m = t['route_metadata']; numerator=m.get('daily_atr'); denominator=m.get('entry_price_ref')
                value = numerator/denominator if numerator is not None and numerator>0 and denominator is not None and denominator>0 else None
                missing += value is None
                rows.append({**r, 'net_bps':float(r['net_bps']), 'net_krw':float(r['net_krw']),
                    'atr_fraction_decision_ref':value, 'decision_reference':denominator})
            assert len(rows) == len(trades) == len(original)
            calendar=read(ROOT / 'analysis' / (cell+'_account.json'))['calendar']
            result=discriminate(rows,'atr_fraction_decision_ref',calendar)
            write(OUT / (cell+'.json'),dict(period=period,scenario=scenario,missing_decision_inputs=missing,statistics=result))
            csvwrite(OUT / (cell+'_rows.csv'),[{k:r[k] for k in ('day','symbol','trade_row','atr_fraction_decision_ref','decision_reference','net_bps','net_krw')} for r in rows])
            results.append(dict(period=period,scenario=scenario,missing=missing,auc=result['auc'],spearman=result['spearman'],source=str(OUT/(cell+'.json'))))
            print(results[-1],flush=True)
    csvwrite(OUT / 'summary.csv',results);write(OUT / 'completion.json',dict(complete=True,cells=8,new_native_accounts=0))


if __name__ == '__main__': main()

"""Complete registered capital dispersion display and exact campaign accounting."""
from collections import defaultdict
from pathlib import Path
import csv
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_round2_assessment_v1 import SOURCE, SCENARIOS, read, write, bound, csvwrite, blocks

ROOT = Path('E:/CodexResearch/k_stock_trader/kalcb_round2_assessment_20261017/v1')
OUT = ROOT/'capital_summary'


def campaigns(path):
    grouped = defaultdict(list)
    for i, t in enumerate(read(path/'trades.json'), 1):
        grouped[t['entry_fill_time'][:10], t['symbol']].append((i, t))
    return grouped


def main():
    assert read(OUT/'completion.json')['complete']
    write(OUT/'display_binding.json', dict(program=bound(__file__), inputs=bound(OUT/'completion.json'),
        estimand='Registered descriptive capital/scenario dispersion; all capital outcomes already observed. No new hypothesis test or economic rule.'))
    medians = []; changed = []; grouped = {}; ranges = []
    for period in ('is', 'oos'):
        for cap in (95000000, 105000000):
            diffs = []
            for sc in SCENARIOS:
                pair = read(OUT/f'{period}_{sc}_{cap}_daily_pair.json')
                diff = np.array(pair['paired_delta_pp']); diffs.append(diff)
                for length in (5, 10, 20):
                    medians.append(dict(period=period, scenario=sc, initial_krw=cap, block_length=length,
                        median_pp=float(np.median(diff[blocks(len(diff), length)].sum(axis=1)))))
                basepath = SOURCE/'cases'/f'combined_{period}_{sc}'
                newpath = ROOT/'noise_r2b/cases'/f'capital_{period}_{sc}_{cap}'
                before = campaigns(basepath); after = campaigns(newpath); reconciled = 0.
                for key in sorted(set(before) | set(after)):
                    old = before.get(key, []); new = after.get(key, [])
                    oldnet = sum(t['net_pnl'] for _, t in old); newnet = sum(t['net_pnl'] for _, t in new)
                    delta = 100*(newnet/cap-oldnet/1e8); reconciled += delta
                    if bool(old) == bool(new): continue
                    changed.append(dict(period=period, scenario=sc, initial_krw=cap, day=key[0], symbol=key[1],
                        status='added' if new else 'removed', baseline_net_krw=oldnet, capital_net_krw=newnet,
                        normalized_contribution_delta_pp=delta, baseline_source=str(basepath/'trades.json'),
                        baseline_rows=','.join(str(i) for i, _ in old), capital_source=str(newpath/'trades.json'),
                        capital_rows=','.join(str(i) for i, _ in new)))
                assert abs(reconciled-sum(diff)) < 1e-8
            mean = np.mean(diffs, axis=0)
            for length in (5, 10, 20):
                medians.append(dict(period=period, scenario='equal_four', initial_krw=cap, block_length=length,
                    median_pp=float(np.median(mean[blocks(len(mean), length)].sum(axis=1)))))
    cells = list(csv.DictReader((OUT/'cells.csv').open(encoding='utf-8')))
    for period in ('is', 'oos'):
        for cap in (95000000, 100000000, 105000000):
            rows = [r for r in cells if r['period']==period and int(r['initial_krw'])==cap]
            values = [float(r['return_pct']) for r in rows]
            ranges.append(dict(period=period, initial_krw=cap, return_mean_pct=float(np.mean(values)),
                return_min_pct=min(values), return_max_pct=max(values), return_range_pp=max(values)-min(values),
                return_descriptive_population_sd_pp=float(np.std(values)), independent_samples=False))
    csvwrite(OUT/'paired_medians.csv', medians); csvwrite(OUT/'changed_issuer_days.csv', changed)
    csvwrite(OUT/'scenario_dispersion.csv', ranges)
    write(OUT/'display_completion.json', dict(complete=True, all_campaign_contributions_reconcile=True, new_accounts=0))


if __name__=='__main__': main()

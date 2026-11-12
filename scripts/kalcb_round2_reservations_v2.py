"""C16 independent flat-opening reservation arithmetic from native decisions."""
from collections import defaultdict
from datetime import datetime, timedelta
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_round2_assessment_v1 import read, write, bound, csvwrite, SOURCE, SCENARIOS

ROOT = Path('E:/CodexResearch/k_stock_trader/kalcb_round2_assessment_20261017/v1')
OUT = ROOT / 'reservations_v2'


def main():
    OUT.mkdir(exist_ok=False)
    sources = [bound(SOURCE / 'cases' / f'combined_{p}_{sc}' / name)
               for p in ('is', 'oos') for sc in SCENARIOS for name in ('decisions.jsonl.gz', 'fills.json')]
    write(OUT / 'binding.json', dict(program=bound(__file__), sources=sources,
        protocol=bound(Path(__file__).resolve().parents[1] / 'docs/optimisation/kalcb/experiments/kalcb-exp-0041-reservation-repair-r1.md')))
    summary = []
    for period in ('is', 'oos'):
        for sc in SCENARIOS:
            folder = SOURCE / 'cases' / f'combined_{period}_{sc}'
            fills = read(folder / 'fills.json'); byday = defaultdict(list)
            for f in fills: byday[f['timestamp'][:10]].append(f)
            previous = None; reserved = 0.; rows = []; originals = {}
            with gzip.open(folder / 'decisions.jsonl.gz', 'rt', encoding='utf-8') as stream:
                for index, line in enumerate(stream, 1):
                    d = json.loads(line); at = datetime.fromisoformat(d['timestamp']); meta = d.get('metadata', {})
                    check = meta.get('pending_funding_reservation')
                    actions = [a for a in d.get('actions', []) if a['action_type'] == 'SubmitEntry']
                    if check is not None and d['decision_code'] == 'order_update':
                        original = originals.get((str(at.date()), d['symbol']))
                        assert original is not None and original[0] == check, ('Unmatched copied reservation', index)
                        rows.append({**original[1], 'raw_decision_row': index, 'decision_at': d['timestamp'],
                            'reason': d['reason'], 'record_kind': 'copied_order_update', 'original_decision_row': original[1]['raw_decision_row'],
                            'emitted_qty': 0})
                        continue
                    if previous != at: reserved = 0.; previous = at
                    if check is not None:
                        assert d['decision_code'] in ('entry', 'entry_rejected'), ('Unclassified computation', index)
                        available = at + timedelta(minutes=5)
                        flat = not any(datetime.fromisoformat(f['timestamp']) < available for f in byday[str(at.date())])
                        remainder = max(0., check['equity'] * check['limit'] - reserved)
                        capped = min(check['original_qty'], int(remainder / check['completed_close']))
                        emitted = sum(a['qty'] for a in actions)
                        good = flat and abs(check['reserved_gross']-reserved) < .02 and abs(check['remaining_notional']-remainder) < .02 and check['capped_qty'] == capped
                        rows.append(dict(period=period, scenario=sc, day=str(at.date()), symbol=d['symbol'],
                            decision_at=d['timestamp'], available_at=available.isoformat(), raw_decision_row=index,
                            reason=d['reason'], record_kind='reservation_computation', original_decision_row=index, original_qty=check['original_qty'], capped_qty=check['capped_qty'], emitted_qty=emitted,
                            reference_price=check['completed_close'], equity=check['equity'], recorded_reserved=check['reserved_gross'],
                            reconstructed_reserved=reserved, recorded_remaining=check['remaining_notional'],
                            reconstructed_remaining=remainder, flat_precondition=flat, passed=good))
                        originals[str(at.date()), d['symbol']] = (check, rows[-1])
                    for a in actions:
                        reserved += a['qty'] * a['metadata']['entry_price_ref']
            csvwrite(OUT / f'{period}_{sc}.csv', rows)
            copies = sum(r['record_kind'] == 'copied_order_update' for r in rows)
            rows = [r for r in rows if r['record_kind'] == 'reservation_computation']
            summary.append(dict(period=period, scenario=sc, copied_updates=copies, decisions=len(rows),
                dates=len({r['day'] for r in rows}), capped_decisions=sum(r['capped_qty'] < r['original_qty'] for r in rows),
                capped_dates=len({r['day'] for r in rows if r['capped_qty'] < r['original_qty']}),
                no_emission_after_cap=sum(r['capped_qty'] > 0 and not r['emitted_qty'] for r in rows),
                unsupported=sum(not r['flat_precondition'] for r in rows), mismatches=sum(not r['passed'] for r in rows),
                max_reservation_error_krw=max((abs(r['recorded_reserved']-r['reconstructed_reserved']) for r in rows), default=0)))
    csvwrite(OUT / 'summary.csv', summary)
    write(OUT / 'completion.json', dict(complete=True, passed=all(not r['mismatches'] for r in summary), cells=8, new_accounts=0))
    print(summary)
    assert all(not r['mismatches'] for r in summary), 'Reservation discrepancy: preserve generation and diagnose'


if __name__ == '__main__': main()

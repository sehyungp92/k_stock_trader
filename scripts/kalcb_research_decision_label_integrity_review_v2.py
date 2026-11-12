"""Read-only investigation of C18's failed label capacity qualification.

No native replay, label repair, score computation or downstream authorisation.
Stream completed batches, retain affected raw records and independently sum
executed quantities by physical bar. The frozen zero-invalid gate is unchanged.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import time

ROOT = Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1')
OUT = ROOT / 'label_integrity_review_v2'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bound(path):
    with Path(path).open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest}


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def main():
    start = time.monotonic()
    reg = read(OUT / 'registration.json')
    assert bound(__file__) == reg['program']
    all_rows, summaries = [], []
    for item in reg['completed_batches']:
        assert bound(item['path']) == item
        completion = read(item['path'])
        folder = Path(item['path']).parent
        scenario = folder.name
        period = folder.parent.name
        for binding in completion['files'].values():
            assert bound(binding['path']) == binding
        targets = {}
        counts = Counter()
        with gzip.open(folder / 'labels.jsonl.gz', 'rt', encoding='utf-8') as handle:
            for number, line in enumerate(handle, 1):
                label = json.loads(line)
                counts[label['status']] += 1
                if label['status'] == 'physical_invalid':
                    assert not label['label_valid'] and not label['eligible_for_scoring']
                    assert label['net_value_notional_bps'] is None
                    targets[number] = label
        assert dict(counts) == completion['statuses']
        found = set()
        with gzip.open(folder / 'native_records.jsonl.gz', 'rt', encoding='utf-8') as handle:
            for number, line in enumerate(handle, 1):
                if number not in targets:
                    continue
                raw = json.loads(line)
                label = targets[number]
                assert (raw['episode_id'], raw['horizon_minutes']) == (label['episode_id'], label['horizon_minutes'])
                found.add(number)
                raw_name = f'{period}_{scenario}_row{number}.json'
                save(OUT / raw_name, raw)
                by_bar = defaultdict(list)
                for fill in raw['fills']:
                    bar = fill['metadata']['execution_bar']
                    by_bar[(bar['symbol'], bar['timestamp'])].append(fill)
                observed = []
                for key, fills in by_bar.items():
                    volumes = {float(f['metadata']['execution_bar']['volume']) for f in fills}
                    assert len(volumes) == 1
                    volume = volumes.pop()
                    quantity = sum(int(f['qty']) for f in fills)
                    if quantity <= volume:
                        continue
                    quotes = [event for event in raw['complete_native_journal']
                              if event['event'] == 'capacity'
                              and event['timestamp'].replace(' ', 'T') == key[1].replace(' ', 'T')]
                    cash = 100_000_000.
                    inventory = 0
                    for fill in raw['fills']:
                        fee = float(fill['metadata']['fee'])
                        cash += (-1 if fill['side'] == 'BUY' else 1) * int(fill['qty']) * float(fill['price']) - fee
                        inventory += (1 if fill['side'] == 'BUY' else -1) * int(fill['qty'])
                        assert inventory >= 0
                    assert inventory == label['terminal_inventory']
                    assert abs(cash - 100_000_000 - label['net_cash_krw']) < .02
                    observed.append((key[0], key[1], quantity, volume))
                    all_rows.append({'period': period, 'scenario': scenario, 'source_row': number,
                        'episode_id': label['episode_id'], 'horizon': label['horizon_minutes'],
                        'day': label['day'], 'symbol': key[0], 'bar': key[1],
                        'filled_quantity': quantity, 'printed_quantity': volume,
                        'excess_quantity': quantity-volume, 'admitted': label['admitted'],
                        'sides': '/'.join(sorted({f['side'] for f in fills})),
                        'phases': '/'.join(sorted({f['metadata']['execution_phase'] for f in fills})),
                        'raw_record': raw_name,
                        'capacity_attempts': len(quotes),
                        'predictors': json.dumps([q['quote'] for q in quotes], sort_keys=True),
                        'label_valid': False, 'eligible_for_scoring': False})
                expected = sorted((x['symbol'], x['timestamp'], x['quantity'], x['realized_bar_volume'])
                                  for x in label['physical_failures'])
                assert sorted(observed) == expected
                if found == set(targets):
                    break
        assert found == set(targets)
        summaries.append({'period': period, 'scenario': scenario, 'counts': dict(counts),
                          'qualification_passes': counts['physical_invalid'] == 0,
                          'completion': item})
    with (OUT / 'violations.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    save(OUT / 'receipt.json', {'passed': True, 'utc': datetime.now(timezone.utc).isoformat(),
        'seconds': time.monotonic()-start, 'native_evaluations': 0,
        'batches': summaries, 'independently_confirmed_violations': len(all_rows),
        'distinct_episodes': len({x['episode_id'] for x in all_rows}),
        'distinct_issuer_days': len({(x['day'], x['symbol']) for x in all_rows}),
        'admitted_affected_episodes': sorted({x['episode_id'] for x in all_rows if x['admitted']}),
        'frozen_gate_unchanged': True, 'downstream_authorised': False,
        'files': {p.name: bound(p) for p in OUT.iterdir() if p.name not in ('receipt.json',)}})
    print(json.dumps({'violations': len(all_rows), 'seconds': time.monotonic()-start}))


if __name__ == '__main__':
    main()

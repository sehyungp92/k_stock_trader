"""Independent late-settlement ledger check for a preserved loader failure."""
from collections import Counter
from datetime import datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_full_oos_recovery import ROOT, read, write, digest, bound
from scripts.kalcb_alignment_walk_forward_v2 import ALIGNMENT, economic_label_contract, verify_bindings
from backtests.strategies.kalcb.executed_value import physical_fill_failures


def run():
    output = ROOT/'alignment_deadline_label_audit_20261010.json'
    if output.exists(): raise ValueError('Preserve previous audit')
    protocol = read(ALIGNMENT/'protocol.json'); complete = read(ALIGNMENT/'completion.json')
    verify_bindings(protocol['bound_inputs'] + complete['grids'])
    cfg = read(ALIGNMENT/'selected_management.json')
    records = []; audited = []; all_counts = Counter(); invalid = []
    for job in protocol['jobs']:
        folder = Path(job['output']); m = read(folder/'grid_manifest.json'); counts = Counter(); labels_count = 0
        for filename, expected in m['files'].items():
            if not filename.endswith('.jsonl'): continue
            path = folder/filename
            if digest(path) != expected: raise ValueError('Changed native labels')
            late = {}
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line); classification = economic_label_contract(row)
                    counts[classification] += 1; labels_count += 1
                    if row['label_valid'] and (classification == 'settled_after_deadline'
                            or datetime.fromisoformat(row['label_available_at']) < datetime.fromisoformat(row['risk_available_at'])):
                        late[row['opportunity_id']] = row
                    if classification == 'invalid_economic_target':
                        invalid.append({'job': job['name'], 'opportunity_id': row['opportunity_id'], 'status': row['status'],
                                        'physical_failures': row['physical_fill_failures'], 'unresolved': row['unresolved']})
            if not late: continue
            ep = path.with_name(path.stem+'.events.jsonl.gz')
            if digest(ep) != m['files'][ep.name]: raise ValueError('Changed native fill evidence')
            seen = set()
            with gzip.open(ep, 'rt', encoding='utf-8') as stream:
                for line in stream:
                    detail = json.loads(line); key = detail['opportunity_id']
                    if key not in late: continue
                    if key in seen: raise ValueError('Duplicate detailed outcome')
                    seen.add(key); row = late[key]; cash = 0.; fees = 0.; inventory = 0; latest = None
                    if physical_fill_failures(detail['fills']): raise ValueError('Invalid volume in settled deadline episode')
                    for fill in detail['fills']:
                        qty, price, side = int(fill['qty']), float(fill['price']), fill['side']
                        if qty <= 0 or price <= 0 or side not in ('BUY', 'SELL'): raise ValueError('Invalid fill')
                        fee = qty*price*(cfg['commission_bps']+(cfg['tax_bps_on_sell'] if side == 'SELL' else 0))/10000
                        fees += fee; cash += qty*price*(1 if side == 'SELL' else -1)-fee
                        inventory += qty*(1 if side == 'BUY' else -1)
                        if inventory < 0: raise ValueError('Sell without owned inventory')
                        available = datetime.fromisoformat(fill['metadata']['execution_bar']['timestamp'])+timedelta(minutes=5)
                        latest = max(latest or available, available)
                    if inventory or abs(cash-row['net_cash_flow_krw']) > .01 or abs(fees-row['fees_and_tax_krw']) > .01:
                        raise ValueError('Independent settlement ledger mismatch')
                    if (latest is None and row['deadline_violation']) or (latest is not None and latest > datetime.fromisoformat(row['label_available_at'])):
                        raise ValueError('Economic target available before final fill observation')
                    audited.append({'job': job['name'], 'opportunity_id': key, 'net_value_krw': cash,
                                    'risk_available_at': row['risk_available_at'], 'label_available_at': row['label_available_at'],
                                    'last_fill_bar_available_at': latest.isoformat() if latest else None, 'deadline_inventory': row['deadline_inventory']})
            if seen != set(late): raise ValueError('Missing late-settlement detailed evidence')
        all_counts.update(counts)
        records.append({'job': job['name'], 'actions': labels_count, 'classes': dict(counts), 'manifest': bound(folder/'grid_manifest.json')})
        print(json.dumps(records[-1]), flush=True)
    payload = {'passed': True, 'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'program': bound(__file__),
               'validator': bound(Path(__file__).with_name('kalcb_alignment_walk_forward_v2.py')), 'protocol': bound(ALIGNMENT/'protocol.json'),
               'counts': dict(all_counts), 'jobs': records, 'settlement_ledger_audits': audited, 'invalid_economic_targets': invalid,
               'diagnosis': 'v1 loader incorrectly equated deadline-risk failure with invalid cash. Actual late settlements have mature reconciled cash and remain unsafe risk targets. The successor preserves native label semantics; no fills or economic target values change.',
               'limitation': 'This checks the distinction, fill volumes, independent cash, inventory and economic availability; it is not full source qualification of new portfolio policies.'}
    write(output, payload); print(json.dumps({'passed': True, 'counts': dict(all_counts), 'output': str(output)}), flush=True)


if __name__ == '__main__': run()

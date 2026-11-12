"""Finite existing-account minute35 support audit. No new action outcomes."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

REPO = Path(__file__).resolve().parents[1]
BASE = Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
HELPER = REPO/'backtests/strategies/kalcb/staged_support_v1.py'
LEDGER = REPO/'backtests/strategies/kalcb/exit_latency_audit_v1.py'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def bound(p):
    p = Path(p).resolve()
    with p.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'path': str(p), 'sha256': digest}


def verify(bindings):
    for b in bindings:
        if bound(b['path'])['sha256'] != b['sha256']:
            raise ValueError('Changed registered binding '+b['path'])


def write(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m)
    return m


def register(root, tests):
    receipt = read(tests)
    if not receipt['passed'] or receipt['tests'] != 7:
        raise ValueError('Focused checks missing')
    verify(receipt['programs']+[receipt['log']])
    original = BASE/'v46_risk_contract_materiality/protocol.json'; p = read(original)
    verify(p['programs']+p['bindings'])
    paths = [original, BASE/'v46_risk_contract_materiality/completion.json',
             BASE/'v46_risk_contract_materiality/review_v1/completion.json', Path(tests)]
    for name in ['protocol.json', 'economic_decision.json', 'completion.json',
                 'review_v1/completion.json', 'post_completion_health.json']:
        paths.append(BASE/'v48_staged_reference_pilot'/name)
    for case in p['cases']:
        folder = Path(case['folder'])
        paths.append(Path(case['assessment']))
        for name in ['completion.json', 'observer_evidence.json', 'fills.json', 'orders.jsonl.gz',
                     'independent_account_verification.json', 'independent_exposure_mtm.json']:
            paths.append(folder/name)
    root.mkdir(exist_ok=False)
    write(root/'protocol.json', {
        'registered_at_utc': datetime.now(timezone.utc).isoformat(),
        'kind': 'existing_parent_staged_confirmation_support', 'cases': p['cases'],
        'databases': p['databases'],
        'population': 'ALL actual first30_open_anchor submitted orders in12existing unchanged accounts: prior100 all4scenarios,prior125 baseline/combined,IS/OOS. No outcome/issuer/date filter. Preserve nofills,one-share and unsupported clocks.',
        'frozen_contrast': 'The exact V47/V48 minute35 price predicate only:7consecutive valid positivevolume completed5minute bars from09:00; close35>=originalclose30 AND close35>=own7barVWAP. Originalqty>=2; single shares unchanged. No threshold/timing variants. This is a necessary price condition, NOT a staged native availability prediction.',
        'measurements': 'Authenticate originalclose30 against raw bars. Attach every original position PnL once through unique entryorder IDs. Parent fill quantities strictly before09:35; parent sell intents <=09:35 retained separately. Record actual preSubmit inventory/pending; ALL size/session/month/issuer support and completedquote change. Partition original net/grosswinner/loss/fees by fixed pricepredicate; zero-fill retained. No posthoc scaledPnL, avoidedDD or pooledscenario support.',
        'review': 'Independent stdlib rawprefix/quantity/cashflow and parent position partition reconstruction after exact producer exit. Historical whole-source/account/MTM qualification reused, not re-rendered. New consumed7bar source rows hashed and independently reread.',
        'decision': 'Determine material targetcoverage, sacrificed originalwinners, causal maturity and portfolio coupling BEFORE any separately frozen larger matching-label or native study. The fourday knownpilot is not robust evidence. No automatic next model/nativecensus/grid.',
        'programs': [bound(x) for x in [Path(__file__), HELPER, LEDGER,
            REPO/'scripts/kalcb_staged_support_review_v1.py',
            REPO/'scripts/kalcb_staged_support_pipeline_v1.py',
            REPO/'tests/test_kalcb_staged_support_v1.py']],
        'bindings': [bound(x) for x in dict.fromkeys(paths)],
        'estimate': '12existing accounts,2-5minutes plus independent review; zero native replays/fits. Background sequential read-only producer/review.',
        'study_complete': False})


def run(root):
    import psutil
    p = read(root/'protocol.json'); verify(p['programs']+p['bindings'])
    me = psutil.Process()
    write(root/'supervisor_start.json', {'pid': me.pid, 'process_created': me.create_time(),
          'command': me.cmdline(), 'at_utc': datetime.now(timezone.utc).isoformat()})
    h = module('frozen_staged_support', HELPER); ledger = module('frozen_staged_ledger', LEDGER)
    cache = {}; outputs = []; summaries = {}
    connections = [sqlite3.connect('file:'+Path(db).as_posix()+'?mode=ro', uri=True) for db in p['databases']]
    try:
        for case in p['cases']:
            folder = Path(case['folder']); key = case['key']
            account = read(case['assessment'])['accounts'][case['assessment_key']]
            positions = account['positions']; fills = read(folder/'fills.json')
            mapped = ledger.lifetimes(fills, positions)
            by_order = {}
            for pos in positions:
                for oid in pos['entry_order_ids']:
                    if oid in by_order:
                        raise ValueError('Repeated lifetime for entryorder')
                    by_order[oid] = pos
            sells = defaultdict(list)
            with gzip.open(folder/'orders.jsonl.gz', 'rt', encoding='utf-8') as f:
                for line in f:
                    journal = json.loads(line)
                    if journal['event'] != 'submit':
                        continue
                    order = journal['order']
                    if order['side'] == 'SELL':
                        sells[order['symbol']].append(order)
            exposures = read(folder/'observer_evidence.json')['entry_exposures']
            rows = []; seen = set()
            for e in exposures:
                a = e['action']; meta = a['metadata']
                if meta.get('entry_route') != 'first30_open_anchor':
                    continue
                if e['order_id'] in seen:
                    raise ValueError('Repeated submitted exposure')
                seen.add(e['order_id'])
                day = h.clock(e['decision_available_at']).date().isoformat(); symbol = a['symbol']
                pair = (symbol, day)
                if pair not in cache:
                    bars = []
                    for conn in connections:
                        bars.extend(conn.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND day=? ORDER BY ts', pair))
                    bars = [list(b) for b in sorted(bars)]
                    if len({b[0] for b in bars}) != len(bars):
                        raise ValueError('Duplicate raw source')
                    cache[pair] = bars
                row = h.confirmation(e, cache[pair]); at = h.clock(row['completion_at'])
                pos = by_order.get(e['order_id']); fs = mapped[pos['position_id']] if pos else []
                bought = [f for f in fs if f['side'] == 'BUY' and h.clock(f['timestamp']) < at]
                parent_intents = [o for o in sells[symbol] if pos
                    and h.clock(pos['entry_fill_time']) <= h.clock(o['metadata']['decision_available_at']) <= at
                    and h.clock(o['metadata']['decision_available_at']) <= h.clock(pos['exit_fill_time'])]
                before = e['account_before']
                row.update({'order_id': e['order_id'], 'symbol': symbol, 'day': day,
                    'decision_available_at': h.clock(e['decision_available_at']).isoformat(),
                    'candidate_rank': meta['candidate_rank'], 'position_id': pos['position_id'] if pos else None,
                    'parent_net_pnl_krw': pos['net_pnl_krw'] if pos else 0.,
                    'parent_entry_cost_krw': pos['entry_cost_krw'] if pos else 0.,
                    'parent_fees_krw': pos['fees_krw'] if pos else 0.,
                    'parent_qty_bought_before35': sum(f['qty'] for f in bought),
                    'parent_full_quantity_filled_before35': sum(f['qty'] for f in bought) == a['qty'],
                    'parent_sale_before35': any(f['side'] == 'SELL' and h.clock(f['timestamp']) < at for f in fs),
                    'parent_sell_intent_at_or_before35': bool(parent_intents),
                    'parent_sell_intents_at_or_before35': [{k:o[k] for k in ['order_id','reason','qty']} for o in parent_intents],
                    'initial_nonflat_or_pending': bool(before['positions'] or before['pending_orders']),
                    'initial_actual_equity': before['actual_equity'],
                    'planned_notional_over_equity': a['qty']*meta['entry_price_ref']/before['actual_equity'],
                    'original_maturity_day': h.clock(pos['exit_fill_time']).date().isoformat() if pos else None,
                    'staged_label_available': False})
                rows.append(row)
            expected = {x['position_id'] for x in positions if x['route'] == 'first30_open_anchor'}
            if {x['position_id'] for x in rows if x['position_id']} != expected:
                raise ValueError('Incomplete original position coverage')
            groups = {category:h.summarize([x for x in rows if x['category'] == category])
                      for category in sorted({x['category'] for x in rows})}
            months = {month:{category:h.summarize([x for x in rows if x['day'][:7] == month and x['category'] == category])
                      for category in groups} for month in sorted({x['day'][:7] for x in rows})}
            batches = defaultdict(list)
            for row in rows:
                batches[row['decision_available_at']].append(row)
            result = {'account': key, 'rows': rows, 'summary': h.summarize(rows), 'groups': groups,
                      'months': months, 'batches': [{'at': at, 'orders': len(rs),
                        'price_confirmed_orders': sum(x['category'] == 'price_confirmed' for x in rs),
                        'original_planned_notional_over_first_equity': sum(x['planned_qty']*x.get('original_reference', 0) for x in rs)/rs[0]['initial_actual_equity']}
                        for at, rs in sorted(batches.items())],
                      'scope': 'Original funded lifetimes attached once. No candidate outcomes or availability inferred.'}
            path = root/'accounts'/Path(key).with_suffix('.json'); write(path, result); outputs.append(bound(path))
            summaries[key] = {'summary': result['summary'], 'groups': groups,
                'batches':len(batches), 'multi_order_batches':sum(len(rs)>1 for rs in batches.values())}
            print(json.dumps({'account':key,'orders':len(rows),'groups':{k:v['submitted_orders'] for k,v in groups.items()}}),flush=True)
            (root/'progress.json').write_text(json.dumps({'complete_accounts':len(outputs),'expected':len(p['cases']),'latest':key}),encoding='utf-8')
            del exposures, fills, mapped, account, positions
    finally:
        for c in connections:c.close()
    verify(p['programs']+p['bindings'])
    write(root/'assessment.json', {'accounts':summaries,'outputs':outputs,'native_cases':0,'fits':0,'study_complete':False})
    write(root/'completion.json', {'complete':True,'at_utc':datetime.now(timezone.utc).isoformat(),
        'assessment':bound(root/'assessment.json'),'outputs':outputs,'study_complete':False})


if __name__ == '__main__':
    ap = argparse.ArgumentParser();ap.add_argument('mode',choices=['register','run'])
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--tests',type=Path);a=ap.parse_args()
    try:
        register(a.root,a.tests) if a.mode=='register' else run(a.root)
    except Exception as error:
        if a.root.exists() and not (a.root/'failure.json').exists():
            write(a.root/'failure.json', {'error':repr(error),'preserve_evidence':True})
        raise

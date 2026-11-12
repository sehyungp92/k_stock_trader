"""Complete censored March31 labels with native April continuation, preserving IS.

Only target evidence is extended. No action population, features, entry decisions
or reporting-account boundary changes; immature outcomes remain unavailable.
"""
from datetime import datetime, timezone
from dataclasses import replace
import argparse
import gzip
import json
from pathlib import Path
import sqlite3
import sys

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('C:/Users/sehyu/.codex/research/kalcb_full_oos_recovery_20261009/v3')
NATIVE = ROOT/'inputs/files'
sys.path.insert(0, str(NATIVE))
import scripts
scripts.__path__.append(str(REPO/'scripts'))
from scripts.kalcb_full_oos_recovery import read, write, bound, digest, rows
from scripts.kalcb_executed_value_study import config_for, connect, load_day, run_episode
from scripts.kalcb_action_quantity_labels import scaled_row
from backtests.auto.shared.cache_keys import source_generation_hash

ALIGNMENT = Path('E:/CodexResearch/k_stock_trader/kalcb_selected_target_alignment_20261010/v2')
DEST = Path('E:/CodexResearch/k_stock_trader/kalcb_label_boundary_completion_20261010/v2')
HISTORY = Path('E:/CodexResearch/k_stock_trader/kalcb_intraday_advantage_20261007/checkpoint')
FORWARD = Path('C:/Users/sehyu/.codex/research/olr_round1_oos_20261009/forward_features')


def check(items):
    for item in items:
        if digest(item['path']) != item['sha256']: raise ValueError('Changed boundary input '+item['path'])


def register(out):
    if out.exists(): raise ValueError('Preserve previous boundary evidence')
    paths = {'original': HISTORY/'labels_combined', 'selected': ALIGNMENT/'labels/is_timed_combined'}
    expected = {'2026-03-31/'+s+'/q1' for s in ('041510', '214370', '010130', '005300')}
    jobs = []; inputs = [bound(__file__), bound(ALIGNMENT/'completion.json')]
    for family, folder in paths.items():
        m = read(folder/'grid_manifest.json'); labels = folder/'2026-03-31.jsonl'
        if digest(labels) != m['files'][labels.name]: raise ValueError('Changed boundary labels')
        prior = {r['opportunity_id']: r for r in rows(labels) if r['unresolved']}
        if set(prior) != expected or m['counts'].get('unresolved', 0) != len(prior):
            raise ValueError('Unexpected censoring population; expand the audit before replay')
        request = read(folder/'request.json'); plan = read(request['arguments']['plan'])
        prepared = Path(request['arguments']['prepared']); pm = read(prepared/'prepared.json')
        if digest(prepared/'prepared.json') != m['prepared_sha256'] or digest(prepared/'opportunities.jsonl') != pm['opportunities_sha256']:
            raise ValueError('Changed boundary feature preparation')
        events = folder/'2026-03-31.events.jsonl.gz'
        if digest(events) != m['files'][events.name]: raise ValueError('Changed historical execution prefix')
        inputs += [bound(labels), bound(events), bound(folder/'grid_manifest.json'), bound(folder/'request.json'),
                   bound(request['arguments']['plan']), bound(prepared/'prepared.json'), bound(prepared/'opportunities.jsonl'), bound(plan['config'])]
        jobs.append({'family': family, 'kind': 'timed', 'scenario': 'combined', 'folder': str(folder),
                     'plan': request['arguments']['plan'], 'prepared': str(prepared), 'actions': sorted(prior)})
    forward_plan = read(FORWARD/'checkpoint_plan.json')
    if digest(forward_plan['database']) != forward_plan['input_sha256'][forward_plan['database']]:
        raise ValueError('Changed continuous forward market source')
    inputs += [bound(FORWARD/'checkpoint_plan.json'), bound(forward_plan['database']), bound(FORWARD/'checkpoint/prepared.json')]
    out.mkdir(parents=True)
    write(out/'protocol.json', {'registered_at_utc': datetime.now(timezone.utc).isoformat(), 'jobs': jobs, 'bound_inputs': inputs,
        'forward_plan': str(FORWARD/'checkpoint_plan.json'), 'native_source_generation_hash': read(ROOT/'preparation.json')['source_generation_hash'],
        'purpose': 'Complete four censored March31 actions under EACH target definition. Preserve exact March31 bars/fills/orders/decisions and raw censored evidence; attach actual mature outcomes only after their real observation time.',
        'scope': 'Mechanical continuous-history repair. No hindsight admission filter, no independent OOS claim, no canonical changes.'})


def run(out):
    protocol = read(out/'protocol.json'); check(protocol['bound_inputs'])
    if source_generation_hash() != protocol['native_source_generation_hash']: raise ValueError('Wrong economic source')
    if (out/'completion.json').exists(): raise ValueError('Preserve existing completion')
    forward = read(protocol['forward_plan']); sessions = read(FORWARD/'checkpoint/prepared.json')['sessions']
    if sessions[0] != '2026-03-31' or sessions[-1] != '2026-07-10': raise ValueError('Wrong continuation calendar')
    conn = connect(forward); overlays = []; proofs = []
    for job in protocol['jobs']:
        old = Path(job['folder']); m = read(old/'grid_manifest.json'); request = read(old/'request.json')
        plan = read(job['plan']); cfg = replace(config_for(plan, 'combined'), entry_action_quantity_scale_enabled=True)
        prepared = Path(job['prepared']); source = read(prepared/'prepared.json')['opportunities_sha256']
        originals = {r['opportunity_id']: r for r in rows(old/'2026-03-31.jsonl')}
        with gzip.open(old/'2026-03-31.events.jsonl.gz', 'rt', encoding='utf-8') as f:
            old_details = {r['opportunity_id']: r for r in map(json.loads, f) if r['opportunity_id'] in job['actions']}
        opportunities = {r['opportunity_id']: r for r in rows(prepared/'opportunities.jsonl') if r['trade_date'] == '2026-03-31'}
        prior_conn = connect(plan)
        for key in job['actions']:
            base = key.rsplit('/q', 1)[0]; raw = opportunities[base]; row = scaled_row(raw, 1.)
            symbol = raw['symbol']
            query = 'SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day=? AND symbol=? ORDER BY ts'
            before = prior_conn.execute(query, ('2026-03-31', symbol)).fetchall()
            after = conn.execute(query, ('2026-03-31', symbol)).fetchall()
            if before != after or not before: raise ValueError('Historical/forward execution bar boundary mismatch')
            value, detail = run_episode(row, load_day(conn, '2026-03-31', source, {symbol}), cfg, 'combined', source, conn, sessions)
            value.update(base_opportunity_id=base, quantity_scale=1.)
            old_detail = old_details[key]
            for field in ('fills', 'orders', 'decisions'):
                if detail[field][:len(old_detail[field])] != old_detail[field]:
                    raise ValueError('Native historical prefix drift: '+field+' '+key)
            if value['deadline_inventory']['2026-03-31'] != originals[key]['deadline_inventory']['2026-03-31']:
                raise ValueError('Changed historical risk event')
            if value['label_valid'] and value['label_available_at'] <= '2026-04-01T00:00:00+09:00':
                raise ValueError('Continuation value became available before actual continuation')
            name = job['family']+'_'+symbol; vp = out/(name+'.label.json'); ep = out/(name+'.events.json.gz')
            if vp.exists() or ep.exists(): raise ValueError('Preserve previous boundary case')
            write(vp, value)
            with gzip.open(ep, 'wt', encoding='utf-8') as f: json.dump(detail, f, default=str, allow_nan=False)
            proofs.append({'family': job['family'], 'opportunity_id': key, 'exact_prefix_bars': len(before),
                           'prefix_counts': {k: len(old_detail[k]) for k in ('fills', 'orders', 'decisions')}, 'value': bound(vp), 'events': bound(ep)})
            overlays.append({'family': job['family'], 'kind': job['kind'], 'scenario': job['scenario'], 'prior_label': originals[key], 'label': value})
            print(json.dumps({'family': job['family'], 'action': key, 'status': value['status'], 'available': value['label_available_at']}), flush=True)
        prior_conn.close()
    conn.close(); check(protocol['bound_inputs'])
    if source_generation_hash() != protocol['native_source_generation_hash']: raise ValueError('Changed economic source')
    write(out/'overlays.json', overlays); write(out/'prefix_bridge.json', {'passed': True, 'cases': proofs})
    write(out/'completion.json', {'complete': True, 'source_generation_hash': protocol['native_source_generation_hash'],
        'protocol': bound(out/'protocol.json'), 'overlays': bound(out/'overlays.json'), 'prefix_bridge': bound(out/'prefix_bridge.json')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['register', 'run']); p.add_argument('--root', type=Path, default=DEST)
    a = p.parse_args(); {'register': register, 'run': run}[a.stage](a.root)

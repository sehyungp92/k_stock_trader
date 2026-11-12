"""Authenticate and reconcile saved release/rotation evidence; no new replay."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('E:/CodexResearch/k_stock_trader/olr_allocation_recovery_20261008')
OUT = Path(__file__).resolve().parents[1]/'docs/optimisation/olr/evidence/earlier-capital-predecessors.json'


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run():
    dependencies = {str(Path(__file__)): sha(Path(__file__))}
    rows = []
    summary = BASE/'native_trial_summary.json'
    dependencies[str(summary)] = sha(summary)
    for generation, recipes in [(4, ['median','routed','release60','release60half','release90']),
                                (5, ['median','routed','rotation_half','rotation_full','rotation_any_half'])]:
        gen = BASE/f'generation_{generation}'
        for name in ('registration.json', 'screen_verdict.json'):
            dependencies[str(gen/name)] = sha(gen/name)
        verdict = read(gen/'screen_verdict.json')
        assert verdict['complete'] and not verdict['eligible']
        for recipe in recipes:
            for scenario in ('baseline','combined'):
                case = gen/'cases'/f'{recipe}_100000000_{scenario}'
                completion, result, trades = [read(case/name) for name in ('completion.json','result.json','trades.json')]
                assert completion['complete'] and sha(case/'result.json') == completion['result_sha256']
                net = sum(t['net_pnl'] for t in trades)
                assert not result['terminal_inventory'] and result['terminal_pending'] == 0
                assert math.isclose(net, result['return_pct']*1_000_000, abs_tol=0.01)
                assert len(trades) == result['trades']
                for name in ('completion.json','result.json','trades.json'):
                    dependencies[str(case/name)] = sha(case/name)
                rows.append(dict(generation=generation, recipe=recipe, scenario=scenario,
                    return_pct=result['return_pct'], drawdown_pct=result['drawdown_pct'],
                    trades=result['trades'], net_pnl=net, execution_validity_reported=result['execution_validity_pass'],
                    audit_status_reported=result['audit_status'], ledger_arithmetic_reconciles=True))
    for row in rows:
        for comparator in ('median','routed'):
            base = next(x for x in rows if x['generation']==row['generation'] and x['scenario']==row['scenario'] and x['recipe']==comparator)
            row[f'delta_return_vs_{comparator}_pp'] = row['return_pct']-base['return_pct']
            row[f'delta_trades_vs_{comparator}'] = row['trades']-base['trades']
    assert all(sha(Path(p)) == h for p,h in dependencies.items())
    out = dict(at=datetime.now(timezone.utc).isoformat(), experiment='olr-exp-0033', inputs=dependencies,
        rows=rows, new_replays=0, original_artifacts_changed=False,
        limits=['Saved-result hash and flat-terminal ledger reconciliation only.',
                'Original audit_status is unverified; not a new source/execution qualification.',
                'Historical populations and execution differ from N0; no pooled estimate.',
                'No stopped study resumed.'])
    with OUT.open('x', encoding='utf8') as f:
        json.dump(out, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(path=str(OUT), accounts=len(rows), all_ledger_arithmetic_reconciles=True)))


if __name__ == '__main__':
    run()

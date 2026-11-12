"""Read-only retained decision-input support census, never a candidate labeler."""
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_round2_assessment_v1 import SOURCE, SCENARIOS, bound, write, csvwrite, lines

REPO = Path(__file__).resolve().parents[1]
OUT = Path('E:/CodexResearch/k_stock_trader/kalcb_round2_assessment_20261017/v1/journal_support')
FIELDS = ('entry_price_ref', 'stop_price', 'risk_per_share', 'portfolio_equity', 'entry_route', 'frontier_selection_score')


def main():
    OUT.mkdir(exist_ok=False)
    sources = {(p, s): SOURCE/'cases'/f'combined_{p}_{s}'/'decisions.jsonl.gz'
               for p in ('is', 'oos') for s in SCENARIOS}
    write(OUT/'binding.json', dict(program=bound(__file__),
        protocol=bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0041-journal-support.md'),
        inputs=[bound(v) for v in sources.values()]))
    summary = []; reason_summary = []
    for (period, scenario), path in sources.items():
        rows = []; groups = defaultdict(list)
        for index, d in enumerate(lines(path), 1):
            if d['decision_code'] not in ('entry', 'entry_rejected'): continue
            m = d['metadata']; actions = [a for a in d['actions'] if a['action_type'] == 'SubmitEntry']
            assert len(actions) <= 1
            row = dict(period=period, scenario=scenario, source_row=index, symbol=d['symbol'],
                timestamp=d['timestamp'], decision_code=d['decision_code'], reason=d['reason'],
                state_snapshot_present=bool(d.get('state_snapshot_ref')), action_qty=actions[0]['qty'] if actions else None,
                **{k: m.get(k) for k in FIELDS})
            row['explicit_geometry_complete'] = all(m.get(k) is not None for k in FIELDS[:4])
            rows.append(row); groups[d['decision_code'], d['reason']].append(row)
        csvwrite(OUT/f'{period}_{scenario}_records.csv', rows)

        def aggregate(records):
            return dict(rows=len(records), dates=len({r['timestamp'][:10] for r in records}),
                issuer_days=len({(r['timestamp'][:10], r['symbol']) for r in records}),
                recorded_event_tuples=len({(r['symbol'], r['timestamp'], r['entry_route'], r['decision_code']) for r in records}),
                explicit_geometry_complete=sum(r['explicit_geometry_complete'] for r in records),
                action_quantity_present=sum(r['action_qty'] is not None for r in records),
                state_snapshot_present=sum(r['state_snapshot_present'] for r in records),
                **{k+'_present':sum(r[k] is not None for r in records) for k in FIELDS})

        for code in ('entry', 'entry_rejected'):
            summary.append(dict(period=period, scenario=scenario, decision_code=code,
                **aggregate([r for r in rows if r['decision_code'] == code])))
        for (code, reason), records in groups.items():
            reason_summary.append(dict(period=period, scenario=scenario, decision_code=code, reason=reason,
                **aggregate(records)))
    csvwrite(OUT/'summary.csv', summary); csvwrite(OUT/'reasons.csv', reason_summary)
    write(OUT/'completion.json', dict(complete=True, cells=8, native_accounts=0, labels=0,
        complete_counterfactual_universe_established=False,
        limitation='Retained entry/rejection attempts only; missing downstream geometry is unknown, not a failed economic outcome.'))


if __name__ == '__main__': main()

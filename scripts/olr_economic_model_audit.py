"""Forward-date forecast falsification; no tuning or portfolio substitution."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtests.strategies.olr.economic_policy import DatedResponseModel


def audit(rows):
    model = DatedResponseModel(rows)
    groups = defaultdict(list)
    for row in rows:
        # Fits freeze at midnight, so intraday checkpoint choice cannot expose
        # same-day outcomes. Labels remain unavailable until final maturity.
        at = datetime.fromisoformat(row['day']).replace(
            hour=14, tzinfo=timezone(timedelta(hours=9)))
        if row['kind'] == 'hold':
            # Entry day is not the holding decision day. Infer only from the
            # explicit decision timestamp when provided; older labels cannot
            # safely be audited as if their holding checkpoint were entry day.
            if not row.get('decision_at'):
                continue
            at = datetime.fromisoformat(row['decision_at'])
        fit = model.fit(at, row['kind'], row['scenario'])
        if fit is None:
            continue
        pred = model.predict(fit, row['x'], fit['days'])
        column = -2 if row['kind'] == 'entry' else -1
        mean = float(pred[:, column].mean())
        # Intercept-only comparator uses exactly the same dated training set
        # and weights every opportunity date equally.
        training = [r for r in rows if r['kind'] == row['kind']
                    and r['scenario'] == row['scenario']
                    and r['day'] in fit['days']
                    and datetime.fromisoformat(r['known_at']) < at.replace(
                        hour=0, minute=0, second=0, microsecond=0)]
        daily = defaultdict(list)
        for r in training:
            daily[r['day']].append(r['y'][column])
        pooled = float(np.mean([np.mean(v) for v in daily.values()]))
        actual = float(row['y'][column])
        groups[(row['kind'], row['scenario'])].append(dict(
            id=row['id'], day=row['day'], predicted=mean, actual=actual,
            pooled=pooled, lower_decile=float(np.quantile(pred[:, column], .1))))
    result = {}
    for (kind, scenario), predictions in groups.items():
        by_day = defaultdict(list)
        for r in predictions:
            by_day[r['day']].append(r)
        def balanced(fn):
            return float(np.mean([np.mean([fn(r) for r in values])
                                  for values in by_day.values()]))
        loss = balanced(lambda r: (r['actual']-r['predicted'])**2)
        pooled_loss = balanced(lambda r: (r['actual']-r['pooled'])**2)
        selected = [r for r in predictions if r['predicted'] > 0]
        result[kind+':'+scenario] = dict(
            rows=len(predictions), independent_date_clusters=len(by_day),
            date_balanced_mse=loss, pooled_mse=pooled_loss,
            skill_vs_pooled=1-loss/pooled_loss if pooled_loss else None,
            predicted_return=balanced(lambda r: r['predicted']),
            actual_return=balanced(lambda r: r['actual']),
            lower_decile_breach_rate=balanced(lambda r: float(r['actual'] < r['lower_decile'])),
            selected_positive_rows=len(selected),
            selected_positive_actual_mean=float(np.mean([r['actual'] for r in selected])) if selected else None,
            predictions=predictions)
    return dict(
        purpose='Forecast falsification only; outcomes are unfunded native response lanes, not portfolio returns.',
        holding_rows_skipped=sum(r['kind']=='hold' and not r.get('decision_at') for r in rows),
        metrics=result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = audit(json.loads(args.labels.read_text(encoding='utf-8')))
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(json.dumps({k: {a:b for a,b in v.items() if a!='predictions'}
                      for k,v in output['metrics'].items()}, indent=2))

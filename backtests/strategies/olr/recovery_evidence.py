"""Project authenticated native audit counts into the shared metric contract."""
from math import isfinite


HYGIENE_FIELDS=('same_bar_fill_count','forced_replay_close_count',
                'rejected_order_count','invalid_order_rejection_count')


def hygiene_metrics(audits, results):
    audits=list(audits);results=list(results)
    if not audits or len(audits)!=len(results):
        raise ValueError('Every capital/scenario cell requires native audit evidence')
    metrics={key:0. for key in HYGIENE_FIELDS}
    for audit in audits:
        for key in HYGIENE_FIELDS:
            # Missing evidence must not silently become a reported zero.
            value=float(audit[key])
            if not isfinite(value) or value<0 or not value.is_integer():
                raise ValueError('Invalid native hygiene count: '+key)
            metrics[key]+=value
    metrics['end_open_position_count']=sum(len(result['terminal_inventory']) for result in results)
    return metrics

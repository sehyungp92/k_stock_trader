"""Cross-sectional leadership known at a common completed-opening cutoff."""
from bisect import bisect_left,bisect_right
import math

SOURCES={'leader_daily_return_rank':'daily_return_20d','leader_recent_return_rank':'daily_return_5d',
    'leader_opening_return_rank':'first30_ret','leader_opening_volume_rank':'first30_rel_volume_log'}
COLUMNS=tuple(SOURCES)+('leader_persistence','leader_emergence')
INTERACTIONS=(('daily_return_20d','leader_persistence'),
    ('first30_ret','leader_opening_return_rank'),('leader_daily_return_rank','leader_opening_volume_rank'))

def rank_at_cutoff(rows,trade_date,available_at):
    # Membership is the observed universe at the common decision cutoff, not a
    # filtered list of eventual trade successes or end-of-day data availability.
    from datetime import datetime
    cutoff=datetime.fromisoformat(available_at)
    if cutoff.date().isoformat()!=trade_date:raise ValueError('Leadership cutoff date mismatch')
    cohort=[r for r in rows if r['trade_date']==trade_date and datetime.fromisoformat(r['feature_available_at'])<=cutoff]
    if len({r['symbol'] for r in cohort})!=len(cohort):raise ValueError('Duplicate leadership observation')
    result={r['symbol']:{} for r in cohort}
    for target,source in SOURCES.items():
        def value(r):
            x=r['features'].get(source)
            return float(x) if x is not None and math.isfinite(float(x)) else None
        ordered=sorted(x for r in cohort if (x:=value(r)) is not None)
        for r in cohort:
            x=value(r)
            result[r['symbol']][target]=(bisect_left(ordered,x)+bisect_right(ordered,x))/(2*len(ordered)) if x is not None and ordered else None
    for f in result.values():
        daily=f['leader_daily_return_rank'];opening=f['leader_opening_return_rank']
        f['leader_persistence']=min(daily,opening) if daily is not None and opening is not None else None
        f['leader_emergence']=opening-daily if daily is not None and opening is not None else None
    return result

"""Bounded allocation learning from this account's matured native lifecycles.

No labels from candidate metadata or never-executed opportunities are consumed.
The estimates describe a selected population, not independent alpha evidence.
"""
from datetime import datetime
from math import isfinite, sqrt
from statistics import mean, stdev


def record_cash(position, fill):
    """Called after fill deduplication, before a closed position is archived."""
    row=position.metadata.setdefault('capacity_value_cash',{
        'buy':0.,'sell':0.,'buy_qty':0,'sell_qty':0,'complete_costs':True})
    side=fill.side.lower()
    fee=fill.metadata.get('fee')
    if fee is None or not isfinite(float(fee)) or float(fee)<0:
        row['complete_costs']=False
        fee=0.
    row[side]+=fill.qty*fill.price+(float(fee) if side=='buy' else -float(fee))
    row[side+'_qty']+=fill.qty
    row['last_fill']=fill.timestamp.isoformat()


def matured_labels(state, at):
    if at.tzinfo is None:
        raise ValueError('Matured values require an aware decision timestamp')
    rows=[]
    for s in state.symbols.values():
        for key,p in s.entry_history.items():
            if (s.position is not None and s.position.entry_order_id==p.entry_order_id
                    or s.pending_entry_order_id==p.entry_order_id):
                continue
            row=p.metadata.get('capacity_value_cash')
            if not row or not row['complete_costs'] or row['buy_qty']!=row['sell_qty'] or row['buy']<=0:
                continue
            closed=datetime.fromisoformat(row['last_fill'])
            if closed.tzinfo is None or closed.date()>=at.date():
                continue
            value=10000*(row['sell']/row['buy']-1)
            rows.append({'key':str(key),'closed':closed.isoformat(),
                'rank':str(p.candidate_rank),
                'band':str(p.metadata.get('afternoon_score_band_rule') or 'unknown'),
                'net_bps':min(1500.,max(-1500.,value))})
    return sorted(rows,key=lambda r:(r['closed'],r['key']))[-60:]


def quote(state, candidate, at, config):
    rows=matured_labels(state,at)
    mode=config.capacity_value_mode
    group=(str(candidate.rank) if mode=='rank' else
        str(candidate.metadata.get('afternoon_score_band_rule') or 'unknown'))
    sample=[r['net_bps'] for r in rows if r[mode]==group]
    evidence={'mode':mode,'group':group,'strategy_support':len(rows),'cohort_support':len(sample),
        'latest_label_exit':max((r['closed'] for r in rows),default=None),
        'label_keys':[r['key'] for r in rows], 'multiplier':1.,'status':'insufficient_matured_support'}
    if len(rows)<30 or len(sample)<10 or group=='unknown':
        return 1.,evidence
    values=[r['net_bps'] for r in rows]
    prior=mean(values);n=len(sample)
    estimate=(sum(sample)+20*prior)/(n+20)
    se=stdev(sample)/sqrt(n)
    reference=prior-.25*stdev(values)/sqrt(len(values))
    robust=estimate-.25*se
    strength=config.capacity_value_strength
    scale=1+strength*(robust-reference)/max(abs(reference),100.)
    scale=min(1+.5*strength,max(1-.5*strength,scale))
    evidence.update(multiplier=scale,status='supported',prior_bps=prior,
        shrunk_mean_bps=estimate,robust_net_bps=robust,reference_bps=reference)
    return scale,evidence

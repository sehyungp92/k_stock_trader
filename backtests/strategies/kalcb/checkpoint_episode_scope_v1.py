"""Account each native position lifetime once and retain episode descendants."""
from datetime import datetime


def lifetimes(fills, costs):
    positions=[];current=None;held=0;previous=None
    for f in fills:
        t=datetime.fromisoformat(f['timestamp'])
        if previous is not None and t<previous:raise ValueError('Unsorted fills')
        previous=t
        q=f['qty'];n=q*f['price'];buy=f['side']=='BUY'
        if type(q) is not int or q<=0 or f['side'] not in ('BUY','SELL'):raise ValueError('Invalid fill')
        if current is None:
            if not buy:raise ValueError('Unowned exit')
            current={'entry':t.isoformat(),'fills':[],'gross_cash_flow':0.,'fees':0.,'closed':False}
        held += q if buy else -q
        if held<0:raise ValueError('Over exit')
        current['fills'].append(f)
        current['gross_cash_flow']+=-n if buy else n
        current['fees']+=n*(costs['commission_bps']+(0 if buy else costs['tax_bps_on_sell']))/10000
        current['terminal_inventory']=held
        current['net_cash_flow']=current['gross_cash_flow']-current['fees']
        if held==0:
            current['closed']=True;current['exit']=t.isoformat();positions.append(current);current=None
    if current is not None:positions.append(current)
    return positions


def decomposition(hold, liquidate, costs):
    h=lifetimes(hold,costs);l=lifetimes(liquidate,costs)
    if not h or not l:raise ValueError('Missing initial held lifetime')
    return {'hold_lifetimes':len(h),'liquidate_lifetimes':len(l),
            'both_initial_closed':h[0]['closed'] and l[0]['closed'],
            'all_closed':all(x['closed'] for x in h+l),
            'initial_net_cash_delta':l[0]['net_cash_flow']-h[0]['net_cash_flow'],
            'descendant_net_cash_delta':sum(x['net_cash_flow'] for x in l[1:])-sum(x['net_cash_flow'] for x in h[1:]),
            'episode_net_cash_delta':sum(x['net_cash_flow'] for x in l)-sum(x['net_cash_flow'] for x in h),
            'hold_descendant_net_cash':sum(x['net_cash_flow'] for x in h[1:]),
            'liquidate_descendant_net_cash':sum(x['net_cash_flow'] for x in l[1:])}

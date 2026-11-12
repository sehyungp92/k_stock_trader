"""Extract current-holding action value without crediting later re-entries."""
from datetime import datetime
import math


def holding_at(positions, clock):
    t=datetime.fromisoformat(clock);matches=[]
    for position in positions:
        held=0
        for fill in position['fills']:
            if datetime.fromisoformat(fill['timestamp'])>=t:continue
            held+=fill['qty'] if fill['side']=='BUY' else -fill['qty']
        if held>0:matches.append((position,held))
    if len(matches)>1:raise ValueError('Multiple current lifetimes')
    return matches[0] if matches else (None,0)


def derive(value, event, context, lifetime_fn):
    clock=value['checkpoint_at'];held=value['hold']['checkpoint']['held_quantity'];costs=event['hold']['costs']
    if costs!=event['liquidate']['costs']:raise ValueError('Different paired costs')
    hp=lifetime_fn(event['hold']['fills'],costs);lp=lifetime_fn(event['liquidate']['fills'],costs)
    h,hq=holding_at(hp,clock);l,lq=holding_at(lp,clock)
    if hq!=held or lq!=held:raise ValueError('Current lifetime differs from saved native inventory')
    state=event['hold']['checkpoint'].get('state',{})
    pending=sum(o['qty'] for o in state.get('orders',[]) if o['side']=='BUY' and o['symbol']==value['symbol'])
    denominator=held*context['completed_close']
    if abs(denominator-value['held_notional_krw'])>1e-6:raise ValueError('Wrong held denominator')
    valid=bool(value['paired_label_valid'] and h and l and h['closed'] and l['closed'])
    if value['paired_label_valid'] and not valid:raise ValueError('Valid episode lacks a closed current holding')
    initial=(l['net_cash_flow']-h['net_cash_flow']) if valid else None
    full=value['liquidate_minus_hold_bps'] if valid else None
    if valid:
        recomputed=sum(p['net_cash_flow'] for p in lp)-sum(p['net_cash_flow'] for p in hp)
        if abs(recomputed-full*denominator/10000)>1e-5:raise ValueError('Episode cash decomposition differs from audited target')
    initial_bps=initial/denominator*10000 if valid else None
    rest=full-initial_bps if valid else None
    features=None
    if held>0:
        volume=context['consumed_bars'][-1]['volume'];entry=value['hold']['checkpoint']['entry_price'];atr=context['daily_atr_pct']
        if volume<=0 or entry<=0 or atr<=0 or value['planned_quantity']<=0:raise ValueError('Invalid causal state feature')
        features={'log_held_notional':math.log(denominator),
                  'log_held_to_last_bar_volume':math.log1p(held/volume),
                  'held_to_planned_quantity':held/value['planned_quantity'],
                  'pending_to_held_quantity':pending/held,
                  'marked_entry_return_atr':(context['completed_close']/entry-1)/atr}
    return {'opportunity_id':value['opportunity_id'],'day':value['day'],'symbol':value['symbol'],
            'scenario':value['scenario'],'quantity_scale':value['quantity_scale'],'checkpoint_at':clock,
            'eligibility':value['eligibility'],'held_quantity':held,'planned_quantity':value['planned_quantity'],
            'pending_buy_quantity':pending,'held_notional_krw':denominator,
            'current_holding_label_valid':valid,'label_available_at':value['label_available_at'] if valid else None,
            'current_holding_advantage_bps':initial_bps,'original_episode_advantage_bps':full,
            'other_lifetimes_advantage_bps':rest,'state_features':features,
            'own_features':value['own_features'],'context_features':value['context_features'],
            'no_pending_entry_state':bool(held>0 and pending==0),
            'hold_execution_pass':value['hold']['independent_execution_pass'],
            'hold_deadline_violation':value['hold']['deadline_violation'],
            'liquidate_execution_pass':value['liquidate']['independent_execution_pass'],
            'liquidate_account_pass':value['liquidate']['independent_account_pass'],
            'liquidate_deadline_violation':value['liquidate']['deadline_violation'],
            'new_action_submitted':value['liquidate']['checkpoint']['action_submitted']}

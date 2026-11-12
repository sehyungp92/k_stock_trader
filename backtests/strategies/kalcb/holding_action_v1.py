"""Research-only executable remaining-exposure contracts; no enabled live hook."""
from dataclasses import dataclass
from datetime import timedelta
import math

VERSION='native-holding-action-v1'
FEATURES=('age_minutes','position_weight','book_weight','buying_power_fraction',
    'account_drawdown','session_return','return_atr','mfe_atr','giveback_atr',
    'last15_return_atr','last15_efficiency','vwap_distance_atr','volume_ratio',
    'remaining_capacity_ratio','pending_to_held','peer_return_atr','peer_negative_share',
    'relative_return_atr','opening_return_atr','opening_volume_log')


@dataclass(frozen=True)
class ValueContract:
    target:str='whole_account_action_minus_incumbent_eod_net_equity_bps'
    quantity_basis:str='exact_current_funded_integer_holding'
    continuation:str='unchanged_native_parent_after_one_action'
    risk_target:str='suffix_full_mtm_drawdown_from_inherited_and_new_peaks'
    maturity:str='session_end_then_one_completed_session_embargo'

    def validate(self,raw):
        from dataclasses import asdict
        if raw!=asdict(self):raise ValueError('Incompatible forecast target or quantity contract')


def observations(adapter,broker,batch):
    """Completed observed bars only; available on a recurring 15-minute cadence.

    Sampling is based on each lifetime's actual elapsed completed observations,
    not on realised outcomes, weakness conditions or the parent's exit score.
    """
    from kis_core.trading_calendar import get_trading_calendar
    stamp=batch[0].timestamp;at=stamp+timedelta(minutes=5)
    views=[];present={b.symbol:b for b in batch}
    equity=broker._portfolio_equity();peak=max(broker.equity_curve)
    if equity<=0:raise ValueError('Nonpositive account equity')
    book=sum(p.qty*broker.last_prices[s] for (_,s),p in broker.positions.items())/equity
    peers=[]
    for ss in adapter.state.symbols.values():
        p=ss.position
        if p and ss.bars_today and ss.bars_today[-1].timestamp==stamp:
            atr=ss.candidate.daily_atr
            if atr>0:peers.append((ss.bars_today[-1].close-p.entry_price)/atr)
    for symbol,ss in sorted(adapter.state.symbols.items()):
        p=ss.position
        if not p or symbol not in present:continue
        if p.metadata.get('entry_route')!='first30_open_anchor':continue
        if p.hold_bars<2 or (p.hold_bars-2)%3:continue
        reason=None
        own=[o for o in broker.orders if o.symbol==symbol]
        actual=broker.positions.get(('KALCB',symbol))
        if actual is None or actual.qty!=p.qty_open:raise ValueError('Core/broker held quantity differs')
        if p.exit_in_flight or p.partial_order_id or any(o.side=='SELL' for o in own):reason='existing_exit_owned_by_parent'
        if any(o.strategy_id!='KALCB' for o in own):reason='shared_symbol_order'
        if p.qty_open<2:reason='integer_partial_alias'
        bars=ss.bars_today;bar=present[symbol];atr=float(ss.candidate.daily_atr)
        if not bars or bars[-1].timestamp!=stamp or any(b.timestamp+timedelta(minutes=5)>at for b in bars):
            raise ValueError('Holding features contain future or mismatched observations')
        if atr<=0 or bar.volume<=0:reason='unsupported_price_or_volume'
        if p.qty_open*bar.close<.01*equity:reason='below_material_holding'
        _,close=get_trading_calendar().session_hours(stamp.date())
        remaining=(at.replace(hour=close.hour,minute=close.minute)-at).total_seconds()/300-2
        if remaining<=0:reason='no_continuous_execution_horizon'
        row={'version':VERSION,'symbol':symbol,'day':str(stamp.date()),'available_at':at.isoformat(),
             'signal_bar':stamp.isoformat(),'entry_order_id':p.entry_order_id,'qty':p.qty_open,
             'equity':equity,'running_peak':peak,'entry_price':p.entry_price,
             'management_reference':p.entry_price,'management_risk_per_share':p.risk_per_share,
             'current_price':bar.close,'held_bars':p.hold_bars,'reason':reason or 'eligible'}
        if reason:views.append(row);continue
        recent=bars[-3:];returns=[b.close-b.open for b in recent]
        volume=sum(b.volume for b in bars);vwap=sum((b.high+b.low+b.close)*b.volume/3 for b in bars)/max(volume,1)
        post=[b for b in bars if b.timestamp>=p.entry_time]
        high=max([p.entry_price]+[b.high for b in post]);current=(bar.close-p.entry_price)/atr
        pending=sum(o.qty for o in own if o.side=='BUY')
        md=p.metadata;peer=sum(peers)/max(1,len(peers))
        f=dict(age_minutes=(at-p.entry_time).total_seconds()/60,position_weight=p.qty_open*bar.close/equity,
            book_weight=book,buying_power_fraction=broker._buying_power_available()/equity,
            account_drawdown=1-equity/peak,session_return=adapter._portfolio_view(broker).session_return_pct,
            return_atr=current,mfe_atr=(high-p.entry_price)/atr,giveback_atr=(high-bar.close)/atr,
            last15_return_atr=(recent[-1].close-recent[0].open)/atr,
            last15_efficiency=sum(returns)/max(sum(abs(x) for x in returns),1e-9),
            vwap_distance_atr=(bar.close-vwap)/atr,
            volume_ratio=sum(b.volume for b in recent)/max(1.,3*ss.candidate.expected_5m_volume),
            remaining_capacity_ratio=p.qty_open/max(1.,remaining*min(b.volume for b in recent)*.005),
            pending_to_held=pending/p.qty_open,peer_return_atr=peer,
            peer_negative_share=sum(x<0 for x in peers)/max(1,len(peers)),relative_return_atr=current-peer,
            opening_return_atr=float(md.get('first30_ret',0))*float(md.get('entry_price_ref',bar.close))/atr,
            opening_volume_log=math.log1p(max(0.,float(md.get('first30_rel_volume',0)))))
        if set(f)!=set(FEATURES) or not all(math.isfinite(x) for x in f.values()):raise ValueError('Invalid causal holding state')
        row['features']=f;views.append(row)
    return views


def apply(adapter,broker,row,arm):
    """Submit through the native adapter; cash/inventory change only on fills."""
    from datetime import datetime
    from strategy_common.actions import CancelOrders,SubmitExit,SubmitPartialExit
    from strategy_kalcb.core.logic import _exit_metadata
    if arm=='hold':return None
    if arm not in ('half','exit') or row['reason']!='eligible':raise ValueError('Invalid holding action')
    ss=adapter.state.symbols[row['symbol']];p=ss.position
    if not p or p.qty_open!=row['qty'] or p.entry_order_id!=row['entry_order_id']:raise ValueError('Stale holding action')
    stamp=datetime.fromisoformat(row['signal_bar']);available=datetime.fromisoformat(row['available_at'])
    if not ss.bars_today or ss.bars_today[-1].timestamp!=stamp or available!=stamp+timedelta(minutes=5):
        raise ValueError('Stale or anticipatory holding clock')
    if p.entry_price!=row['management_reference'] or p.risk_per_share!=row['management_risk_per_share']:
        raise ValueError('Changed management reference')
    own=[o for o in broker.orders if o.symbol==row['symbol']]
    if p.exit_in_flight or p.partial_order_id or any(o.side!='BUY' or o.strategy_id!='KALCB' for o in own):
        raise ValueError('Cannot replace an existing native exit')
    qty=p.qty_open//2 if arm=='half' else p.qty_open
    if not 0<qty<=p.qty_open:raise ValueError('Invalid integer reduction')
    reason='research_remaining_account_'+arm+'_v1'
    meta=_exit_metadata(p,reason,{'order_role':'TP' if arm=='half' else 'EXIT',
        'holding_action_version':VERSION,'action_parent_qty':p.qty_open,'action_available_at':row['available_at']})
    actions=[]
    if own:
        # Symbol cancellation is safe only because there are no other orders.
        if len(own)!=1 or own[0].order_id!=p.entry_order_id:raise ValueError('Unmatched buy remainder')
        adapter.state.order_roles.setdefault(own[0].order_id,{})['research_cancel_preserves_filled']=True
        actions.append(CancelOrders('KALCB',p.symbol,reason,{'research_cancel_preserves_filled':True}))
    if arm=='half':
        p.partial_order_id='__pending__'
        actions.append(SubmitPartialExit('KALCB',p.symbol,qty,'MARKET',None,reason,meta))
    else:
        p.exit_in_flight=True;p.metadata['liquidation_reason']=reason
        actions.append(SubmitExit('KALCB',p.symbol,qty,'MARKET',None,reason,meta))
    clock=broker._decision_available_at
    broker._decision_available_at=available
    try:adapter._submit_actions(actions,broker,stamp)
    finally:broker._decision_available_at=clock
    return {'arm':arm,'qty':qty,'available_at':row['available_at'],'cancelled_buy_remainder':bool(own)}

"""Explicit research-only boundary for an action-independent liquidity panel."""
from datetime import datetime,timedelta
from .liquidity_v6 import request_bound

KEY='structural_liquidity_v6'

def override_forecast(state,bar,config,evidence,limits):
    context=state.meta.get(KEY)
    if context is None:return None
    day=bar.timestamp.date().isoformat();symbol=str(bar.symbol).zfill(6)
    if context['day']!=day or context['model']!='broad_median':
        raise ValueError('Invalid structural liquidity context')
    if symbol not in context['forecasts']:
        raise ValueError('Missing source-qualified liquidity forecast')
    forecast=context['forecasts'][symbol]
    if forecast['symbol']!=symbol or forecast['day']!=day or forecast['model']!=context['model']:
        raise ValueError('Forecast identity mismatch')
    qty=request_bound(forecast,config.capacity_participation,bar.timestamp+timedelta(minutes=5))
    evidence['exit']=dict(quantity=qty,samples=context['samples'][symbol],status=forecast['status'],
        quantile=.5,structural_forecast=forecast,panel_sha256=context['panel_sha256'])
    if config.capacity_exit_quantile!=.5:
        raise ValueError('The registered structural experiment uses median only')
    return min([*limits,qty]),qty,evidence

def day_context(panel,day,panel_sha):
    from .liquidity_v6 import LOOKBACK_DAYS
    cutoff=(datetime.fromisoformat(day)-timedelta(days=LOOKBACK_DAYS)).date().isoformat()
    forecasts={s:panel.forecast(day,s,'broad_median') for s in panel.universe}
    samples={s:[[(datetime.fromisoformat(r['auction_available_at'])-timedelta(minutes=5)).isoformat(),r['auction_volume']]
                for r in panel.by_symbol[s] if cutoff<=r['day']<day and r['auction_volume'] is not None][-20:]
             for s in panel.universe}
    return dict(day=day,model='broad_median',forecasts=forecasts,samples=samples,panel_sha256=panel_sha)

def observed_prefix(panel,through):
    """Full observed panel prefix, independent of positions, preserved at restart.

    Source file hashes may differ across independently assembled windows; their
    actual values and clocks must be equal. Both database hashes remain bound
    separately in the terminal/restart receipts.
    """
    return [{k:r[k] for k in ('day','symbol','prefix_volume','prefix_available_at','prefix_rows','prefix_complete',
                              'auction_volume','auction_available_at','auction_disposition')}
            for key,r in sorted(panel.rows.items()) if r['day']<=through]

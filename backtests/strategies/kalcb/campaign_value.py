"""Prior-confirmed campaign structure observed at the completed opening cutoff.

The box is fixed before today's open. Its start is the AVWAP anchor; no future
extremum or realised payoff participates in anchor/box selection.
"""
from datetime import date, datetime, timedelta
from strategy_kalcb.structural_campaign import build_structural_campaign, _prior_rows, _row_date

COLUMNS = (
    'campaign_available', 'campaign_box_length', 'campaign_box_height_atr',
    'campaign_box_containment', 'campaign_box_squeeze', 'campaign_trend_sign',
    'campaign_accumulation', 'campaign_prior_distance_atr',
    'campaign_open_distance_atr', 'campaign_close_distance_atr',
    'campaign_low_distance_atr', 'campaign_avwap_distance_atr',
    'campaign_opening_above_share', 'campaign_retest_hold',
    'campaign_prior_volume_ratio', 'campaign_ma_alignment',
)
INTERACTIONS = (
    ('daily_return_20d','campaign_box_squeeze'),
    ('daily_return_20d','campaign_opening_above_share'),
    ('first30_ret','campaign_close_distance_atr'),
    ('first30_ret','campaign_prior_volume_ratio'),
)


def features(row, daily_rows):
    day=date.fromisoformat(row['trade_date'])
    cutoff=datetime.fromisoformat(row['feature_available_at'])
    if cutoff.date()!=day:raise ValueError('Campaign decision date mismatch')
    prior=_prior_rows(daily_rows,day)
    bars=sorted((b for b in row['opening_bars']
        if datetime.fromisoformat(b['timestamp'])+timedelta(minutes=5)<=cutoff),
        key=lambda b:b['timestamp'])
    if not bars or any(b['timestamp'][:10]!=row['trade_date'] for b in bars):
        raise ValueError('Campaign requires same-session completed opening prefix')
    campaign=build_structural_campaign(row['symbol'],day,prior)
    box=campaign.compression_box
    result={k:None for k in COLUMNS}
    result.update(campaign_available=float(bool(box and box.available)),
        campaign_trend_sign=float(campaign.daily_trend_sign),
        campaign_accumulation=campaign.accumulation_score/100,
        campaign_ma_alignment=campaign.trend_alignment_detail.get('price_vs_sma50'))
    if not box or not box.available:
        return result,{'available':False,'prior_through':_row_date(prior[-1]).isoformat() if prior else None}
    atr=float(row['features']['daily_atr_pct'])*float(prior[-1]['close'])
    if atr<=0:raise ValueError('Nonpositive campaign ATR')
    anchor=[r for r in prior if _row_date(r).isoformat()>=box.start_date]
    volume=sum(float(r['volume']) for r in anchor)
    avwap=sum((float(r['high'])+float(r['low'])+float(r['close']))/3*float(r['volume']) for r in anchor)/volume if volume else None
    close=bars[-1]['close'];low=min(b['low'] for b in bars);level=box.high
    above=sum(b['close']>=level for b in bars)/len(bars)
    result.update(campaign_box_length=box.length,campaign_box_height_atr=(box.high-box.low)/atr,
        campaign_box_containment=box.containment,campaign_box_squeeze=box.squeeze_pct,
        campaign_prior_distance_atr=(float(prior[-1]['close'])-level)/atr,
        campaign_open_distance_atr=(bars[0]['open']-level)/atr,
        campaign_close_distance_atr=(close-level)/atr,campaign_low_distance_atr=(low-level)/atr,
        campaign_avwap_distance_atr=(close-avwap)/atr if avwap else None,
        campaign_opening_above_share=above,
        campaign_retest_hold=float(low<=level and close>=level),
        campaign_prior_volume_ratio=sum(float(r['volume']) for r in prior[-5:])/5/
            max(1,sum(float(r['volume']) for r in prior[-20:])/20))
    return result,{'available':True,'prior_through':_row_date(prior[-1]).isoformat(),
        'box_start':box.start_date,'box_end':box.end_date,'box_high':box.high,
        'box_low':box.low,'avwap_anchor':box.start_date,'prior_avwap':avwap}

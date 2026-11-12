"""Prove first-OOS predictors carry the IS market history across March 31."""
from collections import defaultdict
from dataclasses import asdict
from datetime import date
import json,math,sqlite3,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_full_oos_recovery import ROOT,FULL,read,write,bound,rows
from backtests.strategies.kalcb.continuation_value import prior_path
from backtests.strategies.kalcb import first30_signal_sweep as native
from backtests.core.input_manifest import InputManifest
from scripts.kalcb_executed_value_study import config_for

def main():
    old=Path('C:/Users/sehyu/.codex/tmp/kalcb_calibrated_execution_20261007/market.sqlite')
    new=FULL/'forward_features/market.sqlite';conn=sqlite3.connect('file:'+str(old)+'?mode=ro',uri=True)
    actual_previous=conn.execute('select max(day) from bars where day<?',('2026-04-01',)).fetchone()[0]
    if actual_previous!='2026-03-31':raise ValueError('IS does not meet OOS boundary')
    def bars(db):
        c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True);result=defaultdict(list)
        for s,ts,o,h,l,close,v in c.execute('select symbol,ts,o,h,l,c,v from bars where day=? order by symbol,ts',(actual_previous,)):
            result[s].append(dict(timestamp=ts,open=o,high=h,low=l,close=close,volume=v))
        c.close();return result
    original=bars(old);halo=bars(new);errors=[];checks=0
    opening=[r for r in rows(FULL/'forward_features/continuation/opportunities.jsonl') if r['trade_date']=='2026-04-01']
    timed=[r for r in rows(FULL/'forward_features/checkpoint/opportunities.jsonl') if r['trade_date']=='2026-04-01' and r['action']['ready'] and r['guard_at_rank1']]
    for kind,items in [('opening',opening),('checkpoint',timed)]:
        for row in items:
            symbol=row['symbol']
            if original.get(symbol)!=halo.get(symbol):errors.append({'kind':kind,'symbol':symbol,'type':'IS_bar_halo_mismatch'})
            uninterrupted=prior_path(original.get(symbol,[]),'2026-04-01')
            for key,value in uninterrupted.items():
                current=row['features'].get(key);checks+=1
                equal=value==current if value is None or current is None else abs(value-current)<=1e-12
                if not equal:errors.append({'kind':kind,'symbol':symbol,'feature':key,'full_history':value,'forward':current})
    generation=InputManifest(FULL/'forward_inputs/manifest.json');daily_root=generation.root/'data/krx_daily_parquet'
    with generation.activate():daily=native._load_daily_rows(daily_root,sorted({r['symbol'] for r in opening}),date(2026,4,1))
    history_extent={}
    for row in opening:
        symbol=row['symbol'];past=[r for r in daily[symbol] if str(r['date'])<'2026-04-01']
        feature=native._daily_feature_from_prior(symbol,date(2026,4,1),past)
        if feature is None:raise ValueError('Daily OOS warm-up missing')
        values={'daily_return_5d':feature.return_5d,'daily_return_20d':feature.return_20d,
            'daily_return_60d':feature.return_60d,'daily_atr_pct':feature.atr14/feature.prev_close,
            'daily_adv20_krw_log':math.log1p(feature.adv20_krw)}
        for key,value in values.items():
            current=row['features'].get(key);checks+=1
            if current is None or abs(value-current)>1e-10:errors.append({'kind':'daily','symbol':symbol,'feature':key,'full_history':value,'forward':current})
        history_extent[symbol]={'rows':len(past),'from':str(past[0]['date']),'through':str(past[-1]['date'])}
    cfg=config_for(read(ROOT/'plan.json'))
    cross_session={'risk_daily_vol_target':cfg.risk_daily_vol_target,'risk_dynamic_notional_enabled':cfg.risk_dynamic_notional_enabled}
    if cross_session['risk_daily_vol_target'] or cross_session['risk_dynamic_notional_enabled']:
        raise ValueError('Active cross-session portfolio control requires state handoff test')
    historical_sessions=read(Path('E:/CodexResearch/k_stock_trader/kalcb_continuation_20261006/prepared_v1/prepared.json'))['sessions']
    sessions=historical_sessions+read(ROOT/'preparation.json')['sessions']
    if sessions!=sorted(set(sessions)):raise ValueError('Noncontinuous or repeated training calendar')
    payload={'passed':not errors,'first_OOS':'2026-04-01','previous_IS_session':actual_previous,
        'opening_rows':len(opening),'checkpoint_rows':len(timed),'feature_comparisons':checks,'errors':errors,
        'daily_history_extent':history_extent,'IS_sessions_available_at_OOS_start':len(historical_sessions),
        'first_OOS_session_index':sessions.index('2026-04-01'),'cross_session_portfolio_controls':cross_session,
        'sources':{'IS_database':bound(old),'OOS_halo_database':bound(new),'market_generation':bound(FULL/'forward_inputs/manifest.json')},
        'program':bound(__file__),
        'interpretation':'The OOS boundary changes the scoring period, not the historical daily or prior-intraday feature state. Opening range and intraday VWAP reset each trading session by design. A one-session training embargo still applies.',
        'account_scope':'The displayed IS and OOS accounts each start at KRW100m. Market-history continuity is distinct from compounded-capital continuity; a larger carried account may have different capacity effects.'}
    write(ROOT/'boundary_continuity.json',payload)
    print(json.dumps({k:payload[k] for k in ('passed','first_OOS','previous_IS_session','opening_rows','checkpoint_rows','feature_comparisons','errors','IS_sessions_available_at_OOS_start','cross_session_portfolio_controls')}),flush=True)
    if errors:raise ValueError('IS/OOS market history continuity mismatch')

if __name__=='__main__':main()

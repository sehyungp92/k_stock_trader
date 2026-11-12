"""Read-only targeted source investigation; never alters market data or orders."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from scripts.portfolio_targeted_validation import ROOT, REPO, read, bound
from backtests.strategies.kalcb.executed_value import write_json


def investigate(root=ROOT, *, fetch=False):
    import pandas as pd
    from scripts.update_kis_intraday import fetch_kis_1m_page, _normalize_kis_minute_row, KST
    inspection=read(root/'oos_input_inspection/window_input_audit.json')
    target=root/'source_resolution'
    target.mkdir(parents=True,exist_ok=False)
    api=None
    if fetch:
        from scripts.update_kis_intraday import _load_dotenv
        from kis_core import KoreaInvestAPI,KoreaInvestEnv,build_kis_config_from_env
        _load_dotenv(REPO/'.env')
        api=KoreaInvestAPI(KoreaInvestEnv(build_kis_config_from_env()))
    records=[]
    for symbol in sorted({r['symbol'] for r in inspection['issues']}):
        paths={tf:next((REPO/'data/kis_intraday_parquet'/symbol).glob(symbol+'_'+tf+'_*_20260710.parquet')) for tf in ('1m','5m')}
        dp=next((REPO/'data/krx_daily_parquet/daily_ohlcv'/symbol).glob('*.parquet'))
        frames={tf:pd.read_parquet(p).sort_values('timestamp') for tf,p in paths.items()}
        for f in frames.values():f['day']=f.timestamp.dt.tz_convert('Asia/Seoul').dt.strftime('%Y-%m-%d')
        daily=pd.read_parquet(dp);daily['day']=pd.to_datetime(daily.date).dt.strftime('%Y-%m-%d')
        for issue in inspection['issues']:
            if issue['symbol']!=symbol:continue
            day=issue['date']; f=frames['5m'][frames['5m'].day==day]; d=daily[daily.day==day].iloc[0]
            observed={'open':float(f.iloc[0].open),'high':float(f.high.max()),'low':float(f.low.min()),
                      'close':float(f.iloc[-1].close),'volume':float(f.volume.sum())}
            reference={k:float(d[k]) for k in observed}
            row={**issue,'sources':[bound(p) for p in (*paths.values(),dp)],'observed':observed,
                 'daily':reference,'differences':{k:observed[k]-reference[k] for k in observed},'gaps':[]}
            for clock in issue['missing']:
                at=datetime.fromisoformat(day+'T'+clock).replace(tzinfo=KST); end=at+timedelta(minutes=5)
                old=frames['1m'][(frames['1m'].timestamp>=at)&(frames['1m'].timestamp<end)]
                gap={'at':at.isoformat(),'archived_1m_rows':len(old),'archived_1m_volume':float(old.volume.sum())}
                if api is not None:
                    try:
                        raw=fetch_kis_1m_page(api,symbol,end,market_code='J')
                        name=symbol+'_'+day+'_'+clock.replace(':','')+'.json';write_json(target/name,raw)
                        normalized=[r for x in raw if (r:=_normalize_kis_minute_row(x)) is not None]
                        current=[r for r in normalized if at<=r['timestamp']<end]
                        same_day=[r for r in normalized if str(r['timestamp'].date())==day]
                        gap.update(provider_page=bound(target/name),provider_rows=len(raw),provider_same_day_rows=len(same_day),
                            provider_gap_rows=len(current),provider_gap_volume=sum(r['volume'] for r in current))
                    except Exception as exc:
                        gap['provider_error_type']=type(exc).__name__
                row['gaps'].append(gap)
            exact=all(v==0 for v in row['differences'].values())
            row['disposition']='sparse_archive_exact_daily_ohlcv_conservation' if exact else 'unresolved_interval_volume_location'
            row['qualification']='Requires source-bound resolution acceptance; no synthetic prints or residual-volume assignment.'
            records.append(row)
    payload={'inspection':bound(root/'oos_input_inspection/window_input_audit.json'),'sessions':records,
        'exact_daily_conservation_sessions':sum(r['disposition']=='sparse_archive_exact_daily_ohlcv_conservation' for r in records),
        'unresolved_sessions':sum(r['disposition']=='unresolved_interval_volume_location' for r in records),
        'performance_evaluated':False,'source_gate_pass':not records,
        'note':'A regular-grid gap is not necessarily missing trades. Preserve sparse sources; daily volume cannot locate missing interval execution capacity.'}
    write_json(target/'resolution.json',payload)
    print(json.dumps({k:v for k,v in payload.items() if k not in ('sessions','inspection')}),flush=True)


def residuals(root=ROOT):
    """Try the existing exact turnover identity for the three residual sessions."""
    import pandas as pd
    from scripts.investigate_incompatible_sessions import collect
    from scripts.update_kis_intraday import fetch_kis_1m_bars, KST, _load_dotenv
    from kis_core import KoreaInvestAPI,KoreaInvestEnv,build_kis_config_from_env
    from backtests.core.daily_volume_residual import verify_daily_close_residual,FIELDS
    source=root/'source_resolution/resolution.json';target=root/'source_residuals'
    target.mkdir(parents=True,exist_ok=False)
    collect(source,target,use_pykrx=False,dispositions=['unresolved_interval_volume_location'])
    _load_dotenv(REPO/'.env');api=KoreaInvestAPI(KoreaInvestEnv(build_kis_config_from_env()))
    results=[]
    for row in read(source)['sessions']:
        if row['disposition']!='unresolved_interval_volume_location':continue
        symbol,day=row['symbol'],row['date'];folder=target/symbol/day
        folder.mkdir(parents=True,exist_ok=False)
        first=datetime.fromisoformat(day+'T09:00:00').replace(tzinfo=KST)
        last=datetime.fromisoformat(day+'T15:35:00').replace(tzinfo=KST)
        raw,pages=fetch_kis_1m_bars(api,symbol,first,last,market_code='J',extra_sleep_sec=.1,max_pages=8,
                                   raw_page_root=folder/'provider_responses')
        raw.to_parquet(folder/'raw_1m.parquet',index=False)
        provider=[x for p in (folder/'provider_responses').rglob('*.json') for x in read(p)['rows']]
        archive=pd.read_parquet(Path(row['sources'][0]['path']))
        archive=archive[archive.timestamp.dt.tz_convert('Asia/Seoul').dt.strftime('%Y-%m-%d')==day].sort_values('timestamp')
        receipt=read(target/'kis_daily'/(symbol+'_'+day+'_original.json'))
        outcome={'symbol':symbol,'date':day,'pages':pages,'source_resolution':bound(source),
                 'provider_files':[bound(p) for p in folder.rglob('*') if p.is_file()]}
        try:
            pd.testing.assert_frame_equal(raw[FIELDS].reset_index(drop=True),archive[FIELDS].reset_index(drop=True),check_dtype=False)
            proof=verify_daily_close_residual(raw,row['daily'],receipt,provider,first.date(),symbol)
            outcome.update(status='exact_residual_identity_pass',proof=proof,archive_equals_current_provider=True)
        except (ValueError,AssertionError) as exc:
            outcome.update(status='source_resolution_failed',reason=str(exc)[:1500])
        results.append(outcome)
    write_json(target/'resolution.json',{'source':bound(source),'sessions':results,
        'all_residuals_resolved':all(r['status']=='exact_residual_identity_pass' for r in results),'performance_evaluated':False})
    print(json.dumps({'residual_checks':[{k:r[k] for k in ('symbol','date','status')} for r in results]}),flush=True)


if __name__=='__main__':
    if '--residuals' in sys.argv:residuals()
    else:investigate(fetch='--fetch' in sys.argv)

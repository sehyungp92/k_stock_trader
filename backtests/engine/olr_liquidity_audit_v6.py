"""Independent source projection and native request arithmetic for v6."""
from collections import defaultdict
from datetime import date,datetime,time,timedelta
import gzip,json,math

def verify_projection(connection,panel,database_hash):
    from kis_core.trading_calendar import get_trading_calendar
    from strategy_common.clock import KST
    from strategy_common.execution_session import session_cutoff,bar_execution_issue,bar_has_periodic_auction
    from strategy_common.market import MarketBar
    calendar=get_trading_calendar();universe=set(panel['universe'])
    if len(universe)!=103 or panel['coarse_database_sha256']!=database_hash:raise ValueError('Unbound liquidity panel')
    indexed={(r['day'],r['symbol']):r for r in panel['rows']}
    if len(indexed)!=len(panel['rows']) or len(indexed)!=len(panel['sessions'])*103:raise ValueError('Incomplete liquidity scope')
    for day_text in panel['sessions']:
        day=date.fromisoformat(day_text);opening,closing=calendar.session_hours(day)
        if not calendar.is_trading_day(day):raise ValueError('Non-session observation')
        start=datetime.combine(day,opening,tzinfo=KST)
        available=datetime.combine(day,session_cutoff(day,time(14,30)),tzinfo=KST)
        groups=defaultdict(list)
        for symbol,stamp,*values in connection.execute('SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day=? ORDER BY symbol,ts',(day_text,)):
            if symbol in universe:groups[symbol].append((datetime.fromisoformat(stamp),*values))
        for symbol in universe:
            row=indexed[(day_text,symbol)];raw=groups[symbol]
            prefix=[b for b in raw if start<=b[0] and b[0]+timedelta(minutes=5)<=available]
            expected=set();tick=start
            while tick+timedelta(minutes=5)<=available:
                halted=any(datetime.combine(day,a,tzinfo=KST)<=tick and tick+timedelta(minutes=5)<=datetime.combine(day,z,tzinfo=KST) for a,z,_ in calendar.intraday_halts_for(symbol,day))
                if not halted:expected.add(tick)
                tick+=timedelta(minutes=5)
            prohibited=bool(calendar.periodic_auction_sources(symbol,day) or calendar.suspension_source(symbol,day))
            complete=bool(expected) and expected=={r[0] for r in prefix} and not prohibited
            volume=sum(r[-1] for r in prefix) if complete else None
            if row['prefix_volume']!=volume or row['prefix_complete']!=complete or row['prefix_rows']!=len(prefix) or row['prefix_available_at']!=available.isoformat():
                raise ValueError('Prefix source projection mismatch')
            match=datetime.combine(day,calendar.closing_match_time(symbol,day),tzinfo=KST)
            auction=[r for r in raw if r[0]==match];actual=None;status='missing_source'
            if prohibited:actual=0.;status='native_execution_prohibited'
            elif auction:
                r=auction[0];b=MarketBar(symbol,r[0],'5m',*r[1:],source_fingerprint=database_hash)
                if not bar_execution_issue(b) and not bar_has_periodic_auction(b) and b.open==b.high==b.low==b.close:
                    actual=float(b.volume);status='source_qualified_auction'
                elif b.volume==0 and b.open==b.high==b.low==b.close and b.close>0:
                    actual=0.;status='observed_zero_volume'
                else:status='unsupported_source_observation'
            if row['auction_volume']!=actual or row['auction_disposition']!=status or row['auction_available_at']!=(match+timedelta(minutes=5)).isoformat() or row['source_hash']!=database_hash:
                raise ValueError('Auction source projection mismatch')
    return dict(passed=True,rows_recomputed=len(indexed),sessions=len(panel['sessions']),symbols=103,
                source_database_sha256=database_hash,independent_source_projection=True)

def audit_requests(path,panel,mode,panel_hash,participation):
    counts=defaultdict(int);by_symbol=defaultdict(list)
    for r in panel['rows']:
        if r['auction_volume'] is not None:by_symbol[r['symbol']].append(r)
    for rows in by_symbol.values():rows.sort(key=lambda r:r['day'])
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            event=json.loads(line);counts['events']+=1
            meta=event.get('metadata',{});ev=meta.get('capacity_forecast')
            if not ev:continue
            counts['forecast_decisions']+=1
            actual=ev['exit'].get('structural_forecast')
            if mode=='off':
                if actual is not None:raise ValueError('Treatment leaked into control')
                continue
            if actual is None:raise ValueError('Missing structural forecast evidence')
            day=actual['day'];symbol=actual['symbol'];at=datetime.fromisoformat(ev['asof'])
            earliest=(date.fromisoformat(day)-timedelta(days=60)).isoformat()
            rows=[r for r in by_symbol[symbol] if earliest<=r['day']<day][-20:]
            values=sorted(r['auction_volume'] for r in rows);n=len(values)
            median=(values[n//2] if n%2 else (values[n//2-1]+values[n//2])/2) if n else None
            expected=math.floor(median*participation) if median is not None else 0
            if actual['predicted_volume']!=median or ev['exit']['quantity']!=expected or ev['exit']['panel_sha256']!=panel_hash:
                raise ValueError('Independent request forecast mismatch')
            if datetime.fromisoformat(actual['available_at'])>at or day!=at.date().isoformat():raise ValueError('Future/stale request forecast')
            if meta.get('capacity_approved_qty',0)>expected:raise ValueError('Requested size exceeds forecast bound')
            counts['structural_decisions']+=1
    if mode!='off' and not counts['structural_decisions']:raise ValueError('No structural forecast activation')
    return dict(passed=True,mode=mode,counts=dict(counts),independent_request_arithmetic=True)

"""C16 time integral of observed minute marks and actual funded inventory."""
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_round2_assessment_v1 import read,write,csvwrite,bound,SOURCE,SCENARIOS

ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_round2_assessment_20261017/v1')
OUT=ROOT/'exposure_v2'


def main():
    OUT.mkdir(exist_ok=False)
    write(OUT/'binding.json',dict(program=bound(__file__),method='Piecewise constant observed gross/equity and negative-cash/equity between consecutive intraday minute marks. No overnight duration. Fill applied at open-after-fill; previous-minute close precedes same-clock fill. Integration clipped to frozen regular/special-session open/close. Gaps retain prior prices; minute-source qualification remains false.'))
    frozen=SOURCE/'runtime/is_cap2/inputs/files'
    sys.path.insert(0,str(frozen))
    from kis_core.trading_calendar import get_trading_calendar
    calendar=get_trading_calendar()
    write(OUT/'calendar_binding.json',dict(inputs=[bound(frozen/x) for x in ('kis_core/trading_calendar.py','kis_core/data/krx_sessions.yaml','kis_core/data/krx_holidays.yaml')],correction=bound(Path(__file__).resolve().parents[1]/'docs/optimisation/kalcb/experiments/kalcb-exp-0041-exposure-correction.md')))
    rows=[]
    for p in ('is','oos'):
        for sc in SCENARIOS:
            fills=read(SOURCE/'cases'/f'combined_{p}_{sc}'/'fills.json');byts=defaultdict(list)
            for f in fills:byts[datetime.fromisoformat(f['timestamp'])].append(f)
            marks=list(csv.DictReader((ROOT/'minute_marks'/f'{p}_{sc}_marks.csv').open()))
            cash=1e8;inventory=defaultdict(int);grossintegral=borrowintegral=heldintegral=0.;seconds=0.;peakq=0;peakgross=0
            last=None;events=0
            for m in marks:
                at=datetime.fromisoformat(m['at']);nav=float(m['nav'])
                if last and last[0].date()==at.date():
                    start,end=calendar.session_hours(at.date())
                    lo=max(last[0],datetime.combine(at.date(),start,at.tzinfo))
                    hi=min(at,datetime.combine(at.date(),end,at.tzinfo))
                    dt=max(0,(hi-lo).total_seconds());seconds+=dt
                    grossintegral+=dt*last[1];borrowintegral+=dt*last[2];heldintegral+=dt*last[3]
                if m['phase']=='open_after_fill':
                    for f in byts[at]:
                        sign=1 if f['side']=='BUY' else -1
                        cash-=sign*f['qty']*f['price']+f['metadata']['fee'];inventory[f['symbol']]+=sign*f['qty'];events+=1
                gross=(nav-cash)/nav;borrow=max(0,-cash)/nav;held=sum(q>0 for q in inventory.values())
                if gross < -1e-10:raise ValueError('Negative long inventory mark')
                peakq=max(peakq,held);peakgross=max(peakgross,gross);last=(at,gross,borrow,held)
            assert events==len(fills) and not any(inventory.values())
            rows.append(dict(period=p,scenario=sc,observed_session_hours=seconds/3600,
                gross_equity_hours=grossintegral/3600,borrowed_equity_hours=borrowintegral/3600,
                position_hours=heldintegral/3600,time_mean_gross_equity=grossintegral/seconds,
                time_mean_borrowed_equity=borrowintegral/seconds,time_mean_concurrent_positions=heldintegral/seconds,
                max_concurrent_positions=peakq,max_observed_gross_equity=peakgross,strict_minute_source_qualified=False))
    csvwrite(OUT/'summary.csv',rows);write(OUT/'completion.json',dict(complete=True,cells=8))


if __name__=='__main__':main()

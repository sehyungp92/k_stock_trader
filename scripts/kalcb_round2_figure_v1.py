"""Static assessment figure from registered, already calculated C16 evidence."""
import csv
from datetime import datetime,timedelta
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_round2_assessment_v1 import read,write,bound,SOURCE,SCENARIOS

ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_round2_assessment_20261017/v1')


def main():
    out=ROOT/'figures';out.mkdir(exist_ok=False)
    write(out/'binding.json',dict(program=bound(__file__),scope='Visualisation of existing registered results; no new treatment or statistic selection.'))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,ax=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    colors=['#126B83','#C97820','#58734B','#A74854']
    for sc,c in [('baseline',colors[0]),('combined',colors[3])]:
        eq=read(SOURCE/'cases'/f'combined_oos_{sc}'/'equity.json')
        times=[datetime.fromisoformat(t)+timedelta(minutes=5) for t,v in eq];values=np.array([v for t,v in eq])
        dd=100*(1-values/np.maximum.accumulate(np.r_[1e8,values])[1:])
        ax[0,0].plot(times,values/1e6,label=sc,color=c,lw=1.25)
        ax[0,1].plot(times,dd,label=sc+' · native',color=c,lw=1.2)
    rows=list(csv.DictReader((ROOT/'minute_marks/oos_baseline_marks.csv').open()))
    times=[datetime.fromisoformat(r['at']) for r in rows];values=np.array([float(r['nav']) for r in rows]);dd=100*(1-values/np.maximum.accumulate(np.r_[1e8,values])[1:])
    ax[0,1].plot(times,dd,label='baseline · retained minute (gaps)',color='#67A9BA',alpha=.7,lw=.6,zorder=1)
    for a in ax[0]:
        a.xaxis.set_major_locator(mdates.MonthLocator());a.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        a.grid(axis='y',alpha=.15);a.legend(frameon=False,fontsize=8,loc='upper left')
    ax[0,0].set(title='A  Funded OOS equity · separate KRW100m starts',ylabel='KRW millions')
    ax[0,1].set(title='B  Observed MTM drawdown · finer marks matter',ylabel='Drawdown (%)');ax[0,1].invert_yaxis()
    months=['2026-04','2026-05','2026-06','2026-07'];x=np.arange(4)
    for j,sc in enumerate(SCENARIOS):
        a=read(ROOT/'analysis'/f'oos_{sc}_account.json');vals=[sum(v for day,v in zip(a['calendar'],a['paired_veto_daily_m']) if day.startswith(m)) for m in months]
        ax[1,0].bar(x+(j-1.5)*.18,vals,width=.18,label=sc,color=colors[j])
    ax[1,0].axhline(0,color='#444',lw=.7);ax[1,0].set_xticks(x,['Apr','May','Jun','Jul'])
    ax[1,0].set(title='C  Cap increment over VETO · full funded accounts',ylabel='Monthly net increment (KRW m)');ax[1,0].legend(frameon=False,fontsize=8,ncol=2)
    a=read(ROOT/'support/oos_baseline_historical_deciles.json');d=[r for r in a if r['feature']=='forecast_core']
    means=np.array([r['mean_net_bps'] for r in d]);lo=np.array([r['ci']['5']['ci95'][0] for r in d]);hi=np.array([r['ci']['5']['ci95'][1] for r in d])
    ax[1,1].errorbar([r['decile'] for r in d],means,yerr=[means-lo,hi-means],fmt='o',color=colors[0],capsize=3)
    ax[1,1].axhline(0,color='#444',lw=.7);ax[1,1].set_xticks(range(1,11))
    ax[1,1].set(title='D  Historical q=1 candidate payoff · OOS baseline',xlabel='Core forecast decile (ties preserved)',ylabel='Net planned-notional bps · 95% block interval')
    for a in ax[1]:a.grid(axis='y',alpha=.15)
    fig.suptitle('KALCB round 2 — reconstructed paths and conditional evidence',fontsize=16,fontweight='bold')
    fig.supxlabel('All history is reused development evidence. Minute marks have gaps; standalone labels are not funded oracle profits.',fontsize=10)
    fig.savefig(out/'assessment.png',dpi=180)
    plt.close(fig)
    write(out/'completion.json',dict(complete=True,image=bound(out/'assessment.png')))


if __name__=='__main__':main()

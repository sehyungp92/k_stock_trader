"""Registered C16 display scales; scaling cannot change economic utility."""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_round2_assessment_v1 import read,write,bound,csvwrite,bootstrap,SOURCE,SCENARIOS

ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_round2_assessment_20261017/v1')


def main():
    out=ROOT/'score_scaling';out.mkdir(exist_ok=False)
    write(out/'binding.json',dict(program=bound(__file__),semantics='Frozen baseline-derived display scales only. R is summed daily PnL/initial, D resamples full ordered intraday relative paths, U resamples daily useful issuer-day counts. Not interchangeable economic values.'))
    rows=[]
    for p in ('is','oos'):
        for sc in SCENARIOS:
            a=read(ROOT/'analysis'/f'{p}_{sc}_account.json');ts=read(SOURCE/'cases'/f'combined_{p}_{sc}'/'trades.json')
            groups=defaultdict(list)
            for t in ts:groups[t['entry_fill_time'][:10],t['symbol']].append(t)
            counts=defaultdict(int)
            for (day,s),trades in groups.items():
                counts[day]+=int(any(t['qty']*t['entry_price']>=.01*t['route_metadata']['portfolio_equity'] for t in trades)
                    and sum(t['net_pnl'] for t in trades)>=.0001*trades[0]['route_metadata']['portfolio_equity'])
            assert sum(counts.values())==a['useful_issuer_days']
            u=bootstrap([counts[d] for d in a['calendar']]);dd=read(ROOT/'support'/f'{p}_{sc}_mtm_dispersion.json')
            rows.append(dict(period=p,scenario=sc,return_pp=a['return_pct'],dd_pct=a['dd_pct'],useful_issuer_days=a['useful_issuer_days'],
                return_se_pp=a['baseline_uncertainty']['5']['se'],dd_se_pp=dd['5']['dd_se_pp'],useful_se_count=u['5']['se'],
                return_display_scale=max(1,a['baseline_uncertainty']['5']['se']),dd_display_scale=max(1,dd['5']['dd_se_pp']),
                useful_display_scale=max(1,u['5']['se'])))
            write(out/f'{p}_{sc}_useful.json',u)
    R=float(np.mean([r['return_pp'] for r in rows]));D=float(np.mean([max(0,r['dd_pct']-(8 if r['scenario']=='baseline' else 10)) for r in rows]))
    csvwrite(out/'cells.csv',rows)
    write(out/'baseline_score.json',dict(R=R,D=D,S=R-.25*D,U_increment=0,ranking_only=True,new_promotion=False,
        risk_metric='Accepted native timestamped MTM; finer-source diagnostics remain separate qualification evidence.',unchanged_economic_coefficients=True))


if __name__=='__main__':main()

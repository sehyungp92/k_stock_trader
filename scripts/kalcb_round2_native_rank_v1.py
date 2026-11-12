"""C16 current native score alias and causal ATR discrimination, no new trading."""
from pathlib import Path
import sys
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_round2_assessment_v1 import read, write, bound, csvwrite, SOURCE, SCENARIOS
from scripts.kalcb_round2_candidate_evidence_v2 import stats
from scripts.kalcb_round2_support_v1 import deciles

ROOT = Path('E:/CodexResearch/k_stock_trader/kalcb_round2_assessment_20261017/v1')
OUT = ROOT / 'native_rank_v1'
FEATURES = ('native_selection_score', 'atr_fraction_decision_ref')


def limits(calendar):
    return {str(length):dict(calendar_sessions=len(calendar),block_equivalents=len(calendar)/length,
        circular_rotation_degenerate=length>=len(calendar),fewer_than_two_nominal_blocks=len(calendar)/length<2,
        confirmatory_ci_credible=False) for length in (5,10,20)}


def main():
    OUT.mkdir(exist_ok=False)
    write(OUT / 'binding.json',dict(program=bound(__file__),
        protocol=bound(Path(__file__).resolve().parents[1] / 'docs/optimisation/kalcb/experiments/kalcb-exp-0041-rank-binding.md'),
        sources=[bound(SOURCE / 'cases' / f'combined_{p}_{sc}' / name) for p in ('is','oos') for sc in SCENARIOS for name in ('trades.json','daily.json')]))
    summary=[]
    for period in ('is','oos'):
        for scenario in SCENARIOS:
            cell=f'{period}_{scenario}';folder=SOURCE / 'cases' / ('combined_'+cell)
            trades=read(folder / 'trades.json');calendar=[r['day'] for r in read(folder / 'daily.json')];rows=[]
            for index,t in enumerate(trades,1):
                m=t['route_metadata'];score=m.get('frontier_selection_score');atr=m.get('daily_atr');ref=m.get('entry_price_ref')
                score=float(score) if score is not None and math.isfinite(float(score)) else None
                ratio=atr/ref if atr is not None and atr>0 and ref is not None and ref>0 else None
                day=t['entry_fill_time'][:10]
                rows.append(dict(day=day,id=day+'/'+t['symbol'],symbol=t['symbol'],trade_row=index,regime=m.get('regime_tier','missing'),
                    native_selection_score=score,atr_fraction_decision_ref=ratio,value=1e4*t['net_pnl']/(t['qty']*t['entry_price'])))
            full=[stats(rows,f,calendar) for f in FEATURES]
            regimes={r:[stats([x for x in rows if x['regime']==r],f,calendar) for f in FEATURES] for r in sorted({x['regime'] for x in rows})}
            months={}
            for month in sorted({d[:7] for d in calendar}):
                cal=[d for d in calendar if d.startswith(month)]
                months[month]=dict(statistics=[stats([r for r in rows if r['day'].startswith(month)],f,cal) for f in FEATURES],inference_limits=limits(cal))
            write(OUT / (cell+'.json'),dict(period=period,scenario=scenario,conditioning='Current realised entries only; not full candidate or random-entry inference',
                full_period=full,deciles=[r for f in FEATURES for r in deciles(rows,f,calendar)],regimes=regimes,months=months,inference_limits=limits(calendar)))
            csvwrite(OUT / (cell+'_rows.csv'),rows)
            for result in full:
                summary.append(dict(period=period,scenario=scenario,feature=result['feature'],rows=result['rows'],missing=len(rows)-result['rows'],
                    auc=result.get('auc'),spearman=result.get('spearman'),auc5_low=result['ci']['5']['auc95'][0],auc5_high=result['ci']['5']['auc95'][1]))
            print(cell,summary[-2:],flush=True)
    csvwrite(OUT / 'summary.csv',summary);write(OUT / 'completion.json',dict(complete=True,cells=8,new_native_accounts=0))


if __name__=='__main__':main()

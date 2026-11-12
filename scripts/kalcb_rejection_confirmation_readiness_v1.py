"""C14: registered negative population, one causal fresh-evidence comparison."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from scripts.kalcb_opportunity_closeout_v1 import bound,read,write,verify,lines

REPO=Path(__file__).resolve().parents[1]
PANEL=Path('E:/CodexResearch/k_stock_trader/kalcb_structural_preparation_20261016/entry_lifecycle_v2')
MAP=Path('E:/CodexResearch/k_stock_trader/kalcb_structural_preparation_20261016/funded_horizon_v1/horizon_map.json')
DESIGN=REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0038.md'


def stats(rows):
    valid=[r for r in rows if r['label_valid'] and not r['unresolved'] and r['prefix_supported']]
    monthly=defaultdict(lambda:dict(count=0,net=0.,earned=0.,remaining=0.))
    for r in valid:
        v=monthly[r['day'][:7]];v['count']+=1
        for key,field in [('net','net_reconstructed_krw'),('earned','marked_net_at35_krw'),('remaining','remaining_net_after35_krw')]:v[key]+=r[field]
    return dict(registered=len(rows),valid_prefix=len(valid),distinct_dates=len({r['day'] for r in valid}),
        net=sum(r['net_reconstructed_krw'] for r in valid),earned=sum(r['marked_net_at35_krw'] for r in valid),
        remaining=sum(r['remaining_net_after35_krw'] for r in valid),
        positive_remaining=sum(max(0,r['remaining_net_after35_krw']) for r in valid),
        negative_remaining=sum(min(0,r['remaining_net_after35_krw']) for r in valid),
        profitable_lifetimes=sum(r['net_reconstructed_krw']>0 for r in valid),
        positive_remaining_lifetimes=sum(r['remaining_net_after35_krw']>0 for r in valid),monthly=dict(monthly))


def main(out):
    out.mkdir(exist_ok=False)
    done=read(PANEL/'completion.json');verify(done['panel']);verify(done['registration'])
    protocol=read(MAP.parent/'protocol.json');verify(protocol['signal_map'])
    inputs=[bound(DESIGN),bound(Path(__file__)),bound(REPO/'scripts/kalcb_opportunity_closeout_v1.py'),
        bound(MAP),bound(MAP.parent/'protocol.json'),bound(PANEL/'completion.json'),done['panel'],done['registration']]
    signals={k:v for k,v in read(MAP).items() if v['predictions']['baseline'] < -23 and v['predictions']['combined'] < -33}
    assert len(signals)==36
    write(out/'registration.json',dict(inputs=inputs,signals=signals,quantity_selection='original map scale',predicate='completed09:30 close strictly above six-bar opening high',native_replays=0,fits=0))
    rows=[];alternatives=[]
    for r in lines(PANEL/'panel.jsonl.gz'):
        if r['base_id'] not in signals:continue
        assert r['available_at30']==signals[r['base_id']]['available_at']
        z={k:v for k,v in r.items() if k not in ('features30','features35')}
        z['registered_scale']=signals[r['base_id']]['quantity_scale']
        z['confirmation']=r.get('states',{}).get('above_opening_high') if r['prefix_supported'] else None
        if float(r['quantity_scale'])==float(z['registered_scale']):rows.append(z)
        else:alternatives.append(z)
    assert len(rows)==72
    cells={}
    for period in ('IS','OOS'):
        for scenario in ('baseline','combined'):
            rr=[r for r in rows if r['period'].upper()==period and r['scenario']==scenario]
            cells[period+'/'+scenario]={str(flag):stats([r for r in rr if r['confirmation'] is flag]) for flag in (True,False,None)}
    for b in inputs:verify(b)
    write(out/'assessment.json',dict(cells=cells,rows=rows,other_registered_quantities=alternatives,
        interpretation='Native original holding continuation, not flat re-entry payoff or funded policy return. No new fit/quantity choice.',economic_objective_achieved=False))
    write(out/'completion.json',dict(complete=True,assessment=bound(out/'assessment.json'),registration=bound(out/'registration.json'),native_replays=0,fits=0,at=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)

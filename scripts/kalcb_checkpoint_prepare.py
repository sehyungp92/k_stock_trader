"""Prepare an explicit later entry decision from completed observations only."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from itertools import groupby
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev, continuation_value as cv
from scripts.kalcb_executed_value_study import load_plan, config_for, connect, load_day, emit
from scripts.kalcb_action_value_study import action_features


def checkpoint_features(bars, cutoff):
    prefix=[b for b in bars if b.timestamp+timedelta(minutes=5)<=cutoff]
    if len(prefix)<9 or any(b.timestamp-a.timestamp!=timedelta(minutes=5) for a,b in zip(prefix,prefix[1:])):
        raise ValueError('Incomplete checkpoint prefix')
    first=prefix[:6];late=prefix[6:];last=prefix[-1]
    changes=[b.close-b.open for b in late]
    vwap=sum((b.high+b.low+b.close)/3*b.volume for b in late)/max(1,sum(b.volume for b in late))
    return {'checkpoint_return':last.close/first[-1].close-1,
        'checkpoint_above_or':last.close/max(b.high for b in first)-1,
        'checkpoint_vwap_distance':last.close/max(vwap,1e-9)-1,
        'checkpoint_volume_ratio':sum(b.volume for b in late)/max(1,sum(b.volume for b in first))*6/len(late),
        'checkpoint_up_share':sum(x>=0 for x in changes)/len(late),
        'checkpoint_efficiency':sum(changes)/max(1e-9,sum(abs(x) for x in changes))}


def prepare(plan,parent,output):
    from strategy_kalcb.core.preview import preview_entry
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    cfg=config_for(plan)
    if not cfg.entry_checkpoint_enabled:raise ValueError('Checkpoint entry disabled')
    meta=json.loads((parent/'prepared.json').read_text())
    if ev.digest(parent/'opportunities.jsonl')!=meta['opportunities_sha256']:raise ValueError('Changed parent')
    conn=connect(plan);source=plan['input_sha256'][plan['database']];counts=Counter()
    import yaml
    universe=tuple(str(s).zfill(6) for s in yaml.safe_load(Path(plan['universe']).read_text())['symbols'])
    portfolio=KALCBPortfolioView(cash=ev.REFERENCE_EQUITY,equity=ev.REFERENCE_EQUITY)
    prior={};previous=None
    with (parent/'opportunities.jsonl').open() as inp,(output/'opportunities.jsonl').open('x') as out:
        for i,(day,group) in enumerate(groupby(map(json.loads,inp),lambda r:r['trade_date'])):
            rows=list(group);cutoff=datetime.fromisoformat(rows[0]['feature_available_at'])+timedelta(minutes=15)
            bars=defaultdict(list)
            for b in load_day(conn,day,source,set(universe)):
                if b.timestamp+timedelta(minutes=5)<=cutoff:bars[b.symbol].append(b)
            prior=defaultdict(list)
            if previous:
                for s,t,o,h,l,c,v in conn.execute('SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day=? ORDER BY symbol,ts',(previous,)):
                    prior[s].append(dict(timestamp=t,open=o,high=h,low=l,close=c,volume=v))
            previous=day
            sectors=defaultdict(list)
            for r in rows:
                p=bars[r['symbol']]
                if p:sectors[r['sector']].append(p[-1].close/p[0].open-1)
            for r in rows:
                counts['considered']+=1;p=bars[r['symbol']]
                try:extra=checkpoint_features(p,cutoff)
                except ValueError:
                    counts['incomplete_prefix']+=1;continue
                r['feature_available_at']=cutoff.isoformat()
                r['candidate']['metadata'].update(entry_checkpoint_at=cutoff.isoformat(),feature_available_at=cutoff.isoformat())
                snap=ev.snapshot_for(day,[(r,0.)],source=source)
                preview=preview_entry(p,snap,cfg,portfolio,available_at=cutoff)
                counts['ready']+=preview['ready']
                # All rows retained in preparation; label membership is the causal native preview.
                r['original_guard_at_rank1']=r['guard_at_rank1'];r['guard_at_rank1']=True
                r['action']={'ready':preview['ready'],'quantity':sum(a['quantity'] for a in preview['intents']),
                    'routes':[a['metadata'].get('entry_route') for a in preview['intents']],
                    'reasons':[d['reason'] for d in preview['decisions']]}
                r['opening_bars']=[dict(symbol=b.symbol,timestamp=b.timestamp.isoformat(),**{k:getattr(b,k) for k in ('open','high','low','close','volume')}) for b in p]
                sector=sectors[r['sector']]
                r['features'].update(cv.opening_path(r['opening_bars'][:6],(p[5].timestamp+timedelta(minutes=5)).isoformat()),
                    **cv.prior_path(prior.get(r['symbol'],[]),day),**action_features(preview,p),**extra,
                    checkpoint_sector_return=sum(sector)/max(1,len(sector)),checkpoint_sector_breadth=sum(v>0 for v in sector)/max(1,len(sector)))
                # No outcome proxy is retained in the new modeling dataset.
                r.pop('quality_target',None);emit(out,r)
            if i%20==0:print(json.dumps({'session':i+1,**counts}),flush=True)
    conn.close()
    ev.write_json(output/'prepared.json',{**meta,'version':'kalcb-checkpoint45-v1','parent_prepared_sha256':ev.digest(parent/'prepared.json'),
        'opportunities_sha256':ev.digest(output/'opportunities.jsonl'),'counts':counts,'observation_symbols':universe,
        'contract':'First thirty minute context plus completed forty-five minute prefix. No later bars or outcome eligibility.'})


def main():
    from backtests.auto.shared.cache_keys import source_generation_hash
    p=argparse.ArgumentParser()
    for k in ('plan','parent','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();plan=load_plan(a.plan);source=source_generation_hash();a.output.mkdir(parents=True,exist_ok=False)
    ev.write_json(a.output/'request.json',{'arguments':vars(a),'source_generation_hash':source})
    prepare(plan,a.parent,a.output)
    if source_generation_hash()!=source:raise RuntimeError('Source changed')
    ev.write_json(a.output/'completion.json',{'complete':True,'source_generation_hash':source})


if __name__=='__main__':main()

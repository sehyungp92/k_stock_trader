"""Enrich authenticated opening candidates with prior-confirmed campaign state."""
import argparse,json,sys
from datetime import date
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev,campaign_value as campaign
from scripts.kalcb_executed_value_study import load_plan,emit

def prepare(plan,parent,output):
    from backtests.core.input_manifest import InputManifest
    from backtests.strategies.kalcb.first30_signal_sweep import _load_daily_rows
    meta=json.loads((parent/'prepared.json').read_text())
    if ev.digest(parent/'opportunities.jsonl')!=meta['opportunities_sha256']:raise ValueError('Changed parent features')
    if meta['sessions'][-1]>'2026-03-31':raise ValueError('Development boundary exceeded')
    generation=InputManifest(plan['market_manifest'])
    with generation.activate():
        daily=_load_daily_rows(generation.root/'data/krx_daily_parquet',meta['context_symbols'],date.fromisoformat(meta['sessions'][-1]))
    counts={'rows':0,'campaign_available':0}
    with (parent/'opportunities.jsonl').open() as stream,(output/'opportunities.jsonl').open('x',encoding='utf-8') as out:
        for line in stream:
            row=json.loads(line)
            block,proof=campaign.features(row,daily[row['symbol']])
            row['features'].update(block);row['campaign_provenance']=proof
            # Do not change execution candidate metadata in this feature-only test.
            emit(out,row);counts['rows']+=1;counts['campaign_available']+=proof['available']
            if counts['rows']%1000==0:print(json.dumps(counts),flush=True)
    ev.write_json(output/'prepared.json',{**meta,'version':'kalcb-campaign-v1',
        'continuation_parent_sha256':ev.digest(parent/'prepared.json'),
        'opportunities_sha256':ev.digest(output/'opportunities.jsonl'),
        'campaign_columns':campaign.COLUMNS,'campaign_counts':counts,
        'campaign_contract':'Prior-day box and box-start AVWAP; completed opening bars only; execution unchanged'})
    print(json.dumps(counts),flush=True)

def main():
    from backtests.auto.shared.cache_keys import source_generation_hash
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','prepared','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();plan=load_plan(a.plan);source=source_generation_hash();a.output.mkdir(parents=True,exist_ok=False)
    ev.write_json(a.output/'request.json',{'arguments':vars(a),'source_generation_hash':source})
    prepare(plan,a.prepared,a.output)
    if source_generation_hash()!=source:raise ValueError('Source changed')
    ev.write_json(a.output/'completion.json',{'complete':True,'source_generation_hash':source})

if __name__=='__main__':main()

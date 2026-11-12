"""Add causal relative leadership without changing the native opportunity set."""
import argparse,json,sys
from itertools import groupby
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev,leadership_value as leader
from scripts.kalcb_executed_value_study import load_plan,emit

def prepare(parent,universe,output):
    meta=json.loads((parent/'prepared.json').read_text());full=json.loads((universe/'prepared.json').read_text())
    if ev.digest(universe/'prepared.json')!=meta['original_prepared_sha256']:raise ValueError('Wrong full-universe parent')
    for directory,metadata in ((parent,meta),(universe,full)):
        if ev.digest(directory/'opportunities.jsonl')!=metadata['opportunities_sha256']:raise ValueError('Changed opportunities')
    if meta['sessions'][-1]>'2026-03-31':raise ValueError('Development boundary exceeded')
    ranks={};cohorts={}
    with (universe/'opportunities.jsonl').open() as stream:
        for day,items in groupby((json.loads(line) for line in stream),key=lambda r:r['trade_date']):
            items=list(items);cutoffs={r['feature_available_at'] for r in items}
            if len(cutoffs)!=1:raise ValueError('Mixed opening decision clocks')
            ranks[day]=leader.rank_at_cutoff(items,day,next(iter(cutoffs)));cohorts[day]=len(ranks[day])
    count=0
    with (parent/'opportunities.jsonl').open() as stream,(output/'opportunities.jsonl').open('x',encoding='utf-8') as out:
        for line in stream:
            r=json.loads(line);r['features'].update(ranks[r['trade_date']][r['symbol']]);emit(out,r);count+=1
    ev.write_json(output/'prepared.json',{**meta,'version':'kalcb-leadership-v1','leadership_columns':leader.COLUMNS,
        'continuation_parent_sha256':ev.digest(parent/'prepared.json'),
        'opportunities_sha256':ev.digest(output/'opportunities.jsonl'),'leader_cohort_sizes':cohorts})
    print(json.dumps({'rows':count,'sessions':len(cohorts),'min_cohort':min(cohorts.values())}),flush=True)

def main():
    from backtests.auto.shared.cache_keys import source_generation_hash
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','prepared','universe','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();load_plan(a.plan);source=source_generation_hash();a.output.mkdir(parents=True,exist_ok=False)
    ev.write_json(a.output/'request.json',{'arguments':vars(a),'source_generation_hash':source})
    prepare(a.prepared,a.universe,a.output)
    if source_generation_hash()!=source:raise ValueError('Source changed')
    ev.write_json(a.output/'completion.json',{'complete':True,'source_generation_hash':source})

if __name__=='__main__':main()

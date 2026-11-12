"""Finite auxiliary physical-risk evidence for the same native minute90 order.

Entry/payoff population remains reclaim-only. These newly replayed outcomes are
not return-training targets, portfolio returns, or automatic risk-gate approval.
"""
from collections import Counter,defaultdict
from itertools import groupby
from pathlib import Path
import gzip,json
from backtests.strategies.kalcb import executed_value as ev
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,verify,bound,digest
from backtests.strategies.kalcb.reclaim_labels_v1 import audited_episode,continuous_connection


def auxiliary(row):
    return row['quarter_preview']['ready'] is True and row['guard_at_rank1'] is False


def labels(root,scenario,smoke=False):
    from scripts.kalcb_executed_value_study import config_for,load_day,emit
    from scripts.kalcb_action_quantity_labels import scaled_row
    p=read(root/'protocol.json');verify(p['programs']+p['bindings']+p['data_bindings'])
    old=Path(p['source_label_root']);meta=read(old/'prepared/prepared.json')
    if digest(old/'prepared/opportunities.jsonl')!=meta['opportunities_sha256']:raise ValueError('Changed source population')
    verify([meta['source_audit']]);assert read(meta['source_audit']['path'])['passed']
    prepared=ev.read_jsonl(old/'prepared/opportunities.jsonl')
    expected={r['opportunity_id']+'/q0.25' for r in prepared if auxiliary(r)}
    if len(expected)!=p['auxiliary_actions']:raise ValueError('Changed finite auxiliary population')
    target=root/('smoke' if smoke else 'labels')/scenario;target.mkdir(parents=True,exist_ok=False)
    plan=read(p['producer_runtime']['plan']['path']);cfg=config_for(plan,scenario)
    conn=continuous_connection(p['databases']);source=ev.signature(p['data_bindings']);counts=Counter();ids=set()
    for day,group in groupby(prepared,lambda r:r['trade_date']):
        rows=[r for r in group if auxiliary(r)];bars=defaultdict(list)
        for b in load_day(conn,day,source,{r['symbol'] for r in rows}):bars[b.symbol].append(b)
        with (target/(day+'.jsonl')).open('x',encoding='utf-8') as out,gzip.open(target/(day+'.events.jsonl.gz'),'xt',encoding='utf-8',compresslevel=1) as events:
            for r in rows:
                action=scaled_row(r,.25)
                value,detail=audited_episode(action,bars[r['symbol']],cfg,scenario,source,conn,meta['sessions'],p)
                if action['opportunity_id'] in ids:raise ValueError('Duplicate auxiliary action')
                ids.add(action['opportunity_id'])
                value.update(auxiliary_physical_only=True,reclaim_eligible=False,return_model_eligible=False,
                    label_role='Fresh same-time/quantity/management order outcome for physical-risk population investigation only')
                detail.update(auxiliary_physical_only=True,reclaim_eligible=False,return_model_eligible=False)
                emit(out,value);emit(events,detail);counts[value['status']]+=1
                if not value['independent_episode_execution_pass']:counts['independent_execution_failures']+=1
                if smoke and value['status']=='filled' and value['independent_episode_execution_pass']:break
        print(json.dumps({'stage':'auxiliary_smoke' if smoke else 'auxiliary_labels','scenario':scenario,'day':day,'actions':len(ids),**counts}),flush=True)
        if smoke and counts['filled']:break
    conn.close()
    if smoke and (not counts['filled'] or counts['independent_execution_failures']):raise ValueError('Auxiliary filled smoke missing')
    if not smoke and ids!=expected:raise ValueError('Incomplete auxiliary population')
    verify(p['programs']+p['bindings']+p['data_bindings'])
    write(target/'completion.json',{'complete':True,'smoke':smoke,'scenario':scenario,'actions':len(ids),'counts':dict(counts),
        'source_preparation':bound(old/'prepared/prepared.json'),'config':bound(plan['config']),
        'files':{f.name:digest(f) for f in target.iterdir() if f.is_file()},'auxiliary_physical_only':True,
        'return_model_eligible':False,'geometry_labels_replayed':False,'new_models':0,'portfolio_cases':0,
        'scope':'Investigation of physical-risk population comparability. Counts and zero failures do not certify risk below.001. Reclaim selection and all guards remain fixed.'})

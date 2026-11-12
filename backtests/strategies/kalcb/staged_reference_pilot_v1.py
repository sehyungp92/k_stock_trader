"""Eight fixed reference contrasts, two disabled bridges, eight reused controls."""
import gzip
import json
from pathlib import Path
from unittest.mock import patch
from backtests.strategies.kalcb.staged_commitment_pilot_v1 import episode
from backtests.strategies.kalcb.checkpoint_contrast_labels_v1 import SourceIndexes
from backtests.strategies.kalcb.late_entry_v1 import continuous_connection
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify


def run(root):
    from scripts.kalcb_executed_value_study import config_for,load_day
    from strategy_kalcb.models import KALCBDailySnapshot
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter
    p=read(root/'protocol.json');specs=read(p['episodes']['path']);out=root/'labels';out.mkdir(exist_ok=False)
    prior=Path(p['parent_pilot']);prior_done=read(prior/'labels/completion.json')
    verify([x for row in prior_done['outputs'] for x in row.values()])
    conn=continuous_connection(p['databases']);sessions=[x[0] for x in conn.execute('SELECT DISTINCT day FROM bars ORDER BY day')]
    if len(sessions)!=286:raise ValueError('Changed source history')
    indexes=SourceIndexes(p['databases']);outputs=[];summary=[];fresh=0
    for row in prior_done['outputs']:
        if Path(row['label']['path']).parent.name=='control':outputs.append(row)
    def inputs(spec):
        cfg=config_for(read(spec['profile']['plan']['path']),spec['scenario'])
        if cfg.exit_hard_stop_enabled or cfg.risk_max_gross_equity_pct:raise ValueError('Unsupported reference contrast configuration')
        snap=KALCBDailySnapshot.from_json_dict(next(x for x in read(spec['snapshot']['path']) if x['trade_date']==spec['day']))
        bars=load_day(conn,spec['day'],'staged-commitment-existing-raw',{spec['symbol']})
        return cfg,snap,bars
    def save(spec,arm,value,detail):
        folder=out/spec['id']/arm;folder.mkdir(parents=True)
        write(folder/'label.json',value)
        with gzip.open(folder/'events.json.gz','wt',encoding='utf-8',compresslevel=1) as f:json.dump(detail,f,default=str,allow_nan=False)
        outputs.append({'label':bound(folder/'label.json'),'events':bound(folder/'events.json.gz')})
    # Both disabled bridges gate any reference-enabled historical replay.
    for spec in [s for s in specs if s['bridge']]:
        cfg,snap,bars=inputs(spec)
        value,detail=episode(spec,snap,bars,cfg,conn,sessions,indexes,'staged')
        path=prior/'labels'/spec['id']/'staged/events.json.gz'
        with gzip.open(path,'rt',encoding='utf-8') as f:old=json.load(f)
        for k in ('fills','orders','journal','equity_curve','timestamps','terminal_cash'):
            if json.dumps(detail[k],default=str,sort_keys=True)!=json.dumps(old[k],default=str,sort_keys=True):
                raise ValueError('Disabled reference changes original staged path')
        save(spec,'disabled_bridge',value,detail);fresh+=1
    old_init=KALCBReplayAdapter.__init__
    def init(self,*a,**kw):
        old_init(self,*a,**kw);self.state.meta['staged_management_reference_enabled']=True
    for i,spec in enumerate(specs,1):
        cfg,snap,bars=inputs(spec)
        with patch.object(KALCBReplayAdapter,'__init__',init):
            value,detail=episode(spec,snap,bars,cfg,conn,sessions,indexes,'staged')
        value['management_reference_enabled']=True;detail['management_reference_enabled']=True
        if not all(value[k] for k in ('label_valid','independent_account_pass','independent_execution_pass')) or value['deadline_violation']:
            raise ValueError('Reference pilot independent native check failed; preserve failure')
        save(spec,'reference',value,detail);fresh+=1
        parent=read(prior/'labels'/spec['id']/'control/label.json');original=read(prior/'labels'/spec['id']/'staged/label.json')
        summary.append({'id':spec['id'],'control_net_krw':parent['net_value_krw'],
            'staged_rebased_net_krw':original['net_value_krw'],'staged_reference_net_krw':value['net_value_krw'],
            'completion_quantity':value['completion_filled_quantity']})
        write(root/'progress.json',{'completed_reference_pairs':i,'expected':8,'fresh_episodes':fresh,'latest':summary[-1]})
        print(json.dumps(summary[-1]),flush=True)
    conn.close();verify(p['bindings'])
    write(out/'completion.json',{'complete':True,'study_complete':False,'episodes':18,'fresh_episodes':fresh,
        'reused_controls':8,'pairs':8,'outputs':outputs,'summary':summary,
        'scope':'Known pilot population, conditional management-reference investigation, not independent or portfolio improvement.'})

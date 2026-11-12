"""Selection-only successor: recheck real combined candidate rank/context before reserving the one later slot."""
from backtests.strategies.kalcb.late_entry_v1 import *


def select_executable(choices,latest,cfg,flat_only,preview_fn=None):
    from strategy_kalcb.core.preview import preview_entry
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    from strategy_common.market import MarketBar
    preview_fn=preview_fn or preview_entry
    for row,score in sorted(choices,key=lambda x:(-min(x[1]['baseline'],x[1]['combined']),-x[1]['baseline'],x[0]['symbol'])):
        value=min(score['baseline'],score['combined']);snap=append_snapshot(latest,row,value,flat_only)
        prefix=[MarketBar(row['symbol'],datetime.fromisoformat(b['timestamp']),'5m',*[b[k] for k in ('open','high','low','close','volume')]) for b in row['opening_bars']]
        preview=preview_fn(prefix,snap,cfg,KALCBPortfolioView(cash=ev.REFERENCE_EQUITY,equity=ev.REFERENCE_EQUITY),available_at=snap.generated_at)
        if any(a['symbol']==row['symbol'] for a in preview['intents']):return row,score,value,snap
    return None


def build_policies(root):
    from scripts.kalcb_action_quantity_labels import scaled_row
    from scripts.kalcb_action_quantity_ranker import capacity_ok
    from scripts.kalcb_executed_value_replay import verify_policy
    from scripts.kalcb_executed_value_study import config_for
    from strategy_kalcb.core.preview import preview_entry
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    from strategy_common.market import MarketBar
    protocol=read(root/'protocol.json');meta=read(root/'prepared/prepared.json')
    models=read(root/'models/completion.json');verify([{'path':str(root/'models'/f),'sha256':h} for f,h in models['files'].items()])
    scores={r['opportunity_id']:r for r in ev.read_jsonl(root/'models/forecasts.jsonl')};daily=defaultdict(list)
    for raw in ev.read_jsonl(root/'prepared/opportunities.jsonl'):
        if raw['guard_at_rank1'] and raw['action']['ready']:
            for q in (.25,.5):
                r=scaled_row(raw,q);s=scores.get(r['opportunity_id'])
                if s and s['failure_probability']<.001 and min(s['baseline'],s['combined'])>48*r['features']['action_notional_equity'] and capacity_ok(r):daily[r['trade_date']].append((r,s))
    for name,recipe in protocol['recipes'].items():
        for period in ('is','oos'):
            old=protocol['profiles'][str(recipe['prior'])][period];source=Path(old['policy']['path']).parent
            m,opening=verify_policy(source,'interaction_cost_margin');updates=cumulative_timed_policy(source,m,opening)
            target=root/'policies'/name/period;target.mkdir(parents=True,exist_ok=False)
            for f in m['files']:link(source/f,target/f)
            timed=[];admissions=ev.read_jsonl(source/m['timed_admission_file']);added=[]
            for day,base in sorted(opening.items()):
                parents=updates.get(day,[]);timed.extend(s.to_json_dict() for s in parents)
                latest=parents[-1] if parents else base
                choices=[(r,s) for r,s in daily[str(day)] if r['symbol'] not in latest.by_symbol() and r['features']['action_quantity_scale']==recipe['quantity']]
                if recipe['quantity']==0 or not choices:continue
                selected=select_executable(choices,latest,config_for(read(old['plan']['path'])),recipe['flat_only'])
                if selected is None:continue
                row,score,value,snap=selected;timed.append(snap.to_json_dict())
                added.append({'trade_date':str(day),'symbol':row['symbol'],'available_at':row['feature_available_at'],'score':value,
                    'action_id':row['opportunity_id'],'quantity_scale':recipe['quantity'],'valid_from':score['valid_from'],'flat_only':recipe['flat_only']})
            # New names avoid mutating hard-linked parent policy files.
            for f,values in [('later60.snapshots.jsonl',timed),('later60.admission.jsonl',admissions+added),('later60.new.jsonl',added)]:
                with (target/f).open('x',encoding='utf-8') as stream:
                    for r in values:stream.write(json.dumps(r,default=str,allow_nan=False)+'\n')
            chronology=read(source/'chronology.json');chronology['folds']+=read(root/'models/chronology.json')['folds']
            # Chronology is a hardlink: replace with a new inode, never overwrite.
            (target/'chronology.json').unlink();write(target/'chronology.json',chronology)
            for f in models['files']:link(root/'models'/f,target/(f if f.endswith('.joblib') else 'later_model_'+f))
            write(target/'later_lineage.json',{'registration':bound(root/'protocol.json'),'preparation':bound(root/'prepared/prepared.json'),
                'models':bound(root/'models/completion.json'),'label_manifests':[bound(root/'labels'/s/'completion.json') for s in ('baseline','combined')],
                'parent_policy':old['policy'],'parent_snapshots_preserved':True,'recipe':recipe})
            m.update(version='kalcb-cumulative-first-hour-v1',observation_minutes=60,timed_snapshot_file='later60.snapshots.jsonl',
                timed_admission_file='later60.admission.jsonl',contract={**m['contract'],'timed_budget':13,'new_budget':1},
                files={p.name:digest(p) for p in target.iterdir() if p.is_file()})
            write(target/'policies_manifest.json',m);write(target/'completion.json',{'complete':True})
            actual,ss=verify_policy(target,'interaction_cost_margin');cumulative_timed_policy(target,actual,ss)
            print(json.dumps({'stage':'policy60','recipe':name,'period':period,'added_decisions':len(added)}),flush=True)
    write(root/'policies/completion.json',{'complete':True,'independent_validation':False})

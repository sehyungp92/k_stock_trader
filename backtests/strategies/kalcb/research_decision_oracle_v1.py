"""Default-off, oracle-only policy projection over the unchanged native replay.

Future values live in a separate choice book, never in causal candidate scores
or a training feature table. This is a fixed feasible heuristic, not an optimum.
"""
from collections import defaultdict
from dataclasses import replace
from datetime import date,datetime
import gzip,json,hashlib
from pathlib import Path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(p):return dict(path=str(p),sha256=sha(p))
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)

class OracleProjection:
    def __init__(self,arm,period,scenario,registration,output):
        if arm not in ('off','S','E','SE'):raise ValueError('Unregistered oracle arm')
        self.arm=arm;self.period=period;self.scenario=scenario;self.reg=registration;self.output=Path(output)
        self.installed=False;self.membership={};self.entry_records={};self.choices=[];self.native_snapshots={};self.native_updates={}
        self.labels=defaultdict(dict);self.cohorts={};self.menus={};self.exit_records=[]
        if arm=='off':return
        label=registration['labels'][period][scenario]
        assert sha(label['path'])==label['sha256']
        for r in rows(label['path']):
            if r['label_valid']:self.labels[r['episode_id']][r['horizon_minutes']]=r
        cohort=registration['cohorts'];assert sha(cohort['path'])==cohort['sha256']
        for r in rows(cohort['path']):
            if r['period']==period and r['episode_id'] in self.labels:self.cohorts[r['episode_id']]=r
        for key,menu in self.labels.items():
            # Cash objective; ties preserve implemented exit, then shorter hold.
            self.menus[key]=min(menu.values(),key=lambda r:(-r['net_cash_krw'],r['horizon_minutes']))

    def choose(self,day,cohort,limit,existing=()):
        candidates=[]
        for key,row in self.cohorts.items():
            if row['trade_date']!=day or row['cohort']!=cohort or row['symbol'] in existing:continue
            label=self.labels[key].get(0) if self.arm=='S' else self.menus[key]
            if label is None or label['net_cash_krw']<=0:continue
            admission=row.get('original_admission')
            core=bool(admission and (cohort=='checkpoint' or admission['record'].get('forecast_origin')=='frozen_core'))
            candidates.append((key,row,label,core))
        selected=[];support=0
        for key,row,label,core in sorted(candidates,key=lambda x:(-x[2]['net_cash_krw'],x[0])):
            if len(selected)>=limit:break
            if not core and support>=1:continue
            support+=not core;selected.append((key,row,label))
        self.choices.append(dict(day=day,cohort=cohort,capacity=limit,positive_menu_candidates=len(candidates),
            selected=[dict(episode_id=k,label_horizon=l['horizon_minutes'],label_cash=l['net_cash_krw']) for k,r,l in selected],
            new_support_slots=support,oracle_only=True))
        return selected

    def snapshot(self,day,selected,incumbents=()):
        from backtests.strategies.kalcb import executed_value as ev
        pairs=[]
        for key,row,label in selected:
            candidate={**row['candidate'],'metadata':{**row['candidate']['metadata'],'action_quantity_scale':label['quantity_scale']}}
            pairs.append((dict(candidate=candidate,feature_available_at=row['feature_available_at']),label['raw_score']))
            self.membership[day,row['symbol']]=key
        snap=ev.snapshot_for(day,pairs,source=self.reg['cohorts']['sha256'])
        if incumbents:
            additions=tuple(replace(c,metadata={**c.metadata,'candidate_rank':i+len(incumbents)+1,
                'frontier_rank':i+len(incumbents)+1}) for i,c in enumerate(snap.candidates))
            combined=tuple(incumbents)+additions
            snap=replace(snap,candidates=combined,metadata={**snap.metadata,'active_symbols':[c.symbol for c in combined],
                'active_symbol_count':len(combined),'candidate_pool_count':len(combined),'selection_count':len(combined),'frontier_symbol_count':len(combined)})
        return snap

    def register_original_membership(self,snapshots):
        for d,snap in snapshots.items():
            for c in snap.candidates:
                self.membership.setdefault((str(d),c.symbol),self.period+'/'+str(d)+'/'+c.symbol+'/opening')

    def save_policy(self):
        payload=dict(arm=self.arm,period=self.period,scenario=self.scenario,oracle_only=True,eligible_for_scoring=False,
            causal_trading_policy=False,objective='terminal after-cost cash; fixed restricted feasible heuristic',
            source_registration=self.reg['registration_binding'],
            snapshots=[s.to_json_dict() for _,s in sorted(self.native_snapshots.items())],
            updates=[s.to_json_dict() for _,seq in sorted(self.native_updates.items()) for s in seq],
            membership=[dict(day=d,symbol=s,episode_id=k) for (d,s),k in sorted(self.membership.items())],choices=self.choices,
            menus={k:dict(horizon=v['horizon_minutes'],net_cash=v['net_cash_krw'],first_fill=v['first_fill'],quantity=v['buy_quantity']) for k,v in self.menus.items()})
        p=self.output/'oracle_policy.json'
        with p.open('x',encoding='utf-8') as f:json.dump(payload,f,default=str,allow_nan=False)
        self.policy_binding=bound(p)

    def install(self):
        if self.arm=='off':return
        from scripts import kalcb_executed_value_replay as replay
        from backtests.engine import integrity
        self.original_verify=replay.verify_policy;self.original_timed=replay.timed_policy;self.original_audit=integrity.audit_broker
        owner=self
        def verify_policy(directory,name):
            manifest,original=owner.original_verify(directory,name)
            owner.manifest=manifest
            if owner.arm=='E':
                owner.native_snapshots=original;owner.register_original_membership(original);return manifest,original
            result={}
            for d in original:
                selected=owner.choose(str(d),'opening',manifest['contract']['budget'])
                result[d]=owner.snapshot(str(d),selected)
            owner.native_snapshots=result;return manifest,result
        def timed_policy(directory,manifest,snapshots):
            if owner.arm=='E':
                updates=owner.original_timed(directory,manifest,snapshots)
                for d,seq in updates.items():
                    for snap in seq:
                        for c in snap.candidates:
                            owner.membership.setdefault((str(d),c.symbol),owner.period+'/'+str(d)+'/'+c.symbol+'/checkpoint')
            else:
                updates={}
                for d,snap in snapshots.items():
                    selected=owner.choose(str(d),'checkpoint',max(0,manifest['contract']['timed_budget']-len(snap.candidates)),{c.symbol for c in snap.candidates})
                    if selected:updates[d]=[owner.snapshot(str(d),selected,snap.candidates)]
            owner.native_updates=updates;owner.save_policy();return updates
        def audit(broker,*,identity=None):
            identity=dict(identity or {})
            identity['parent_causal_policy_manifest_sha256']=identity.pop('candidate_policy_manifest_sha256',None)
            identity.update(scope='C18 oracle-only conditional execution; not causal policy qualification',
                oracle_policy=owner.policy_binding,oracle_programs=owner.reg['programs'],oracle_registration=owner.reg['registration_binding'],
                future_information_permitted_only_for_declared_hindsight_choices=True,eligible_for_scoring=False)
            # Full existing source, ledger, clock, capacity and population checks
            # still run on the actual newly consumed snapshots and observations.
            return owner.original_audit(broker,identity=identity)
        replay.verify_policy=verify_policy;replay.timed_policy=timed_policy;integrity.audit_broker=audit;self.installed=True

    def horizon(self,ss,position,bar):
        key=self.membership.get((str(position.entry_time.date()),position.symbol))
        label=self.menus.get(key)
        horizon=label['horizon_minutes'] if label else 0
        identity=(position.entry_order_id,position.entry_time.isoformat())
        if identity not in self.entry_records:
            rec=dict(episode_id=key,entry_order_id=position.entry_order_id,actual_entry=position.entry_time.isoformat(),
                actual_qty=position.qty_open,horizon=horizon,missing_menu=label is None,
                label_entry=label['first_fill'] if label else None,label_qty=label['buy_quantity'] if label else None,
                scope='oracle choice; not available to causal features or training')
            self.entry_records[identity]=rec;self.exit_records.append(rec)
        return horizon

    def restore(self):
        if self.installed:
            from scripts import kalcb_executed_value_replay as replay
            from backtests.engine import integrity
            replay.verify_policy=self.original_verify;replay.timed_policy=self.original_timed;integrity.audit_broker=self.original_audit
            self.installed=False

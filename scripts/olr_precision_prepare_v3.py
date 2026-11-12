"""Surgical changes to authenticated immutable-parent generations only."""
import copy,os,shutil,subprocess,sys,difflib
from pathlib import Path
from scripts import olr_precision_study_v3 as t
n,s,b=t.n,t.s,t.b

def once(value,old,new):
    assert value.count(old)==1,(old,value.count(old));return value.replace(old,new)

def patch(name,v):
    if name=='strategy_olr/config.py':
        marker='    capacity_weak_partial_fraction: float = 0.0\n'
        v=once(v,marker,marker+''.join('    '+x+': bool = False\n' for x in ['capacity_liquidation_tail','capacity_structural_partial','capacity_late_confirmation']))
        marker='        import math\n'
        checks='''        for field in ('capacity_liquidation_tail','capacity_structural_partial','capacity_late_confirmation'):
            if type(getattr(self,field)) is not bool:raise ValueError(field+' must be boolean')
        if self.capacity_liquidation_tail and (not self.capacity_joint_allocation or not self.capacity_loss_total_pct or self.capacity_opening_tail):
            raise ValueError('Liquidation-tail experiment requires joint loss budgets and replaces the opening-tail variant')
        if self.capacity_structural_partial and (self.capacity_weak_partial_fraction!=.5 or not self.capacity_exit_priority):
            raise ValueError('Structural partial requires registered half fraction and full-risk priority')
'''
        return once(v,marker,marker+checks)
    if name=='strategy_olr/research.py':
        marker='    "capacity_weak_partial_fraction",\n'
        return once(v,marker,marker+''.join('    "'+x+'",\n' for x in ['capacity_liquidation_tail','capacity_structural_partial','capacity_late_confirmation']))
    if name=='strategy_olr/core/logic.py':
        v=once(v,'    entry_plan = _entry_plan(config)\n','    entry_plan = _entry_plan(config)\n    from .precision_v3 import entry_plan as precision_entry_plan\n    entry_plan = precision_entry_plan(entry_plan, config)\n')
        marker='    protective_stop = _initial_protective_stop('
        v=once(v,marker,"    from .precision_v3 import entry_evidence\n    metadata.update(entry_evidence(symbol_state.session_bars, bar, config))\n"+marker)
        marker='    metadata["reserved_notional"] = int(qty) * estimated_price * (1.0 + entry_cost_buffer)\n'
        v=once(v,marker,'''    if config.capacity_liquidation_tail:
        from .liquidation_tail_v3 import evidence as liquidation_evidence
        multiplier=max(1.,config.capacity_redistribution_max_multiplier) if config.capacity_redistribute else 1.
        upper=equity*min(config.max_position_pct,weight*multiplier)
        # Full desired allocation upper bound includes any permitted recycled extras.
        metadata['liquidation_tail_evidence']=liquidation_evidence(state,symbol,candidate.sector,
            bar.timestamp+timedelta(minutes=5),max(upper,qty*estimated_price),config.capacity_participation)
'''+marker)
        marker='    pending_id = symbol_state.pending_exit_order_id\n'
        v=once(v,marker,'''    if config.capacity_structural_partial:
        from .precision_v3 import observe_entry_day
        observe_entry_day(position, bar)
'''+marker)
        return v
    if name=='strategy_olr/core/weak_partial.py':
        marker='    qty=min(position.qty_open-1,max(1,floor(position.qty_open*fraction)))\n'
        v=once(v,marker,'''    structural = None
    if config.capacity_structural_partial:
        from .precision_v3 import structural_evidence
        structural=structural_evidence(symbol_state,position,bar,risk)
        if structural is None:return None
'''+marker)
        return once(v,"    action=SubmitPartialExit(","    if structural is not None:metadata['structural_partial_evidence']=structural\n    action=SubmitPartialExit(")
    if name=='strategy_olr/core/loss_budget.py':
        marker='    return distance + price * costs\n'
        return once(v,marker,'''    if config.capacity_liquidation_tail:
        evidence=metadata.get('liquidation_tail_evidence')
        if not evidence or not isfinite(evidence['fraction']) or evidence['fraction']<0:
            raise ValueError('Liquidation sizing requires finite sourced evidence')
        distance=max(distance,price*evidence['fraction'])
'''+marker)
    if name=='scripts/portfolio_synergy_study.py':
        marker='    capacity_history_receipt = None\n'
        block='''    liquidation_panel = None
    if ocfg.capacity_liquidation_tail:
        from collections import defaultdict
        from strategy_olr.core.liquidation_tail_v3 import KEY as liquidation_key, empty as liquidation_empty, advance as liquidation_advance
        liquidation_panel=bound('liquidation_observations')
        if (liquidation_panel['database_hash']!=study['database_hash'] or set(liquidation_panel['universe'])!=capacity_observation_symbols or not liquidation_panel['prefix_invariance_passed']):
            raise ValueError('Unqualified liquidation source panel')
        prior_days=[d for d in liquidation_panel['sessions'] if d<str(min(ks))]
        if not prior_days:raise ValueError('Missing liquidation warm-up')
        prior_rows=[r for r in liquidation_panel['observations'] if r['day']<str(min(ks))]
        independent_prefix=liquidation_advance(liquidation_empty(),prior_rows,prior_days[-1],prior_days)
        liquidation_adapter=next(a for a in adapters if a.strategy_id=='OLR')
        if study.get('capacity_checkpoint'):
            restored=checkpoint.get(liquidation_key)
            if restored!=independent_prefix:raise ValueError('Actual IS liquidation history differs from independently sourced OOS prefix')
            liquidation_adapter.state.meta[liquidation_key]=json.loads(json.dumps(restored))
        else:liquidation_adapter.state.meta[liquidation_key]=independent_prefix
        liquidation_by_day=defaultdict(list)
        for row in liquidation_panel['observations']:liquidation_by_day[row['day']].append(row)
        write_json(args.output/'liquidation_restart_receipt.json',{'actual_is_terminal_restored':bool(study.get('capacity_checkpoint')),
            'independent_source_prefix_equal':True,'strictly_prior_sessions_only':True,'restart_date':str(min(ks)),
            'through':independent_prefix['through'],'sessions':len(independent_prefix['sessions']),
            'observations':len(independent_prefix['observations']),'state_hash':stable_signature(independent_prefix),
            'panel_sha256':digest(generation.root/study['liquidation_observations'])})
'''
        v=once(v,marker,block+marker)
        marker='        daily.append({"day": str(day), "equity": broker.equity_curve[-1], "cash": broker.cash,\n'
        v=once(v,marker,'''        if liquidation_panel is not None:
            liquidation_adapter.state.meta[liquidation_key]=liquidation_advance(
                liquidation_adapter.state.meta[liquidation_key],liquidation_by_day[str(day)],str(day),[str(day)])
'''+marker)
        marker="            'opening_tail_history_v1':olr_adapter.state.meta.get('opening_tail_history_v1'),\n"
        return once(v,marker,marker+"            'liquidation_tail_history_v3':olr_adapter.state.meta.get('liquidation_tail_history_v3'),\n")
    raise ValueError(name)

def prepare():
    t.register();hashes={};records=[];diffs=[]
    for w in ('train','oos'):
        ready=t.ENGINES/w/'ready.json'
        if ready.exists():
            saved=n.read(ready);hashes[w]=saved['study']['source_generation_hash'];records.append(saved['preparation_record']);continue
        assert not (t.ROOT/'cases').exists(),'Never alter generations after native execution starts'
        parent_path=t.PARENT/'runtime'/t.REFERENCE/w/'baseline/study.json';parent=n.read(parent_path)
        manifest=n.read(parent['manifest']);manifest.pop('manifest_hash',None)
        parent_files=Path(parent['code']);files=t.ENGINES/w/'inputs/files'
        for name,expected in manifest['files'].items():
            source=parent_files/name;target=files/name;assert n.sha(source)==expected,name
            target.parent.mkdir(parents=True,exist_ok=True)
            if source.suffix.lower() in ('.parquet','.sqlite','.db','.jsonl','.gz','.pickle','.joblib','.pkl'):
                if not target.exists():os.link(source,target)
                assert n.sha(target)==expected
            else:shutil.copy2(source,target)
        changes=[]
        for name in t.PATCHED:
            target=files/name;target.parent.mkdir(parents=True,exist_ok=True)
            old=target.read_text(encoding='utf-8') if target.exists() else ''
            if name in t.NEW_MODULES or name=='scripts/olr_liquidation_panel_v3.py':
                value=(t.REPO/name).read_text(encoding='utf-8')
            else:value=patch(name,old)
            compile(value,str(target),'exec');target.write_text(value,encoding='utf-8',newline='\n')
            changes.append({'window':w,'name':name,'before':manifest['files'].get(name),'after':n.sha(target)})
            manifest['files'][name]=n.sha(target)
            bundle=t.OUT/'review_bundle'/w/name;bundle.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,bundle)
            diffs.extend(difflib.unified_diff(old.splitlines(True),value.splitlines(True),fromfile=w+'/before/'+name,tofile=w+'/after/'+name))
        panel='research/precision_v3/liquidation_observations.json'
        subprocess.run([sys.executable,'-B','-u',str(files/'scripts/olr_liquidation_panel_v3.py'),'--study',str(parent_path),'--output',str(files/panel)],cwd=files,check=True)
        manifest['files'][panel]=n.sha(files/panel)
        actual=subprocess.check_output([sys.executable,'-B','-c','from backtests.auto.shared.cache_keys import source_generation_hash; print(source_generation_hash())'],cwd=files,text=True).strip()
        hashes[w]=actual;manifest['source_generation_hash']=actual
        study=copy.deepcopy(parent)
        for key in ('capacity_checkpoint','capacity_checkpoint_completion','capacity_checkpoint_result'):study.pop(key,None)
        study.update(code=str(files),source_generation_hash=actual,liquidation_observations=panel,
                     manifest=str(t.ENGINES/w/'inputs/manifest.json'),spec_path='research/precision_v3/control_spec.json')
        study['configuration']['input_manifest_path']=study['manifest']
        n.write(files/study['spec_path'],{k:v for k,v in study.items() if k not in ('code','manifest','spec_path','source_generation_hash')})
        manifest['files'][study['spec_path']]=n.sha(files/study['spec_path'])
        n.verify_market_bindings(manifest);assert not n.retained_parser_mismatches(manifest,files)
        n.write(study['manifest'],{**manifest,'manifest_hash':n.sig(manifest)})
        record={'window':w,'parent_source':parent['source_generation_hash'],'new_source':actual,'changes':changes,'panel_sha256':n.sha(files/panel)}
        n.write(ready,{'files':str(files),'study':study,'manifest':manifest,'preparation_record':record});records.append(record)
    panels={w:n.read(t.ENGINES/w/'inputs/files/research/precision_v3/liquidation_observations.json') for w in ('train','oos')}
    end=panels['train']['sessions'][-1]
    assert panels['train']['sessions']==[d for d in panels['oos']['sessions'] if d<=end]
    assert panels['train']['observations']==[r for r in panels['oos']['observations'] if r['day']<=end]
    plan=n.read(t.ROOT/'mechanism_plan.json');plan['snapshot_acceleration']['validated_source_hashes']=hashes;n.write(t.ROOT/'mechanism_plan.json',plan)
    n.write(t.ROOT/'preparation_complete.json',{'utc':s.utc(),'sources':hashes,'changes':records,'all_overlapping_observation_rows_exact':True,'is_through':end})
    if diffs:(t.OUT/'review_bundle/implementation.patch').write_text(''.join(diffs),encoding='utf-8')
    n.write(t.OUT/'implementation_receipt.json',{'changed_sources':[c for r in records for c in r['changes']]})

def parity(candidate,w,scenario='baseline',capped=False):
    old=t.PARENT/'cases'/t.REFERENCE/w/('100000000_'+scenario+('_capped' if capped else ''))
    new=b.output(candidate,w,scenario,capped)
    s.forensic.authenticate(old);s.forensic.authenticate(new)
    for name in ('daily.json','equity.json'):assert n.read(old/name)==n.read(new/name),name
    fields=('symbol','qty','entry_decision_time','entry_fill_time','entry_price','exit_fill_time','exit_price',
            'gross_pnl','commission','net_pnl','exit_reason','mfe','mae','r_multiple')
    def trades(path):return [{k:x.get(k) for k in fields} for x in n.read(path/'trades.json')]
    assert trades(old)==trades(new)
    assert n.read(old/'capacity_terminal_state.json')['capacity_history_v1']==n.read(new/'capacity_terminal_state.json')['capacity_history_v1']
    n.write(t.ROOT/f'control_parity_{w}_{scenario}_{capped}.json',{'passed':True,'parent':str(old),'new':str(new),
        'daily_and_intraday_marks_exact':True,'parent_completion':n.sha(old/'completion.json'),'new_completion':n.sha(new/'completion.json')})

if __name__=='__main__':prepare()

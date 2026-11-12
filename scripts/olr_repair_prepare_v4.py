"""Freeze a new generation from v3; never modify prior research inputs."""
import copy,difflib,os,shutil,subprocess,sys
from pathlib import Path
from scripts import olr_repair_study_v4 as t
n,s,b=t.n,t.s,t.b

def once(v,old,new):
    assert v.count(old)==1,(old,v.count(old));return v.replace(old,new)

def patch(name,v):
    if name=='strategy_olr/config.py':
        v=once(v,'    capacity_liquidation_tail: bool = False\n','    capacity_liquidation_tail: bool = False\n    capacity_risk_repair_mode: str = "off"\n    afternoon_empty_sector_flow_rescue: bool = False\n')
        v=once(v,'        import math\n','''        import math
        if self.capacity_risk_repair_mode not in ('off','quantity','separated','portfolio'):
            raise ValueError('Unknown risk repair mode')
        if self.capacity_risk_repair_mode!='off' and (not self.capacity_liquidation_tail or not self.capacity_joint_allocation or not all((self.capacity_loss_position_pct,self.capacity_loss_sector_pct,self.capacity_loss_total_pct))):
            raise ValueError('Risk repair requires sourced liquidation history and all joint loss budgets')
        if type(self.afternoon_empty_sector_flow_rescue) is not bool:
            raise ValueError('Empty-afternoon rescue must be boolean')
''')
        return once(v,'    "olr.afternoon.min_sector_flow": "afternoon_min_sector_flow",','    "olr.afternoon.empty_sector_flow_rescue": "afternoon_empty_sector_flow_rescue",\n    "olr.afternoon.min_sector_flow": "afternoon_min_sector_flow",')
    if name=='strategy_olr/research.py':
        v=once(v,'    "capacity_liquidation_tail",\n','    "capacity_liquidation_tail",\n    "capacity_risk_repair_mode",\n    "afternoon_empty_sector_flow_rescue",\n')
        marker='    cfg = config or OLRConfig()\n    if bool(cfg.shadow_reranker_enabled) and cfg.shadow_reranker_profile:'
        v=once(v,marker,'''    cfg = config or OLRConfig()
    if cfg.afternoon_empty_sector_flow_rescue:
        from .flow_rescue_v4 import select
        return select(snapshot,contexts,cfg)
    if bool(cfg.shadow_reranker_enabled) and cfg.shadow_reranker_profile:''')
        marker='''        if key not in _FINAL_CONFIG_HASH_EXCLUDED_FIELDS
    }
    return _stable_hash('''
        return once(v,marker,'''        if key not in _FINAL_CONFIG_HASH_EXCLUDED_FIELDS
    }
    if config.afternoon_empty_sector_flow_rescue:
        payload['afternoon_empty_sector_flow_rescue']=True
    return _stable_hash(''')
    if name=='strategy_olr/core/logic.py':
        marker="        from .liquidation_tail_v3 import evidence as liquidation_evidence\n"
        v=once(v,marker,"        from .risk_repair_v4 import quote as repaired_quote\n        liquidation_evidence=lambda st,sy,se,at,no,pa:repaired_quote(st,sy,se,at,no,pa,config.capacity_risk_repair_mode)\n")
        marker='    metadata["reserved_notional"] = int(qty) * estimated_price * (1.0 + entry_cost_buffer)\n'
        return once(v,marker,"    if candidate.metadata.get('empty_sector_flow_rescue_v4'):\n        metadata['empty_sector_flow_rescue_v4']=True\n"+marker)
    if name=='strategy_olr/core/loss_budget.py':
        marker='    if not enabled(config):\n'
        return once(v,marker,"    if config.capacity_risk_repair_mode!='off':\n        from .risk_repair_v4 import allocate as repaired_allocate\n        return repaired_allocate(intents,account,at,config,state,by_id,prior=prior)\n"+marker)
    if name=='strategy_olr/core/allocation.py':
        marker='    at = stamp + timedelta(minutes=5)\n'
        return once(v,marker,marker+'''    if config.capacity_risk_repair_mode=='portfolio':
        marks=state.meta.setdefault('risk_repair_marks_v4',{})
        for bar in bars:marks[bar.symbol]={'price':float(bar.close),'at':at.isoformat()}
''')
    if name=='scripts/portfolio_synergy_study.py':
        return once(v,'decision_stream = gzip.open(args.output / "native_decisions.jsonl.gz", "wt", encoding="utf-8") if allocation is not None or native_budget is not None else None',
            'decision_stream = gzip.open(args.output / "native_decisions.jsonl.gz", "wt", encoding="utf-8")')
    raise ValueError(name)

def prepare():
    t.register();hashes={};records=[];diffs=[]
    for w in ('train','oos'):
        ready=t.ENGINES/w/'ready.json'
        if ready.exists():
            saved=n.read(ready);hashes[w]=saved['study']['source_generation_hash'];records.append(saved['preparation_record']);continue
        assert not (t.ROOT/'cases').exists(),'Never alter sources after execution'
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
            value=(t.REPO/name).read_text(encoding='utf-8') if name in t.NEW_MODULES else patch(name,old)
            compile(value,str(target),'exec');target.write_text(value,encoding='utf-8',newline='\n')
            manifest['files'][name]=n.sha(target);changes.append(dict(window=w,name=name,before=n.sha(parent_files/name) if (parent_files/name).exists() else None,after=n.sha(target)))
            dest=t.OUT/'review_bundle'/w/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,dest)
            diffs.extend(difflib.unified_diff(old.splitlines(True),value.splitlines(True),fromfile='a/'+w+'/'+name,tofile='b/'+w+'/'+name))
        actual=subprocess.check_output([sys.executable,'-B','-c','from backtests.auto.shared.cache_keys import source_generation_hash; print(source_generation_hash())'],cwd=files,text=True).strip()
        hashes[w]=actual;manifest['source_generation_hash']=actual
        study=copy.deepcopy(parent)
        for key in ('capacity_checkpoint','capacity_checkpoint_completion','capacity_checkpoint_result'):study.pop(key,None)
        study.update(code=str(files),source_generation_hash=actual,manifest=str(t.ENGINES/w/'inputs/manifest.json'),spec_path='research/repair_v4/control_spec.json')
        study['configuration']['input_manifest_path']=study['manifest']
        n.write(files/study['spec_path'],{k:v for k,v in study.items() if k not in ('code','manifest','spec_path','source_generation_hash')})
        manifest['files'][study['spec_path']]=n.sha(files/study['spec_path'])
        n.verify_market_bindings(manifest);assert not n.retained_parser_mismatches(manifest,files)
        n.write(study['manifest'],{**manifest,'manifest_hash':n.sig(manifest)})
        record=dict(window=w,parent_source=parent['source_generation_hash'],new_source=actual,changes=changes,
            liquidation_panel_sha256=n.sha(files/study['liquidation_observations']))
        n.write(ready,dict(files=str(files),study=study,manifest=manifest,preparation_record=record));records.append(record)
    plan=n.read(t.ROOT/'mechanism_plan.json');plan['snapshot_acceleration']['validated_source_hashes']=hashes;n.write(t.ROOT/'mechanism_plan.json',plan)
    n.write(t.ROOT/'preparation_complete.json',dict(utc=s.utc(),sources=hashes,changes=records,inherited_causal_panels_unchanged=True))
    if diffs:(t.OUT/'review_bundle/implementation.patch').write_text(''.join(diffs),encoding='utf-8')
    n.write(t.OUT/'implementation_receipt.json',dict(changed_sources=[x for r in records for x in r['changes']],default_off=True,promotion=False))

def parity(candidate,w,scenario='baseline',capped=False):
    old=t.PARENT/'cases'/t.REFERENCE/w/('100000000_'+scenario+('_capped' if capped else ''))
    new=b.output(candidate,w,scenario,capped);s.forensic.authenticate(old);s.forensic.authenticate(new)
    for name in ('daily.json','equity.json'):assert n.read(old/name)==n.read(new/name),name
    keys=('symbol','qty','entry_decision_time','entry_fill_time','entry_price','exit_fill_time','exit_price','gross_pnl','commission','net_pnl','exit_reason','mfe','mae','r_multiple')
    econ=lambda p:[{k:x.get(k) for k in keys} for x in n.read(p/'trades.json')]
    assert econ(old)==econ(new)
    assert n.read(old/'capacity_terminal_state.json')['capacity_history_v1']==n.read(new/'capacity_terminal_state.json')['capacity_history_v1']
    n.write(t.ROOT/f'control_parity_{w}_{scenario}_{capped}.json',dict(passed=True,parent=str(old),new=str(new),daily_and_intraday_marks_exact=True,parent_completion=n.sha(old/'completion.json'),new_completion=n.sha(new/'completion.json')))

if __name__=='__main__':prepare()

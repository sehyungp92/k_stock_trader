"""Freeze isolated source generations without changing prior study inputs."""
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
from scripts import olr_opening_risk_study_v2 as t
n,s,b=t.n,t.s,t.b
PATCHED=(*t.PATCHED,'scripts/olr_opening_tail_panel_v2.py')


def patch_replay(value):
    marker='    capacity_history_receipt = None\n'
    assert value.count(marker)==1
    block='''    opening_tail_panel = None
    if ocfg.capacity_opening_tail:
        from collections import defaultdict
        from strategy_olr.core.opening_tail import KEY as opening_tail_key, empty as tail_empty, advance as tail_advance
        opening_tail_panel=bound('opening_tail_observations')
        if (opening_tail_panel['database_hash']!=study['database_hash']
                or set(opening_tail_panel['universe'])!=capacity_observation_symbols
                or not opening_tail_panel['prefix_invariance_passed']):
            raise ValueError('Unqualified opening-tail source panel')
        tail_days=opening_tail_panel['sessions']
        prior_days=[d for d in tail_days if d<str(min(ks))]
        if not prior_days:raise ValueError('Missing opening-tail warm-up history')
        prior_rows=[r for r in opening_tail_panel['observations'] if r['day']<str(min(ks))]
        independent_prefix=tail_advance(tail_empty(),prior_rows,prior_days[-1],prior_days)
        tail_adapter=next(a for a in adapters if a.strategy_id=='OLR')
        if study.get('capacity_checkpoint'):
            restored=checkpoint.get(opening_tail_key)
            if restored!=independent_prefix:
                raise ValueError('Actual IS terminal tail history differs from OOS sourced prefix')
            tail_adapter.state.meta[opening_tail_key]=json.loads(json.dumps(restored))
        else:
            tail_adapter.state.meta[opening_tail_key]=independent_prefix
        tail_by_day=defaultdict(list)
        for row in opening_tail_panel['observations']:tail_by_day[row['day']].append(row)
        write_json(args.output/'opening_tail_restart_receipt.json',{
            'actual_is_terminal_restored':bool(study.get('capacity_checkpoint')),
            'independent_source_prefix_equal':True,'restart_date':str(min(ks)),
            'through':independent_prefix['through'],'sessions':len(independent_prefix['sessions']),
            'observations':len(independent_prefix['observations']),
            'state_hash':stable_signature(independent_prefix),
            'panel_sha256':digest(generation.root/study['opening_tail_observations']),
            'strictly_prior_sessions_only':True})
'''
    value=value.replace(marker,block+marker)
    marker='        daily.append({"day": str(day), "equity": broker.equity_curve[-1], "cash": broker.cash,\n'
    assert value.count(marker)==1
    value=value.replace(marker,'''        if opening_tail_panel is not None:
            tail_adapter.state.meta[opening_tail_key]=tail_advance(
                tail_adapter.state.meta[opening_tail_key],tail_by_day[str(day)],str(day),[str(day)])
'''+marker)
    marker="            'capacity_history_v1':olr_adapter.state.meta.get('capacity_history_v1',{}),\n"
    assert value.count(marker)==1
    return value.replace(marker,marker+"            'opening_tail_history_v1':olr_adapter.state.meta.get('opening_tail_history_v1'),\n")


def prepare():
    t.register();hashes={};records=[]
    for w in ('train','oos'):
        ready=t.ENGINES/w/'ready.json'
        if ready.exists():
            saved=n.read(ready);hashes[w]=saved['study']['source_generation_hash']
            records.append(saved['preparation_record']);continue
        assert not (t.ROOT/'cases').exists(),'Never modify generations after native work begins'
        parent_study_path=t.PARENT/'runtime'/t.REFERENCE/w/'baseline/study.json'
        parent=n.read(parent_study_path);manifest=n.read(parent['manifest']);manifest.pop('manifest_hash',None)
        parent_files=Path(parent['code']);files=t.ENGINES/w/'inputs/files'
        print('Preparing isolated opening-risk generation',w,flush=True)
        for name,expected in manifest['files'].items():
            source=parent_files/name;target=files/name
            assert n.sha(source)==expected,name
            target.parent.mkdir(parents=True,exist_ok=True)
            if source.suffix.lower() in ('.parquet','.sqlite','.db','.jsonl','.gz','.pickle','.joblib','.pkl'):
                if not target.exists():os.link(source,target)
            else:shutil.copy2(source,target)
            if not os.path.samefile(source,target):assert n.sha(target)==expected,name
        before={name:manifest['files'].get(name) for name in PATCHED}
        for name in PATCHED:
            target=files/name;target.parent.mkdir(parents=True,exist_ok=True)
            if name=='strategy_olr/config.py':
                value=target.read_text(encoding='utf-8')
                marker='    capacity_loss_atr_multiple: float = 1.25\n';assert value.count(marker)==1
                value=value.replace(marker,marker+'    capacity_opening_tail: bool = False\n    capacity_weak_partial_fraction: float = 0.0\n')
                workspace=(t.REPO/name).read_text(encoding='utf-8')
                checks=workspace.split('        if type(self.capacity_opening_tail)')[1].split('        if type(self.capacity_exit_priority)')[0]
                value=value.replace('        import math\n','        import math\n        if type(self.capacity_opening_tail)'+checks,1)
                target.write_text(value,encoding='utf-8',newline='\n')
            elif name=='strategy_olr/research.py':
                value=target.read_text(encoding='utf-8');marker='    "capacity_loss_atr_multiple",\n';assert value.count(marker)==1
                target.write_text(value.replace(marker,marker+'    "capacity_opening_tail",\n    "capacity_weak_partial_fraction",\n'),encoding='utf-8',newline='\n')
            elif name=='scripts/portfolio_synergy_study.py':
                target.write_text(patch_replay(target.read_text(encoding='utf-8')),encoding='utf-8',newline='\n')
            else:shutil.copy2(t.REPO/name,target)
            manifest['files'][name]=n.sha(target)
        panel='research/opening_risk_v2/observations.json'
        command=[sys.executable,'-B','-u',str(files/'scripts/olr_opening_tail_panel_v2.py'),
                 '--study',str(parent_study_path),'--output',str(files/panel)]
        subprocess.run(command,cwd=files,check=True)
        manifest['files'][panel]=n.sha(files/panel)
        actual=subprocess.check_output([sys.executable,'-B','-c',
            'from backtests.auto.shared.cache_keys import source_generation_hash; print(source_generation_hash())'],cwd=files,text=True).strip()
        hashes[w]=actual;manifest['source_generation_hash']=actual
        study=copy.deepcopy(parent)
        for name in ('capacity_checkpoint','capacity_checkpoint_completion','capacity_checkpoint_result'):study.pop(name,None)
        study.update(code=str(files),source_generation_hash=actual,opening_tail_observations=panel,
            manifest=str(t.ENGINES/w/'inputs/manifest.json'),spec_path='research/opening_risk_v2/control_spec.json')
        study['configuration']['input_manifest_path']=study['manifest']
        n.write(files/study['spec_path'],{k:v for k,v in study.items() if k not in ('code','manifest','spec_path','source_generation_hash')})
        manifest['files'][study['spec_path']]=n.sha(files/study['spec_path'])
        n.verify_market_bindings(manifest);assert not n.retained_parser_mismatches(manifest,files)
        n.write(study['manifest'],{**manifest,'manifest_hash':n.sig(manifest)})
        record={'window':w,'parent_source':parent['source_generation_hash'],'new_source':actual,
            'before':before,'after':{name:n.sha(files/name) for name in PATCHED},'panel_sha256':n.sha(files/panel)}
        n.write(ready,{'files':str(files),'study':study,'manifest':manifest,'preparation_record':record})
        records.append(record)
    panels={w:n.read(t.ENGINES/w/'inputs/files/research/opening_risk_v2/observations.json') for w in ('train','oos')}
    end=panels['train']['sessions'][-1]
    assert panels['train']['sessions']==[d for d in panels['oos']['sessions'] if d<=end]
    assert panels['train']['observations']==[r for r in panels['oos']['observations'] if r['day']<=end]
    plan=n.read(t.ROOT/'mechanism_plan.json');plan['snapshot_acceleration'].update(validated_source_hashes=hashes)
    n.write(t.ROOT/'mechanism_plan.json',plan)
    n.write(t.ROOT/'preparation_complete.json',{'utc':s.utc(),'sources':hashes,'changes':records,
        'all_overlapping_observation_rows_exact':True,'is_through':end,
        'source_qualified_native_execution_only':True})


def parity(candidate,w,scenario='baseline',capped=False):
    old=t.PARENT/'cases'/t.REFERENCE/w/('100000000_'+scenario+('_capped' if capped else ''))
    new=b.output(candidate,w,scenario,capped)
    s.forensic.authenticate(old);s.forensic.authenticate(new)
    fields=('return_pct','drawdown_pct','trades','sessions')
    assert all(n.read(old/'result.json')[k]==n.read(new/'result.json')[k] for k in fields)
    for name in ('daily.json','equity.json'):assert n.read(old/name)==n.read(new/name),name
    a,z=n.read(old/'trades.json'),n.read(new/'trades.json');assert len(a)==len(z)
    economic=('symbol','qty','entry_decision_time','entry_fill_time','entry_price','exit_fill_time',
              'exit_price','gross_pnl','commission','net_pnl','exit_reason','mfe','mae','r_multiple')
    assert all({k:x.get(k) for k in economic}=={k:y.get(k) for k in economic} for x,y in zip(a,z))
    assert n.read(old/'capacity_terminal_state.json')['capacity_history_v1']==n.read(new/'capacity_terminal_state.json')['capacity_history_v1']
    n.write(t.ROOT/f'control_parity_{w}_{scenario}_{capped}.json',{'passed':True,'parent':str(old),'new':str(new),
        'daily_and_intraday_marks_exact':True,'economic_trade_fields':economic,'trades':len(a),
        'parent_completion':n.sha(old/'completion.json'),'new_completion':n.sha(new/'completion.json')})


if __name__=='__main__':prepare()

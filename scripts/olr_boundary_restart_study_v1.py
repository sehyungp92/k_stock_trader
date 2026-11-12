"""Literal IS-to-OOS state handoff versus broad, declared-universe history.

All executions retain the native source, ledger and capacity audits. Replay-only
changes reuse explicitly authenticated, unchanged feature implementations.
"""
from __future__ import annotations
import argparse,copy,json,os,shutil,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import olr_continuous_state_study_v1 as x
s=x.s;n=x.n
ROOT=s.ROOT.parent/'boundary_restart_v1'
ENGINES=Path.home()/'.codex/research/olr_boundary_restart_v1'


def patched_replay(value):
    before='    capacity_history_receipt = None\n'
    after='''    # Bind observation membership to the declared universe, not every archived file.
    import yaml
    from datetime import datetime, time, timedelta
    from strategy_common.clock import KST
    from backtests.auto.shared.cache_keys import stable_signature
    universe_name=Path(study['configuration']['universe'])
    universe_paths=(generation.root/universe_name,generation.root/'config'/universe_name)
    universe_path=next(p for p in universe_paths if generation.digest(p))
    capacity_observation_symbols={str(v).zfill(6) for v in yaml.safe_load(generation.read(universe_path))['symbols']}
    if study.get('capacity_checkpoint'):
        from strategy_common.market import MarketBar
        from strategy_olr.core.capacity import phase
        checkpoint=bound('capacity_checkpoint')
        parent_completion=bound('capacity_checkpoint_completion')
        parent_result=bound('capacity_checkpoint_result')
        checkpoint_hash=digest(generation.root/study['capacity_checkpoint'])
        if (not parent_completion['complete'] or parent_completion['artifacts']['capacity_terminal_state.json']!=checkpoint_hash
                or parent_completion['artifacts']['result.json']!=digest(generation.root/study['capacity_checkpoint_result'])
                or parent_result['audit_status']!='passed' or not parent_result['execution_validity_pass']
                or parent_result['terminal_inventory'] or parent_result['terminal_pending']):
            raise ValueError('Unqualified parent checkpoint')
        if (checkpoint['effective_config_hash']!=stable_signature(asdict(ocfg)) or checkpoint['scenario']!=args.scenario
                or checkpoint['capped']!=args.no_upward_compounding or checkpoint['initial_equity']!=study['initial_equity']
                or checkpoint['through']>=str(min(ks)) or checkpoint['source_generation_hash']!=parent_completion['source_generation_hash']
                or ocfg.capacity_broad_auction_history or ocfg.capacity_history_scope!='legacy_feed' or ocfg.capacity_value_mode!='off'):
            raise ValueError('Incompatible checkpoint contract')
        histories=checkpoint['capacity_history_v1'];cutoff=datetime.combine(min(ks),time(0),tzinfo=KST)
        checked=0
        for symbol,history in histories.items():
            if symbol not in capacity_observation_symbols:raise ValueError('Checkpoint symbol outside declared universe')
            for kind,limit in (('auction',20),('continuous',3)):
                samples=history[kind]
                if len(samples)>limit or len({t for t,v in samples})!=len(samples) or samples!=sorted(samples):
                    raise ValueError('Malformed checkpoint history')
                for stamp,volume in samples:
                    at=datetime.fromisoformat(stamp)
                    if at.tzinfo is None or at+timedelta(minutes=5)>cutoff or at.date()>=cutoff.date():
                        raise ValueError('Checkpoint contains future observations')
                    rows=list(connection.execute('SELECT o,h,l,c,v FROM bars WHERE day=? AND symbol=? AND ts=?',(stamp[:10],symbol,stamp)))
                    if len(rows)!=1 or rows[0][-1]!=volume:raise ValueError('Checkpoint observation is not in OOS retained source')
                    bar=MarketBar(symbol,at,'5m',*rows[0],source_fingerprint=study['database_hash'])
                    if phase(bar)!=kind:raise ValueError('Checkpoint observation phase changed')
                    checked+=1
        for adapter in adapters:
            if adapter.strategy_id=='OLR':adapter.state.meta['capacity_history_v1']=json.loads(json.dumps(histories))
        write_json(args.output/'capacity_restart_receipt.json',{'parent_checkpoint_sha256':checkpoint_hash,
            'parent_completion_sha256':digest(generation.root/study['capacity_checkpoint_completion']),
            'parent_source_generation_hash':checkpoint['source_generation_hash'],'parent_database_hash':checkpoint['database_hash'],
            'current_database_hash':study['database_hash'],'through':checkpoint['through'],'restart_date':str(min(ks)),
            'verified_observations':checked,'state_hash':stable_signature(histories),'restored_state':histories,
            'same_configuration_and_scenario':True,'all_samples_verified_against_oos_retained_ohlcv':True,
            'policy':'Literal IS terminal market-history state; account reset after verified flat terminal liquidation.'})
    capacity_history_receipt = None
'''
    value=x.panel.replace_once(value,before,after)
    value=x.panel.replace_once(value,"return [b for b in bars if phase(b)=='auction']",
        "return [b for b in bars if b.symbol in capacity_observation_symbols and phase(b)=='auction']")
    value=x.panel.replace_once(value,'    connection.close()\n', '''    olr_adapter=next((a for a in adapters if a.strategy_id=='OLR'),None)
    if olr_adapter is not None:
        write_json(args.output/'capacity_terminal_state.json',{'through':str(max(ks)),
            'capacity_history_v1':olr_adapter.state.meta.get('capacity_history_v1',{}),
            'effective_config_hash':stable_signature(asdict(ocfg)),'scenario':args.scenario,
            'capped':args.no_upward_compounding,'initial_equity':study['initial_equity'],
            'database_hash':study['database_hash'],'source_generation_hash':study['source_generation_hash']})
    connection.close()
''')
    return value


def register():
    if (ROOT/'registration.json').exists():return
    base=copy.deepcopy(n.read(x.ROOT/'candidate_plan.json')['baseline']);candidates=[]
    for label,policy,current,guard in (
        ('literal_carry_original','literal_carry',False,None),
        ('literal_carry_guard_1p0','literal_carry',False,1.),
        ('literal_carry_full_panel_guard_1p0','literal_carry',True,1.),
        ('broad_declared_universe_full_panel','broad_declared_universe',True,None),
        ('broad_declared_universe_full_panel_guard_1p0','broad_declared_universe',True,1.)):
        m=copy.deepcopy(base);m[x.PANEL_KEY]=current
        if policy=='broad_declared_universe':m['capacity_broad_auction_history']=True
        if guard is not None:m[s.EXIT].update(failed_followthrough_bars=1,failed_followthrough_close_r=-guard,failed_followthrough_mfe_r=4.)
        key=n.sig(m);cid='c'+n.sig({'mutations':m,'history_policy':policy})[:16]
        candidates.append({'id':cid,'label':label,'history_policy':policy,'mutations':m,'mutations_hash':key,'aliases':[],
            'kind':'boundary_restart_v1','paths':[list(p) for p,_,_ in n.mutation_edits(base,m)]})
    cp={'baseline':base,'baseline_id':candidates[0]['id'],'candidates':candidates};n.write(ROOT/'candidate_plan.json',cp)
    plan=copy.deepcopy(n.read(x.ROOT/'mechanism_plan.json'));plan.update(cp);n.write(ROOT/'mechanism_plan.json',plan)
    protocol=copy.deepcopy(n.read(x.ROOT/'protocol.json'));protocol['generation_directories']={w:str(ENGINES) for w in ('train','oos')}
    n.write(ROOT/'protocol.json',protocol)
    n.write(ROOT/'registration.json',{'registered_utc':s.utc(),'candidates':candidates,'windows':protocol['windows'],
        'purpose':'Separate literal continuous native history from the broader auction observation model. Repair observation membership to the source-qualified declared universe.',
        'literal_policy':'Replay IS from its original declared start. Carry its actual terminal liquidity state into OOS; copy no cash, positions, orders, or PnL labels. Match candidate, costs, capacity scenario and compounding mode.',
        'broad_policy':'Warm both windows from all completed prior auctions in the declared 103-stock universe and observe that same universe thereafter.',
        'acceptance':n.read(s.ROOT/'registration.json')['acceptance'],
        'selection':'Require continuous signal-panel history and either native-state continuity policy; compare original benchmark and matched continuous unguarded control. Prefer both returns improved, then the smaller return gain, then OOS return.',
        'failed_predecessor':str(x.ROOT/'cases/cd3e0ff51518ae701/train/100000000_baseline/assessment.json'),
        'failure_disposition':'All-database broad observation consumed out-of-universe archives. Preserve failure; never loosen source or basis qualification.',
        'independent_validation':False,'promotion':False})


def prepare():
    register();hashes={};records=[]
    for w in ('train','oos'):
        ready=ENGINES/w/'ready.json'
        if ready.exists():hashes[w]=n.read(ready)['study']['source_generation_hash'];continue
        print('Preparing source-qualified restart generation',w,flush=True)
        parent=n.read(x.ENGINES/w/'ready.json');files=ENGINES/w/'inputs/files';manifest=copy.deepcopy(parent['manifest']);manifest.pop('manifest_hash',None)
        for name,expected in manifest['files'].items():
            source=Path(parent['files'])/name;target=files/name
            assert n.sha(source)==expected,name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert n.sha(target)==expected,name
            elif source.suffix.lower() in ('.parquet','.sqlite','.db','.jsonl','.gz','.pickle'):
                # Immutable source observations only; executable/configuration files are copied.
                os.link(source,target)
            else:shutil.copy2(source,target)
        replay=files/'scripts/portfolio_synergy_study.py';before=n.sha(replay)
        replay.write_text(patched_replay(replay.read_text(encoding='utf-8')),encoding='utf-8',newline='\n')
        manifest['files']['scripts/portfolio_synergy_study.py']=n.sha(replay)
        actual=subprocess.check_output([sys.executable,'-B','-c','from backtests.auto.shared.cache_keys import source_generation_hash; print(source_generation_hash())'],cwd=files,text=True).strip()
        hashes[w]=actual;manifest['source_generation_hash']=actual;study=copy.deepcopy(parent['study'])
        study.update(code=str(files),source_generation_hash=actual,manifest=str(ENGINES/w/'inputs/manifest.json'),spec_path='research/boundary_restart_v1/control_spec.json')
        study['configuration']['input_manifest_path']=study['manifest']
        spec=files/study['spec_path'];n.write(spec,{k:v for k,v in study.items() if k not in ('code','manifest','spec_path','source_generation_hash')})
        manifest['files'][study['spec_path']]=n.sha(spec);n.write(study['manifest'],{**manifest,'manifest_hash':n.sig(manifest)})
        n.write(ready,{'files':str(files),'study':study,'manifest':manifest})
        records.append({'window':w,'source_before':parent['study']['source_generation_hash'],'source_after':actual,
            'replay_before':before,'replay_after':n.sha(replay),'feature_implementations_changed':False,
            'control_snapshot_reuse':{'path':study['olr_snapshots'],'sha256':n.sha(files/study['olr_snapshots'])},
            'unchanged_feature_code':{name:n.sha(files/name) for name in ('strategy_olr/config.py','strategy_olr/research.py','strategy_common/sector_intraday.py')}})
    plan=n.read(ROOT/'mechanism_plan.json');plan['snapshot_acceleration'].update(validated_source_hashes=hashes,implementation_sha256=n.sha(x.BUILDER))
    n.write(ROOT/'mechanism_plan.json',plan)
    n.write(ROOT/'preparation_complete.json',{'completed_utc':s.utc(),'source_hashes':hashes,'replay_only_changes':records,'builder_sha256':n.sha(x.BUILDER)})


def output(candidate,w,scenario='baseline',capped=False):
    return ROOT/'cases'/candidate['id']/w/('100000000_'+scenario+('_capped' if capped else ''))


def evaluate(candidate,w,scenario='baseline',capped=False):
    out=output(candidate,w,scenario,capped)
    if (out/'assessment.json').exists():
        a=n.read(out/'assessment.json');assert a['qualified'],str(out);s.forensic.authenticate(out);return a
    parent=n.prepare_case(ROOT,w,candidate,100_000_000);study=copy.deepcopy(n.read(parent));files=Path(study['code'])
    manifest=n.read(study['manifest']);manifest.pop('manifest_hash');tag=scenario+('_capped' if capped else '')
    prefix='research/boundary_restart_v1/runs/'+candidate['id']+'/'+w+'/'+tag+'/'
    runtime=ROOT/'runtime'/candidate['id']/w/tag;runtime.mkdir(parents=True,exist_ok=True)
    if w=='oos' and candidate['history_policy']=='literal_carry':
        trained=output(candidate,'train',scenario,capped);s.forensic.authenticate(trained)
        for key,name in (('capacity_checkpoint','capacity_terminal_state.json'),('capacity_checkpoint_completion','completion.json'),('capacity_checkpoint_result','result.json')):
            target=files/(prefix+name);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(trained/name,target)
            study[key]=target.relative_to(files).as_posix();manifest['files'][study[key]]=n.sha(target)
    study.update(manifest=str(files.parent/('boundary_'+candidate['id']+'_'+w+'_'+tag+'.manifest.json')),spec_path=prefix+'spec.json')
    assert (Path(study['manifest']).parent/'files').resolve()==files.resolve()
    study['configuration']['input_manifest_path']=study['manifest']
    spec=files/study['spec_path'];n.write(spec,{k:v for k,v in study.items() if k not in ('code','manifest','spec_path','source_generation_hash')})
    manifest['files'][study['spec_path']]=n.sha(spec);n.write(study['manifest'],{**manifest,'manifest_hash':n.sig(manifest)})
    registered=runtime/'study.json';n.write(registered,study)
    command=[sys.executable,'-B','-u',str(files/'scripts/portfolio_synergy_study.py'),'replay','--study',str(registered),
        '--recipe','candidate','--scenario',scenario,'--standalone','OLR','--output',str(out),'--full-audit']
    if capped:command.append('--no-upward-compounding')
    print({'candidate':candidate['label'],'window':w,'scenario':scenario,'capped':capped,'status':'running'},flush=True)
    rc=n.run_child(command,files,out.with_suffix('.log'))
    if rc:raise RuntimeError('Native replay failed: '+str(out.with_suffix('.log')))
    a=n.read(out/'result.json');audit=n.read(out/'execution_audit.json')
    a.update(candidate_id=candidate['id'],label=candidate['label'],window=w,capital=100_000_000,capped=capped,
        history_policy=candidate['history_policy'],output=str(out),qualified=bool(a['execution_validity_pass'] and audit['audit_pass'] and a['audit_status']=='passed' and not a['terminal_inventory'] and not a['terminal_pending']))
    n.write(out/'assessment.json',a)
    assert a['qualified'],'Full native audit did not qualify: '+str(out)
    s.forensic.authenticate(out);n.verify_window_contract(ROOT,w,study,out)
    print({k:a[k] for k in ('label','window','return_pct','drawdown_pct','trades','qualified')},flush=True)
    return a


def run():
    prepare()
    validator=REPO/'scripts/olr_boundary_restart_validate_v1.py'
    for w in ('train','oos'):
        receipt=ROOT/f'boundary_validation_{w}.json'
        if not receipt.exists():subprocess.run([sys.executable,'-B','-u',str(validator),'--root',str(ROOT),'--window',w],check=True)
        proof=n.read(receipt)
        assert proof['status']=='passed' and proof['validator_sha256']==n.sha(validator)
        assert proof['source_generation_hash']==n.read(ENGINES/w/'ready.json')['study']['source_generation_hash']
    for c in n.read(ROOT/'candidate_plan.json')['candidates']:
        for w in ('train','oos'):
            if (ROOT/'STOP.json').exists():raise RuntimeError('Stopped before next replay')
            evaluate(c,w)
    n.write(ROOT/'primary_completion.json',{'completed_utc':s.utc(),'native_replays':10,'promotion':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','prepare','run']);a=p.parse_args()
    {'register':register,'prepare':prepare,'run':run}[a.action]()

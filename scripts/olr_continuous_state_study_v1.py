"""Source-bound experiment carrying signal and liquidity history across windows.

All changes are applied to isolated frozen research engines. No live defaults
or historical result artifacts are overwritten.
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import olr_round1_optimization_v1 as s
from scripts import olr_continuous_panel_study_v1 as panel
n=s.n
ROOT=s.ROOT.parent/'continuous_state_v1'
ENGINES=Path.home()/'.codex/research/olr_continuous_state_v1'
BUILDER=REPO/'scripts/olr_continuous_state_builder_v1.py'
PANEL_KEY='olr.research.continuous_intraday_panel'


def register():
    if (ROOT/'registration.json').exists():return
    base=copy.deepcopy(n.read(s.ROOT/'candidate_plan.json')['baseline']);base[PANEL_KEY]=False;candidates=[]
    def add(label,warm=False,current=False,guard=None):
        m=copy.deepcopy(base)
        if warm:m['capacity_broad_auction_history']=True
        if current:m[PANEL_KEY]=True
        if guard is not None:m[s.EXIT].update(failed_followthrough_bars=1,failed_followthrough_close_r=-guard,failed_followthrough_mfe_r=4.)
        key=n.sig(m)
        candidates.append({'id':'c'+key[:16],'label':label,'kind':'continuous_state_repair',
            'mutations':m,'mutations_hash':key,'aliases':[],'paths':[list(p) for p,_,_ in n.mutation_edits(base,m)]})
    add('cold_state_parity_control')
    add('continuous_auction_history',warm=True)
    add('continuous_auction_history_guard_1p0',warm=True,guard=1.)
    add('both_histories_continuous_guard_1p0',warm=True,current=True,guard=1.)
    add('both_histories_continuous',warm=True,current=True)
    add('both_histories_continuous_guard_0p8',warm=True,current=True,guard=.8)
    add('both_histories_continuous_guard_1p2',warm=True,current=True,guard=1.2)
    plan=copy.deepcopy(n.read(panel.ROOT/'mechanism_plan.json'))
    plan.update(candidates=candidates,baseline=base,baseline_id=candidates[0]['id'])
    plan.pop('snapshot_acceleration',None);plan.pop('native_replay_acceleration',None)
    protocol=copy.deepcopy(n.read(s.ROOT/'protocol.json'))
    protocol.update(generation_directories={w:str(ENGINES) for w in ('train','oos')},
        review_scope='Causal full prior-auction warm start and explicit broad auction observation; current-panel ablation and local guard neighborhood.')
    n.write(ROOT/'candidate_plan.json',{'baseline':base,'baseline_id':candidates[0]['id'],'candidates':candidates})
    n.write(ROOT/'protocol.json',protocol);n.write(ROOT/'mechanism_plan.json',plan)
    n.write(ROOT/'registration.json',{'registered_utc':s.utc(),'candidates':candidates,
        'primary_queue':[c['id'] for c in candidates[:5]],'neighborhood_queue':[c['id'] for c in candidates[5:]],
        'causal_policy':'Seed every completed prior session from the authenticated database; retain native last 20 observations per symbol; observe all valid universe auctions thereafter.',
        'legacy_comparison':'This repairs cold initialization AND replaces incidental KALCB feed observation with an explicit broad market policy; not claimed to reproduce the legacy partial feed state.',
        'signal_policy':'Original or full prior-session current-afternoon panel, explicitly configured and separately hashed.',
        'history_start':'All available prior intraday sessions; account equity/holdings still start at the declared scoring boundary.',
        'acceptance':n.read(s.ROOT/'registration.json')['acceptance'],
        'staged_execution':'Run parity control and primary queue before neighborhood; skip demonstrably redundant native treatments only with equality evidence.',
        'adaptive_development':True,'independent_validation':False,'promotion':False})


def patch_replay(text):
    text=panel.replace_once(text,'WHERE day<? ORDER BY day DESC LIMIT 20','WHERE day<? ORDER BY day DESC')
    before="        capacity_history_receipt={'database_hash':study['database_hash'],'seed_rows':len(seeds),"
    after="""        capacity_state=next(a.state for a in adapters if a.strategy_id=='OLR')
        capacity_history_receipt={'history_policy':'all_prior_completed_sessions_native_20_per_symbol',
            'seeded_state':json.loads(json.dumps(capacity_state.meta['capacity_history_v1'])),
            'seeded_state_hash':stable_signature(capacity_state.meta['capacity_history_v1']),
            'database_hash':study['database_hash'],'seed_rows':len(seeds),"""
    return panel.replace_once(text,before,after)


def prepare():
    register();plan=n.read(ROOT/'mechanism_plan.json');hashes={};records=[]
    for window in ('train','oos'):
        destination=ENGINES/window;files=destination/'inputs/files';ready=destination/'ready.json'
        if ready.exists():hashes[window]=n.read(ready)['study']['source_generation_hash'];continue
        parent=n.read(panel.ENGINES/window/'ready.json');study=copy.deepcopy(parent['study'])
        manifest=copy.deepcopy(parent['manifest']);manifest.pop('manifest_hash',None)
        for name,expected in manifest['files'].items():
            source=Path(parent['files'])/name;target=files/name
            assert n.sha(source)==expected,name
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert n.sha(target)==expected,name
            else:shutil.copy2(source,target)
        config=files/'strategy_olr/config.py'
        value=panel.replace_once(config.read_text(encoding='utf-8'),'    require_complete_feature_prefix: bool = False\n',
            '    require_complete_feature_prefix: bool = False\n    research_continuous_intraday_panel: bool = False\n')
        value=panel.replace_once(value,'    "olr.research.min_sector_participation": "research_min_sector_participation",',
            '    "olr.research.continuous_intraday_panel": "research_continuous_intraday_panel",\n    "olr.research.min_sector_participation": "research_min_sector_participation",')
        config.write_text(value,encoding='utf-8',newline='\n')
        replay=files/'scripts/portfolio_synergy_study.py';replay.write_text(patch_replay(replay.read_text(encoding='utf-8')),encoding='utf-8',newline='\n')
        for name in ('strategy_olr/config.py','scripts/portfolio_synergy_study.py'):manifest['files'][name]=n.sha(files/name)
        actual=subprocess.check_output([sys.executable,'-B','-c','from backtests.auto.shared.cache_keys import source_generation_hash; print(source_generation_hash())'],cwd=files,text=True).strip()
        hashes[window]=actual;manifest['source_generation_hash']=actual
        study.update(code=str(files),manifest=str(destination/'inputs/manifest.json'),source_generation_hash=actual)
        study['configuration']['input_manifest_path']=study['manifest']
        study['spec_path']='research/continuous_state_v1/spec.json'
        # Control snapshots are rebuilt below under the explicit default-false panel policy.
        n.write(ready,{'study':study,'manifest':manifest,'files':str(files),'control_pending':True})
        records.append({'window':window,'source_before':parent['study']['source_generation_hash'],'source_after':actual})
    plan['snapshot_acceleration']={'module':BUILDER.name,'implementation_sha256':n.sha(BUILDER),
        'validated_source_hashes':hashes,'cache_root':str(panel.CACHE),'prefix_memo':False,
        'dependencies':{name:n.sha(REPO/'scripts'/name) for name in ('olr_snapshot_streaming.py','olr_snapshot_acceleration.py')},
        'semantic_scope':'Explicit intentional panel-history flag; default false preserves old scored features.'}
    n.write(ROOT/'mechanism_plan.json',plan)
    for window in ('train','oos'):
        ready=ENGINES/window/'ready.json';info=n.read(ready)
        if not info.get('control_pending'):continue
        files=Path(info['files']);study=info['study'];manifest=info['manifest']
        candidate=ROOT/'control_candidate.json';n.write(candidate,plan['candidates'][0])
        output=files/'research/continuous_state_v1/control_snapshots_v2.json'
        # The snapshot worker verifies its complete generation manifest.
        spec=files/study['spec_path'];n.write(spec,{k:v for k,v in study.items() if k not in ('code','manifest','spec_path','source_generation_hash')})
        manifest['files'][study['spec_path']]=n.sha(spec)
        n.write(study['manifest'],{**manifest,'manifest_hash':n.sig(manifest)});n.write(ready,info)
        log=ROOT/f'control_{window}_build.log'
        if log.exists():log.rename(log.with_name(log.stem+'_'+str(__import__('time').time_ns())+'.log'))
        rc=n.run_child([sys.executable,'-B','-u',str(Path(n.__file__).resolve()),'snapshots','--root',str(ROOT),'--window',window,'--candidate',str(candidate),'--output',str(output)],files,log)
        if rc:raise RuntimeError('State control snapshot failure '+window)
        study['olr_snapshots']=output.relative_to(files).as_posix()
        manifest['files'][study['olr_snapshots']]=n.sha(output)
        manifest['files'][output.with_suffix('.receipt.json').relative_to(files).as_posix()]=n.sha(output.with_suffix('.receipt.json'))
        n.write(spec,{k:v for k,v in study.items() if k not in ('code','manifest','spec_path','source_generation_hash')})
        manifest['files'][study['spec_path']]=n.sha(spec)
        n.write(study['manifest'],{**manifest,'manifest_hash':n.sig(manifest)})
        info.pop('control_pending');n.write(ready,info)
    n.write(ROOT/'preparation_complete.json',{'completed_utc':s.utc(),'source_hashes':hashes,'changes':records,'builder_sha256':n.sha(BUILDER)})


def revise_preparation():
    """Pre-replay repair: immutable seed evidence and unambiguous cache keys."""
    revision=ROOT/'preparation_revision_2.json'
    if revision.exists():return prepare()
    assert (ROOT/'preparation_complete.json').exists()
    assert not list((ROOT/'cases').glob('**/assessment.json')) and not (ROOT/'registrations').exists()
    archive=ROOT/'preparation_revisions/initial';archive.mkdir(parents=True,exist_ok=True)
    for name in ('candidate_plan.json','mechanism_plan.json','registration.json','preparation_complete.json','control_candidate.json'):
        shutil.copy2(ROOT/name,archive/name)
    for receipt in ROOT.glob('boundary_validation_*.json'):
        destination=archive/receipt.name
        assert receipt.resolve().is_relative_to(ROOT.resolve()) and destination.resolve().is_relative_to(ROOT.resolve())
        receipt.rename(destination)
    cp=n.read(ROOT/'candidate_plan.json');cp['baseline'][PANEL_KEY]=False;mapping={}
    for c in cp['candidates']:
        old=c['id'];c['mutations'].setdefault(PANEL_KEY,False)
        c['mutations_hash']=n.sig(c['mutations']);c['id']='c'+c['mutations_hash'][:16];mapping[old]=c['id']
        c['paths']=[list(p) for p,_,_ in n.mutation_edits(cp['baseline'],c['mutations'])]
    cp['baseline_id']=mapping[cp['baseline_id']];n.write(ROOT/'candidate_plan.json',cp)
    plan=n.read(ROOT/'mechanism_plan.json');plan.update(cp)
    reg=n.read(ROOT/'registration.json');reg['candidates']=cp['candidates']
    for key in ('primary_queue','neighborhood_queue'):reg[key]=[mapping[c] for c in reg[key]]
    reg['pre_replay_revision']='Explicit false panel flag prevents collisions with inherited old snapshot cache paths; seed evidence copied before streaming mutations.'
    reg['revised_utc']=s.utc();n.write(ROOT/'registration.json',reg)
    records=[]
    for w in ('train','oos'):
        ready=ENGINES/w/'ready.json';info=n.read(ready);files=Path(info['files']);old=info['study']['source_generation_hash']
        n.write(archive/(w+'_ready.json'),info)
        source=files/'scripts/portfolio_synergy_study.py'
        value=source.read_text(encoding='utf-8')
        value=panel.replace_once(value,"'seeded_state':capacity_state.meta['capacity_history_v1'],",
            "'seeded_state':json.loads(json.dumps(capacity_state.meta['capacity_history_v1'])),")
        source.write_text(value,encoding='utf-8',newline='\n')
        actual=subprocess.check_output([sys.executable,'-B','-c','from backtests.auto.shared.cache_keys import source_generation_hash; print(source_generation_hash())'],cwd=files,text=True).strip()
        info['study']['source_generation_hash']=actual;info['manifest']['source_generation_hash']=actual
        info['manifest']['files']['scripts/portfolio_synergy_study.py']=n.sha(source)
        info['control_pending']=True;n.write(ready,info)
        plan['snapshot_acceleration']['validated_source_hashes'][w]=actual
        records.append({'window':w,'old_source':old,'new_source':actual})
    n.write(ROOT/'mechanism_plan.json',plan)
    n.write(revision,{'time_utc':s.utc(),'before_any_native_outcomes':True,'changes':records,'candidate_id_mapping':mapping})
    prepare()


def run(which):
    prepare();plan=n.read(ROOT/'candidate_plan.json')
    for window in ('train','oos'):
        receipt=ROOT/f'boundary_validation_{window}.json'
        validator=REPO/'scripts/olr_continuous_state_validate_v1.py'
        if not receipt.exists():
            subprocess.run([sys.executable,'-B','-u',str(validator),'--root',str(ROOT),'--window',window],check=True)
        proof=n.read(receipt)
        assert proof['status']=='passed' and proof['validator_sha256']==n.sha(validator)
        assert proof['source_generation_hash']==n.read(ENGINES/window/'ready.json')['study']['source_generation_hash']
    candidates=plan['candidates'][:5] if which=='primary' else plan['candidates'][5:]
    s.mechanism.batch(ROOT,candidates,workers=1,phase=which)
    for r in n.results(ROOT):
        if r.get('qualified'):s.forensic.authenticate(Path(r['output']))
    n.write(ROOT/(which+'_completion.json'),{'completed_utc':s.utc(),'candidate_ids':[c['id'] for c in candidates],'promotion':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','prepare','revise','primary','neighborhood']);a=p.parse_args()
    if a.action=='register':register()
    elif a.action=='prepare':prepare()
    elif a.action=='revise':revise_preparation()
    else:run(a.action)

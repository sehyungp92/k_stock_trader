"""Isolated, source-bound continuous current-sector-panel experiment."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import olr_round1_optimization_v1 as s
n=s.n
ROOT=s.ROOT.parent/'continuous_panel_v1'
ENGINES=Path.home()/'.codex/research/olr_continuous_panel_v1'
CACHE=ROOT/'cache_sources'
BUILDER=REPO/'scripts/olr_continuous_panel_builder_v1.py'


def replace_once(text,before,after):
    assert text.count(before)==1,before[:100]
    return text.replace(before,after,1)


def patched_research(text):
    for func in ('build_afternoon_contexts','afternoon_selection_from_snapshot'):
        start=text.index('def '+func+'(');end=text.index('\n\n\ndef ',start+1)
        part=text[start:end]
        part=replace_once(part,'    sector_map: Mapping[str, str] | None = None,\n',
            '    sector_map: Mapping[str, str] | None = None,\n    *,\n    sector_intraday_panel=None,\n')
        if func=='build_afternoon_contexts':
            part=replace_once(part,'    sector_panel = build_sector_intraday_panel(\n',
                '    # Explicit source-bound history panel; default preserves saved round 1.\n    if sector_intraday_panel is not None and sector_intraday_panel.cutoff_label != INTRADAY_SELECTION_CUTOFF_LABEL:\n        raise ValueError("Current-sector history panel has wrong decision cutoff")\n    sector_panel = sector_intraday_panel if sector_intraday_panel is not None else build_sector_intraday_panel(\n')
        else:
            part=replace_once(part,'contexts = build_afternoon_contexts(snapshot, bars_by_symbol, cfg, sector_map=sector_map)',
                'contexts = build_afternoon_contexts(snapshot, bars_by_symbol, cfg, sector_map=sector_map, sector_intraday_panel=sector_intraday_panel)')
        text=text[:start]+part+text[end:]
    return text


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    original=n.read(s.PRIOR/'candidate_plan.json');catalog={c['id']:c for c in original['candidates']}
    catalog.update({c['id']:c for c in n.read(s.ROOT/'candidate_plan.json')['candidates']})
    ids=n.read(ROOT/'registration.json')['candidates'] if (ROOT/'registration.json').exists() else [original['baseline_id'],'c272fe0390df855bd']
    plan=copy.deepcopy(n.read(s.PRIOR/'mechanism_plan.json'))
    plan['candidates']=[catalog[cid] for cid in ids]
    plan.pop('native_replay_acceleration',None)
    protocol=copy.deepcopy(n.read(s.ROOT/'protocol.json'))
    protocol.update(generation_directories={w:str(ENGINES) for w in ('train','oos')},
        review_scope='Separate full-history current-afternoon sector panel; compare original and guard policies.',
        feature_treatment='continuous_current_afternoon_panel_v1')
    n.write(ROOT/'candidate_plan.json',{'baseline_id':ids[0],'baseline':original['baseline'],'candidates':plan['candidates']})
    n.write(ROOT/'protocol.json',protocol)
    registered=[];source_hashes={}
    for window in ('train','oos'):
        destination=ENGINES/window;files=destination/'inputs/files'
        ready=destination/'ready.json'
        if ready.exists():
            source_hashes[window]=n.read(ready)['study']['source_generation_hash'];continue
        print({'preparing_feature_generation':window},flush=True)
        parent=s.PRIOR/'registrations'/window/'c272fe0390df855bd_100000000'
        source_study=n.read(parent/'replay_study.json');source=Path(source_study['code'])
        manifest=n.read(source_study['manifest']);manifest.pop('manifest_hash')
        for name,expected in manifest['files'].items():
            target=files/name
            if target.exists():
                assert n.sha(target)==expected,name
            else:
                assert n.sha(source/name)==expected,name
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
        research=files/'strategy_olr/research.py'
        before=n.sha(research);research.write_text(patched_research(research.read_text(encoding='utf-8')),encoding='utf-8',newline='\n')
        manifest['files']['strategy_olr/research.py']=n.sha(research)
        actual=subprocess.check_output([sys.executable,'-B','-c','from backtests.auto.shared.cache_keys import source_generation_hash; print(source_generation_hash())'],cwd=files,text=True).strip()
        manifest['source_generation_hash']=actual;source_hashes[window]=actual
        study=copy.deepcopy(source_study)
        study.update(code=str(files),source_generation_hash=actual,manifest=str(destination/'inputs/manifest.json'))
        study['configuration']['input_manifest_path']=study['manifest']
        request=n.read(files/study['olr_request']);request['mutations']={k:v for k,v in original['baseline'].items() if not k.startswith('capacity_')}
        request['config']['initial_mutations']=request['mutations']
        study['olr_request']='research/continuous_panel_v1/control_request.json'
        n.write(files/study['olr_request'],request);manifest['files'][study['olr_request']]=n.sha(files/study['olr_request'])
        study['spec_path']='research/continuous_panel_v1/spec.json'
        n.write(files/study['spec_path'],{k:v for k,v in study.items() if k not in ('manifest','code','spec_path','source_generation_hash')})
        manifest['files'][study['spec_path']]=n.sha(files/study['spec_path'])
        n.write(study['manifest'],{**manifest,'manifest_hash':n.sig(manifest)})
        # The old snapshots declare dates only until the corrected control is built.
        n.write(ready,{'study':study,'manifest':manifest,'files':str(files),'feature_control_pending':True})
        old_study=n.read(parent/'study.json');snap=Path(old_study['code'])/old_study['olr_snapshots']
        receipt=n.read(snap.with_suffix('.receipt.json'));cache=Path(receipt['acceleration_profile']['cache'])
        n.write(CACHE/window/'source.json',{'cache':str(cache),'index_sha256':n.sha(cache/'index.json'),
            'source_receipt':str(snap.with_suffix('.receipt.json')),'source_receipt_sha256':n.sha(snap.with_suffix('.receipt.json'))})
        registered.append({'window':window,'original_study':str(parent/'replay_study.json'),'original_study_sha256':n.sha(parent/'replay_study.json'),
            'source_before':source_study['source_generation_hash'],'source_after':actual,'research_before':before,'research_after':n.sha(research),'market_observations_changed':False})
    plan['snapshot_acceleration']={'module':BUILDER.name,'implementation_sha256':n.sha(BUILDER),
        'validated_source_hashes':source_hashes,'cache_root':str(CACHE),'prefix_memo':False,
        'dependencies':{name:n.sha(REPO/'scripts'/name) for name in ('olr_snapshot_streaming.py','olr_snapshot_acceleration.py')},
        'semantic_scope':'Intentional current-panel feature change; not parity-preserving acceleration.'}
    n.write(ROOT/'mechanism_plan.json',plan)
    prior_registration=n.read(ROOT/'registration.json') if (ROOT/'registration.json').exists() else {}
    if prior_registration and prior_registration.get('builder_sha256')!=n.sha(BUILDER):
        archive=ROOT/('registration_builder_'+prior_registration['builder_sha256'][:12]+'.json')
        if not archive.exists():n.write(archive,prior_registration)
    n.write(ROOT/'registration.json',{'registered_utc':prior_registration.get('registered_utc',datetime.now(timezone.utc).isoformat()),'candidates':ids,'generations':registered or prior_registration.get('generations',[]),
        'builder_sha256':n.sha(BUILDER),'hypothesis':'Use preceding IS bars for current as well as previous-session sector relative volume; preserve complete-prefix exclusions, exact cutoff and original market observations.',
        'adaptive_development':True,'promotion':False})
    for window in ('train','oos'):
        ready=ENGINES/window/'ready.json';info=n.read(ready)
        if not info.get('feature_control_pending'):continue
        candidate=ROOT/'control_candidate.json';n.write(candidate,catalog[ids[0]])
        files=Path(info['files']);output=files/'research/continuous_panel_v1/control_snapshots.json'
        log=ROOT/f'control_{window}_build.log'
        if log.exists():
            archived=log.with_name(log.stem+'_attempt_'+str(__import__('time').time_ns())+'.log')
            assert log.resolve().is_relative_to(ROOT.resolve()) and archived.resolve().is_relative_to(ROOT.resolve())
            log.rename(archived)
        rc=n.run_child([sys.executable,'-B','-u',str(Path(n.__file__).resolve()),'snapshots','--root',str(ROOT),'--window',window,'--candidate',str(candidate),'--output',str(output)],files,log)
        if rc:raise RuntimeError('Continuous-panel control builder failed: '+window)
        info['study']['olr_snapshots']=output.relative_to(files).as_posix()
        info['manifest']['files'][info['study']['olr_snapshots']]=n.sha(output)
        info['manifest']['files'][output.with_suffix('.receipt.json').relative_to(files).as_posix()]=n.sha(output.with_suffix('.receipt.json'))
        spec=files/info['study']['spec_path'];n.write(spec,{k:v for k,v in info['study'].items() if k not in ('manifest','code','spec_path','source_generation_hash')})
        info['manifest']['files'][info['study']['spec_path']]=n.sha(spec)
        n.write(info['study']['manifest'],{**info['manifest'],'manifest_hash':n.sig(info['manifest'])})
        info.pop('feature_control_pending');n.write(ready,info)
    n.write(ROOT/'preparation_complete.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'source_hashes':source_hashes})


def validate():
    for window in ('train','oos'):
        subprocess.run([sys.executable,'-B','-u',str(REPO/'scripts/olr_continuous_panel_validate_v1.py'),
            '--root',str(ROOT),'--window',window,'--repo',str(REPO)],cwd=Path(n.read(ENGINES/window/'ready.json')['files']),check=True)


def run():
    prepare();plan=n.read(ROOT/'candidate_plan.json')
    if not all((ROOT/f'feature_validation_{w}.json').exists() for w in ('train','oos')):validate()
    s.mechanism.batch(ROOT,plan['candidates'],workers=1,phase='continuous_panel')
    for row in n.results(ROOT):s.forensic.authenticate(Path(row['output']))
    n.write(ROOT/'completion.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'full_native_replays':2*len(plan['candidates']),'promotion':False})


def extend():
    cid='c8d12020033107bb0';reg=n.read(ROOT/'registration.json')
    if cid in reg['candidates']:return
    candidate=next(c for c in n.read(s.ROOT/'candidate_plan.json')['candidates'] if c['id']==cid)
    extension={'registered_utc':datetime.now(timezone.utc).isoformat(),'candidate_id':cid,
        'reason':'Pair the stronger audited 1.0R guard with the independently specified continuous current-panel correction; retain the original and 1.2R controls.',
        'consumed_guard_assessments':{w:n.sha(s.ROOT/'cases'/cid/w/'100000000_baseline/assessment.json') for w in ('train','oos')},'adaptive_development':True,'promotion':False}
    n.write(ROOT/'guard_1p0_extension.json',extension)
    reg['candidates'].append(cid);n.write(ROOT/'registration.json',reg)
    for name in ('candidate_plan.json','mechanism_plan.json'):
        p=n.read(ROOT/name);p['candidates'].append(candidate);n.write(ROOT/name,p)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','validate','extend','run']);a=p.parse_args()
    {'prepare':prepare,'validate':validate,'extend':extend,'run':run}[a.action]()

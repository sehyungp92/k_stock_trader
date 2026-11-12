"""Evidence-bound publication for the finite allocation challenge; research only."""
from pathlib import Path

from scripts.portfolio_allocation_challenge import BATTERY, PROTOCOL, OVERNIGHT_PROTOCOL, select, risk_metrics, passes, budget_jobs, midpoint_grid
from scripts.portfolio_native_budget_study import read, checked, verified_spec
from scripts.replay_selected_portfolio import binding, bound, checked_case, digest

VERSION='portfolio-allocation-challenge-profile-v1'


def verify_frozen_bytes(study,cache):
    """Hash current frozen bytes; deduplicate hardlinks within this verification."""
    from kis_core.input_manifest import InputManifest
    manifest=InputManifest(study['manifest'])
    for name,expected in manifest.files.items():
        path=Path(name);stat=path.stat()
        identity=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if identity not in cache:cache[identity]=digest(path)
        if cache[identity]!=expected:raise ValueError('Changed frozen generation bytes: '+name)


def validate_overnight_registration(root,reference,selection):
    parent=read(reference/'mechanisms/study.json')
    base=parent['candidates']['k075_o075']
    expected_recipes,expected_jobs=budget_jobs(base,OVERNIGHT_PROTOCOL['budget_grid'])
    data={f'k{k*100:g}_o{o*100:g}':selection['candidates'][f'k{k*100:g}_o{o*100:g}']
          for k,o in OVERNIGHT_PROTOCOL['budget_grid']}
    grid=midpoint_grid(data)
    decision={'registered_midpoints_triggered':grid,'coarse_completions':{
        n:{s:m['completion'] for s,m in b.items()} for n,b in data.items()}}
    if read(root/'stage_decision.json')!=decision:raise ValueError('Midpoint decision does not reconstruct')
    groups=[('coarse',expected_recipes,expected_jobs)]
    if grid:
        recipes,jobs=budget_jobs(base,grid);groups.append(('midpoint',recipes,jobs))
    elif (root/'midpoint/registration.json').exists():raise ValueError('Unregistered midpoint study')
    expected_candidates=set(data)
    for child,recipes,jobs in groups:
        study=read(root/child/'study.json');reg=read(root/child/'registration.json');verified_spec(study)
        if (study['candidates']!=recipes or reg['jobs']!=jobs
            or study['source_generation_hash']!=parent['source_generation_hash']):
            raise ValueError('Unregistered budget, scenario or source change')
        expected_candidates.update(j['candidate'] for j in jobs)
        if child=='midpoint':bound(reg['stage_decision'])
    if set(selection['candidates'])!=expected_candidates:raise ValueError('Unexpected selection candidate set')


def make_challenge_profile(root):
    root=Path(root).resolve()
    protocol=read(root/'protocol.json')
    expected_protocol=OVERNIGHT_PROTOCOL if protocol['version']==OVERNIGHT_PROTOCOL['version'] else PROTOCOL
    if any(protocol.get(k)!=v for k,v in expected_protocol.items()):raise ValueError('Unsupported challenge protocol')
    reference=Path(protocol.get('reference_challenge_root',str(root)))
    initial_evidence=None
    if reference!=root:
        bound(protocol['reference_protocol'])
        if read(reference/'selection.json')['winner'] is not None or select(reference,save=False)['winner'] is not None:
            raise ValueError('Overnight follow-up requires the documented initial-grid failure')
        initial_evidence=binding(reference/'selection.json')
    selection=read(root/'selection.json')
    if selection!=select(root,save=False):raise ValueError('Selection does not reconstruct from registered complete cases')
    if reference!=root:validate_overnight_registration(root,reference,selection)
    name=selection['winner']
    if not name or name.startswith('random'):raise ValueError('No eligible selected research candidate')
    qualified=read(root/'qualification.json')
    if qualified['winner']!=name or qualified['selection_sha256']!=digest(root/'selection.json'):
        raise ValueError('Unbound qualification')
    cases={};paths=set();studies={};verified_bytes={}
    for label,m in selection['candidates'][name].items():
        study_path=Path(m['study']);study=read(study_path);verified_spec(study)
        if str(study_path) not in studies:verify_frozen_bytes(study,verified_bytes)
        if study['initial_equity']!=25_000_000:raise ValueError('Wrong account capital')
        path=study_path.parent/'qualified'/(m['recipe']+'_'+m['scenario']);paths.add(str(path))
        checked(path,True)
        result,request=checked_case(path,study['source_generation_hash'],m['recipe'],m['scenario'],full=True)
        if request['recipe']!=study['candidates'][m['recipe']]:raise ValueError('Qualified recipe changed')
        actual=risk_metrics(path)
        if not passes(actual,label,protocol)['pass']:raise ValueError('Qualified risk battery failure')
        for file in ('trades.json','daily.json','equity.json'):
            if digest(path/file)!=digest(Path(m['case'])/file):raise ValueError('Qualification changed economic paths')
        cases[label]={'output':str(path),'recipe':m['recipe'],'scenario':m['scenario'],
            'evidence':{f:binding(path/f) for f in ('completion.json','request.json','result.json','execution_audit.json','allocation_audit.json')},
            'metrics':{k:v for k,v in actual.items() if k!='completion'}}
        studies[str(study_path)]={'study':binding(study_path),'manifest':binding(study['manifest']),
            'registration':binding(study_path.parent/'registration.json')}
    if set(cases)!=set(BATTERY) or paths!=set(qualified['cases']):raise ValueError('Incomplete qualification battery')
    base=selection['candidates'][name]['baseline'];study=read(base['study'])
    incumbent=bound(protocol['incumbent_profile'])
    # The old pointer was intentionally retained until publication, but it will
    # change on promotion; bind the immutable old profile, not that mutable pointer.
    from scripts.replay_selected_portfolio import verify_profile
    verify_profile(protocol['incumbent_profile']['path'])
    verify_frozen_bytes(read(incumbent['study']['path']),verified_bytes)
    bridge={}
    mechanisms=reference/'mechanisms'
    mechanism_study=read(mechanisms/'study.json');verified_spec(mechanism_study)
    verify_frozen_bytes(mechanism_study,verified_bytes)
    studies[str(mechanisms/'study.json')]={'study':binding(mechanisms/'study.json'),
        'manifest':binding(mechanism_study['manifest']),'registration':binding(mechanisms/'registration.json')}
    for scenario in ('baseline','combined'):
        old=Path(incumbent['qualified_cases'][scenario]['output'])
        new=mechanisms/'cases'/('k075_o075_'+scenario)
        checked(new)
        hashes={f:digest(new/f) for f in ('trades.json','daily.json','equity.json')}
        if any(h!=digest(old/f) for f,h in hashes.items()):raise ValueError('Incumbent bridge changed')
        bridge[scenario]={'new':binding(new/'completion.json'),'old':binding(old/'completion.json'),'economic_hashes':hashes}
    benchmark_ref=read(reference/'partition_study.json');benchmark=Path(benchmark_ref['path']).parent
    benchmark_study=bound(benchmark_ref);verified_spec(benchmark_study)
    verify_frozen_bytes(benchmark_study,verified_bytes)
    benchmark_cases={}
    for mode in ('shared','partition','daily_rebalanced'):
        for scenario in ('baseline','combined'):
            path=benchmark/'cases'/(mode+'_'+scenario);checked(path)
            audit=read(path/'allocation_audit.json')
            if mode=='daily_rebalanced' and (not audit.get('wealth_transfers') or any(
                abs(t['kalcb_transfer'])>max(0.,t['donor_cash'])+1e-7 for t in audit['wealth_transfers'])):
                raise ValueError('Cash-funded benchmark transfer evidence missing or invalid')
            benchmark_cases[mode+'_'+scenario]=binding(path/'completion.json')
    return {'version':VERSION,'status':'research_selected_full_native_audits_passed',
        'scope':'reused_development_only','live_deployment_enabled':False,'independent_performance_established':False,
        'recipe':base['recipe'],'settings':study['candidates'][base['recipe']],
        'initial_equity_krw':25_000_000,'evaluation_window':study['configuration']['date_range'],
        'source_generation_hash':study['source_generation_hash'],'study':binding(base['study']),
        'challenge_root':str(root),'protocol':binding(root/'protocol.json'),'selection':binding(root/'selection.json'),
        'qualification':binding(root/'qualification.json'),'studies':studies,'qualified_cases':cases,
        'incumbent_profile':protocol['incumbent_profile'],'incumbent_bridge':bridge,
        'initial_challenge_selection':initial_evidence,
        'funded_benchmarks':{'study':benchmark_ref,'cases':benchmark_cases,'scope':'Physical/account-audited mechanism diagnostics; not full source-qualified portfolio candidates.'},
        'risk_interpretation':'Empirical eight-scenario gates, not future loss bounds. Native ranks are not calibrated expected return.',
        'operational_scope':'Research only; native broker funding/execution parity and unused-data performance remain unestablished.'}


def verify_challenge_profile(profile):
    if (profile.get('status')!='research_selected_full_native_audits_passed'
        or profile.get('live_deployment_enabled') is not False
        or profile.get('independent_performance_established') is not False):
        raise ValueError('Qualified research-only challenge profile required')
    for name in ('protocol','selection','qualification','incumbent_profile','study'):bound(profile[name])
    for study in profile['studies'].values():
        for item in study.values():bound(item)
    for case in profile['qualified_cases'].values():
        for item in case['evidence'].values():bound(item)
    if profile.get('initial_challenge_selection'):bound(profile['initial_challenge_selection'])
    bound(profile['funded_benchmarks']['study'])
    for item in profile['funded_benchmarks']['cases'].values():bound(item)
    expected=make_challenge_profile(profile['challenge_root'])
    if profile!=expected:raise ValueError('Challenge profile differs from reconstructed evidence')
    return profile

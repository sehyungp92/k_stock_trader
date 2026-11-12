"""Authenticate completed v2 accounts; preserve the failed numerical execution."""
from pathlib import Path

from .retention_research_io_v1 import read, write, bound, verify, digest, link
from .late_artifact_validation_v2 import validate_native
from .population_research_io_v1 import bridge_accounts
from .research_process_v1 import exact_process


def authenticate(root):
    root = Path(root).resolve()
    protocol = read(root/'protocol.json')
    if protocol['version'] != 'opening-complete-account-v2-validator-recovery':
        raise ValueError('Expected registered v2 predecessor')
    for identity in (read(root/'supervisor_start.json'),read(root/'status.json')['child']):
        if exact_process(identity):
            raise ValueError('Predecessor still running; do not duplicate')
    verify(protocol['programs']+protocol['bindings']+protocol['database_bindings'])
    failed = read(root/'failed_stage.json')
    if failed['stage'] != 'uniform_oos_participation' or failed['returncode'] != 1:
        raise ValueError('Not the diagnosed numerical failure')
    verify([failed['log']])
    if not Path(failed['log']['path']).read_text(encoding='utf-8').rstrip().endswith(
            'ValueError: Joint allocation failed feasibility: Iteration limit reached'):
        raise ValueError('Unexpected native failure')
    progress = read(root/'progress.json')
    expected = {(n,p,s) for n in ('control','joint','uniform') for p in ('is','oos')
                for s in ('baseline','participation','cost','combined')
                if not (n=='uniform' and p=='oos' and s!='baseline')}
    if len(progress)!=21 or {(r['name'],r['period'],r['scenario']) for r in progress}!=expected or (root/'completion.json').exists():
        raise ValueError('Unexpected predecessor progress')
    bindings = [bound(root/name) for name in ('protocol.json','study.json','progress.json','status.json',
        'supervisor_start.json','supervisor.log','supervisor.err.log','failed_stage.json')]+[failed['log']]
    cases = []
    for name,period,scenario in sorted(expected):
        case = root/'cases'/name/period/scenario
        profile_path = root/'runtime'/name/period/'profile.json'
        profile = read(profile_path)
        verify([profile[k] for k in ('plan','manifest','policy')])
        metrics = validate_native(case,profile,scenario,save=False)
        done = read(case/'completion.json')
        if done['mode']!=name or not done['complete']:
            raise ValueError('Wrong completed account identity')
        before = {x['path']:x['sha256'] for x in done['origins_before']}
        after = {x['path']:x['sha256'] for x in done['origins_after']}
        if any(after.get(path)!=sha for path,sha in before.items()):
            raise ValueError('Runtime imported source changed')
        verify(done['origins_after']+[done['decisions'],done['contrasts']])
        manifest = read(profile['manifest']['path'])
        code = Path(profile['manifest']['path']).parent/'files'
        for path,sha in after.items():
            relative = Path(path).resolve().relative_to(code.resolve()).as_posix()
            if manifest['files'].get(relative)!=sha:
                raise ValueError('Imported module is not source-bound')
        bridge = None
        if name=='control':
            bridge = bridge_accounts(case,Path(protocol['references'][period][scenario]))
            if not bridge['passed']:
                raise ValueError('Control does not reproduce parent')
        files = {str(p.relative_to(case)):bound(p) for p in sorted(case.rglob('*')) if p.is_file()}
        bindings += [bound(profile_path),*[profile[k] for k in ('manifest','plan','policy')],*files.values()]
        cases.append(dict(name=name,period=period,scenario=scenario,case=str(case),profile=bound(profile_path),
            files={k:v for k,v in files.items() if k not in ('independent_account_verification.json','activity.json')},
            metrics=metrics,bridge=bridge))
        print(f'Authenticated preserved case: {name}/{period}/{scenario}',flush=True)
    partial = root/'cases/uniform/oos/participation'
    bindings += [bound(p) for p in sorted(partial.rglob('*')) if p.is_file()]
    return dict(predecessor=str(root),bindings=bindings,cases=cases,reused_cases=21,new_executions=3,
        overlays={k:v['sha256'] for k,v in protocol['overlays'].items()},
        repair='Equivalent convex epigraph fallback only when the original nonsmooth solver fails; '
               'same utility, paths, costs, budget, funding and integer checks. Original successful solver decisions retained.',
        economic_parameters_changed=False,failed_partial_preserved=str(partial))


def publish(root,proof):
    root = Path(root)
    verify(proof['bindings'])
    for item in proof['cases']:
        out = root/'cases'/item['name']/item['period']/item['scenario']
        for relative,binding in item['files'].items():
            if (out/relative).exists():
                raise ValueError('Do not overwrite published successor evidence')
            link(binding['path'],out/relative)
            if digest(out/relative)!=binding['sha256']:
                raise ValueError('Published evidence differs from predecessor')
        # This pointer preserves each ORIGINAL absolute source/input identity.
        # Mutable successor summaries are never linked to predecessor files.
        write(out/'input_profile.json',read(item['profile']['path']))
    write(root/'recovery_publication.json',dict(passed=True,reused_cases=21,new_executions=3,
        scope='Per-case original receipt profiles; unchanged historical outputs. New source only for three missing executions.'))

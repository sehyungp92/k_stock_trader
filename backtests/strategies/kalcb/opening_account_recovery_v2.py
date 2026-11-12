"""Authenticate and reuse the completed control after a validator wiring error.

No execution source, strategy action, receipt or failed evidence is rewritten.
The existing policy-bound first-hour validator retains all account checks and
additionally authenticates the first-hour population and signal lineage.
"""
from pathlib import Path

from .retention_research_io_v1 import read, write, bound, verify, digest, link
from .late_artifact_validation_v2 import validate_native
from .population_research_io_v1 import bridge_accounts
from .research_process_v1 import exact_process


def authenticate(root):
    root = Path(root).resolve()
    protocol = read(root / 'protocol.json')
    if protocol['version'] != 'opening-complete-account-v1':
        raise ValueError('Recovery only supports the registered v1 predecessor')
    for identity in (read(root / 'supervisor_start.json'), read(root / 'status.json')['child']):
        if exact_process(identity):
            raise ValueError('Predecessor still running; do not duplicate it')
    verify(protocol['programs'] + protocol['bindings'] + protocol['database_bindings'])
    error = (root / 'supervisor.err.log').read_text(encoding='utf-8')
    if not error.rstrip().endswith('ValueError: Shortened source audit scope'):
        raise ValueError('Not the diagnosed validator wiring failure')
    if read(root / 'progress.json') or (root / 'completion.json').exists():
        raise ValueError('Unexpected predecessor lifecycle')
    profile_path = root / 'runtime/control/is/profile.json'
    profile = read(profile_path)
    case = root / 'cases/control/is/baseline'
    metrics = validate_native(case, profile, 'baseline', save=False)
    bridge = bridge_accounts(case, Path(protocol['references']['is']['baseline']))
    if not bridge['passed']:
        raise ValueError('Completed predecessor does not exactly reproduce parent')
    done = read(case / 'completion.json')
    before = {x['path']: x['sha256'] for x in done['origins_before']}
    after = {x['path']: x['sha256'] for x in done['origins_after']}
    if done['mode'] != 'control' or any(after.get(path) != sha for path, sha in before.items()):
        raise ValueError('Wrong action or changed runtime imports')
    verify(done['origins_after'] + [done['decisions'], done['contrasts']])
    manifest = read(profile['manifest']['path'])
    code = Path(profile['manifest']['path']).parent / 'files'
    for path, sha in after.items():
        relative = Path(path).resolve().relative_to(code.resolve()).as_posix()
        if manifest['files'].get(relative) != sha:
            raise ValueError('Imported source was not in the frozen manifest')
    audit = read(case / 'execution_audit.json')['audit_receipt']
    files = {str(p.relative_to(case)): bound(p) for p in sorted(case.rglob('*')) if p.is_file()}
    if any(name in files for name in ('independent_account_verification.json', 'activity.json')):
        raise ValueError('Successor must not overwrite linked predecessor summaries')
    bindings = [bound(root / name) for name in ('protocol.json', 'study.json', 'status.json',
        'supervisor_start.json', 'supervisor.err.log', 'supervisor.log', 'progress.json')]
    bindings += [bound(profile_path), *[profile[k] for k in ('manifest', 'plan', 'policy')], *files.values()]
    return {'predecessor': str(root), 'bindings': bindings, 'case_files': files,
        'profile': bound(profile_path), 'case': str(case), 'metrics': metrics, 'bridge': bridge,
        'scope': audit['identity']['configuration']['source_dependency_scope'],
        'first_hour_population': audit['qualification']['first_hour_population'],
        'overlays': {k: v['sha256'] for k, v in protocol['overlays'].items()},
        'repair': 'Use existing late_artifact_validation_v2 instead of the obsolete fixed-45-minute validator. '
                  'All original independent accounting checks and additional first-hour source/lineage checks apply.',
        'economic_changes': False, 'reused_cases': ['control/is/baseline']}


def publish(root, proof):
    """Reuse immutable bytes with their original absolute receipt identities."""
    root = Path(root)
    verify(proof['bindings'])
    profile = root / 'runtime/control/is/profile.json'
    link(proof['profile']['path'], profile)
    for relative, binding in proof['case_files'].items():
        target = root / 'cases/control/is/baseline' / relative
        link(binding['path'], target)
        if digest(target) != binding['sha256']:
            raise ValueError('Copied completed evidence differs from predecessor')
    # All consumers read the original plan/manifest/runtime from this profile.
    # New independent verification and activity files are absent in v1 and
    # written only into the successor; existing linked files stay immutable.
    validate_native(root / 'cases/control/is/baseline', read(profile), 'baseline', save=False)
    write(root / 'recovery_publication.json', {'passed': True, 'reused_cases': proof['reused_cases'],
        'files': len(proof['case_files']), 'original_profile': proof['profile'],
        'scope': 'Original source/input/output receipt identities preserved; no economic rerun or relabeling.'})

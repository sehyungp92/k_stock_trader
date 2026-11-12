"""Import-isolated model/policy producer; stdlib only before frozen bootstrap."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def bootstrap(root,code,check_only=False):
    root=Path(root);code=Path(code).resolve();repo=Path(__file__).resolve().parents[1]
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    for binding in protocol['programs']:
        if digest(binding['path'])!=binding['sha256']:raise ValueError('Changed registered program')
    if not check_only and not any(Path(v['path']).resolve()==Path(__file__).resolve() for v in protocol['programs']):
        raise ValueError('Unregistered producer bootstrap')
    # The working tree is removed before ANY application import. Do not delete
    # already imported modules as a substitute; contamination must fail closed.
    prefixes=('backtests','scripts','kis_core','strategy_kalcb','strategy_common')
    if any(k.split('.')[0] in prefixes for k in sys.modules):raise ValueError('Application imported before frozen bootstrap')
    sys.path[:]=[str(code)]+[p for p in sys.path if p and not Path(p).resolve().is_relative_to(repo)]
    manifest_path=Path(protocol['producer_runtime']['manifest']['path'])
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if digest(manifest_path)!=protocol['producer_runtime']['manifest']['sha256']:raise ValueError('Changed runtime manifest')
    target=next(v for v in protocol['programs'] if Path(v['path']).name=='late_incremental_v1.py')
    name='backtests.strategies.kalcb.late_incremental_v1'
    spec=importlib.util.spec_from_file_location(name,target['path']);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    # Force actual policy dependencies to load before any model or policy output.
    from scripts.kalcb_executed_value_study import config_for
    from backtests.strategies.kalcb.late_policy_v2 import select_executable
    from kis_core.trading_calendar import get_trading_calendar
    calendar=get_trading_calendar()
    calendar.session_hours(__import__('datetime').date(2026,7,10))

    def audit():
        origins={}
        for key,value in list(sys.modules.items()):
            if key.split('.')[0] not in prefixes:continue
            file=getattr(value,'__file__',None)
            if not file:continue
            path=Path(file).resolve()
            if key==name:
                if path!=Path(target['path']).resolve() or digest(path)!=target['sha256']:raise ValueError('Changed signal module')
                origins[key]={'path':str(path),'sha256':digest(path),'registered_new_signal':True};continue
            if not path.is_relative_to(code):raise ValueError('Application import escaped frozen runtime: '+key+' '+str(path))
            rel=path.relative_to(code).as_posix();expected=manifest['files'].get(rel)
            if not expected or digest(path)!=expected:raise ValueError('Unbound runtime module: '+rel)
            origins[key]={'path':str(path),'sha256':expected}
        return {'checked_modules':len(origins),'origins':origins,'runtime_manifest':protocol['producer_runtime']['manifest'],
            'source_generation_hash':protocol['producer_runtime']['source_generation_hash'],'check_only':check_only}
    before=audit()
    if check_only:
        print(json.dumps({'frozen_import_check':True,'modules':before['checked_modules'],
            'calendar':before['origins']['kis_core.trading_calendar']}));return before
    (root/'producer_imports_before.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
    module.produce(root)
    after=audit()
    (root/'producer_imports_after.json').write_text(json.dumps(after,indent=2),encoding='utf-8')
    return after


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--code',type=Path,required=True);p.add_argument('--check-only',action='store_true')
    a=p.parse_args();bootstrap(a.root,a.code,a.check_only)

"""Single policy data producer, not an optimization lifecycle."""
import argparse,importlib.util,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--code',type=Path,required=True);p.add_argument('--module',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.code))
from backtests.strategies.kalcb.retention_research_io_v1 import read,verify,bound
protocol=read(a.root/'protocol.json');verify(protocol['programs'])
expected=[x for x in protocol['programs'] if Path(x['path']).name=='late_policy_v2.py']
if len(expected)!=1 or bound(a.module)['sha256']!=expected[0]['sha256']:raise ValueError('Unregistered policy producer')
spec=importlib.util.spec_from_file_location('late_policy_producer_v2',a.module);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.build_policies(a.root)

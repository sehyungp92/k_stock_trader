"""Single policy data producer, not an optimization lifecycle."""
import argparse,importlib.util,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--code',type=Path,required=True);p.add_argument('--module',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.code))
from backtests.strategies.kalcb.retention_research_io_v1 import read,verify
protocol=read(a.root/'protocol.json');verify(protocol['programs'])
allowed={str(Path(x['path']).resolve()) for x in protocol['programs']}
if str(a.module.resolve()) not in allowed:raise ValueError('Unregistered policy producer')
spec=importlib.util.spec_from_file_location('late_policy_producer_v2',a.module);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.build_policies(a.root)

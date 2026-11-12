from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.round3_monitor_v1 import check
p=argparse.ArgumentParser();p.add_argument('--registration',required=True);a=p.parse_args();r=check(a.registration)
print(json.dumps({k:r[k] for k in ('at_utc','controller_alive','accepted_accounts','process_completed','documentation_completed','errors','prediction_contradictions')}))
print('native_workers='+str([p['pid'] for p in r['native_workers']]))

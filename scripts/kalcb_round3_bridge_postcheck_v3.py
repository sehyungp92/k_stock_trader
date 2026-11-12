"""Fresh read-only verification of preserved v2 exports; not process completion."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.round3_integrity_v1 import reconstruct_journal,exact_paths
from scripts.kalcb_round3_freeze_v1 import read,write,bound
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_round3_20261017/attempt_2')
SOURCE=Path('E:/CodexResearch/k_stock_trader/kalcb_opportunity_lifecycle_20261016/funded_interaction_v1')
p=sys.argv[1];s='baseline' if p=='is' else 'combined'
folder=ROOT/'bridge_v2/accounts'/f'{p}_{s}_observer'
out=ROOT/'bridge_v2'/f'{p}_postcheck_v3.json'
account=reconstruct_journal(folder,1e8)
paths=exact_paths(folder,SOURCE/'cases'/f'combined_{p}_{s}')
write(out,dict(passed=True,program=bound(__file__),verifier=bound(Path(__file__).resolve().parents[1]/'backtests/strategies/kalcb/round3_integrity_v1.py'),
    independent_account=account,paths=paths,native_audit=bound(folder/'execution_audit.json'),
    process_completion=False,repair='Parse equal timezone-aware timestamps rather than comparing ISO T versus space formatting; no economic change'))
print('Fresh independent account and full path verified: '+p)

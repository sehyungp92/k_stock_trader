"""Paired same-process timings for pure helpers, never strategy performance."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from datetime import date,timedelta

v=Path('E:/CodexResearch/k_stock_trader/olr_round1_comprehensive_20261009/full_oos_20260401_20260710/continuous_history_v10/revisions/revision_19_native_overheads')
g=json.loads(Path('C:/Users/sehyu/.codex/research/olr_native_r19/train/inputs/generation.json').read_text())
sys.path.insert(0,g['engine'])
from strategy_common.session_coverage import session_bar_coverage,_calendar_grid
from backtests.auto.shared.cache_keys import stable_signature
def original(relative,label):
    spec=importlib.util.spec_from_file_location(label,Path(g['original_engine'])/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
old=original('strategy_common/session_coverage.py','old_grid')
old_sign=original('backtests/auto/shared/cache_keys.py','old_sign')
rows=json.loads((v/'native_proofs/control_train/replay/equity.json').read_text())
cases=[(date(2025,8,7)+timedelta(days=d),s) for d in range(20)
       for s in ('000100','005930','000660','240810','035420','051910','005380','068270')]
results=[]
for stage in ('calendar','signature'):
    for kind in ('original','optimized','optimized','original'):
        if stage=='calendar':_calendar_grid.cache_clear()
        start=time.perf_counter();cpu=time.process_time()
        if stage=='calendar':
            fn=old.session_bar_coverage if kind=='original' else session_bar_coverage
            values=[fn((),d,5,symbol=s) for d,s in cases]
            result=hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()
        else:
            fn=old_sign.stable_signature if kind=='original' else stable_signature
            result=fn(rows)
        results.append({'stage':stage,'kind':kind,'wall_seconds':time.perf_counter()-start,
                        'cpu_seconds':time.process_time()-cpu,'sha256':result})
for stage in ('calendar','signature'):
    assert len({x['sha256'] for x in results if x['stage']==stage})==1
(v/'paired_helper_benchmarks.json').write_text(json.dumps({'measurements':results,
    'calendar_cases':len(cases),'signature_records':len(rows),'not_end_to_end_throughput':True},indent=2))
print(results,flush=True)

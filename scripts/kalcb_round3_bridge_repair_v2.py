"""Versioned repair: saved mutations plus frozen dataclass defaults, no trading change."""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.kalcb_round3_freeze_v1 import write,bound

worker=(REPO/'scripts/kalcb_round3_native_worker_v1.py').read_text(encoding='utf-8')
anchor="    if effective!=expected:raise ValueError('Evaluator did not receive exact effective configuration')"
replacement="""    from strategy_kalcb.config import KALCBConfig
    expected_effective=neutral(asdict(KALCBConfig.from_mapping(expected)))
    if effective!=expected_effective:raise ValueError('Evaluator did not receive exact effective configuration')
    if any(effective[k]!=v for k,v in expected.items()):raise ValueError('Saved setting override was dropped')"""
assert worker.count(anchor)==1
target=REPO/'scripts/kalcb_round3_native_worker_v2.py'
with target.open('x',encoding='utf-8') as stream:stream.write(worker.replace(anchor,replacement))
preparer=(REPO/'scripts/kalcb_round3_prepare_v1.py').read_text(encoding='utf-8')
preparer=preparer.replace('bridge_v1','bridge_v2').replace('kalcb_round3_native_worker_v1.py','kalcb_round3_native_worker_v2.py')
exec(compile(preparer,__file__,'exec'),{'__name__':'__main__','__file__':__file__})

"""Preserve failed v1; run the prospectively repaired effective-config bridge."""
from pathlib import Path
source=Path(__file__).with_name('kalcb_round3_bridge_runner_v1.py').read_text(encoding='utf-8')
source=source.replace("OUT=ROOT/'bridge_v1'","OUT=ROOT/'bridge_v2'").replace('scripts.kalcb_round3_native_worker_v1','scripts.kalcb_round3_native_worker_v2')
exec(compile(source,__file__,'exec'),{'__name__':'__main__','__file__':__file__})

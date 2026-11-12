"""Preserved v1 commitment-serialization failure; unchanged native preflight."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import round3_native_io_v1 as io
io.NATIVE=io.ROOT/'native_v5'
source=Path(__file__).with_name('kalcb_round3_preflight_v1.py').read_text(encoding='utf-8')
source=source.replace("OUT=ROOT/'preflight_v1'","OUT=ROOT/'preflight_v2'")
exec(compile(source,__file__,'exec'),{'__name__':'__main__','__file__':__file__})

"""Preserve first controller freeze; bind E:-backed diagnostics and strict finer qualification."""
from pathlib import Path
source=Path(__file__).with_name('kalcb_round3_register_run_v1.py').read_text()
source=source.replace("OUT=ROOT/'round_run_v1'", "OUT=ROOT/'round_run_v2'")
source=source.replace('preparation_elapsed=6*60','preparation_elapsed=60*60')
source=source.replace("version='C17-full-native-round-v1'", "version='C17-full-native-round-v2'")
exec(compile(source,__file__,'exec'),{'__name__':'__main__','__file__':__file__})

"""Restart comparator aligns the broker's initial NAV with native equity_records."""
from pathlib import Path
source=Path(__file__).with_name('kalcb_round3_restart_v2.py').read_text()
source=source.replace("exec(compile(source,__file__,'exec')", "source=source.replace('zip(broker.timestamps,broker.equity_curve)', 'zip(broker.timestamps,broker.equity_curve[1:])')\nexec(compile(source,__file__,'exec')")
exec(compile(source,__file__,'exec'),{'__name__':'__main__','__file__':__file__})

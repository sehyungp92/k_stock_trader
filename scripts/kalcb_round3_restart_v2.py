"""Fix restart test import alias: call the installed native driver wrapper."""
from pathlib import Path
source=Path(__file__).with_name('kalcb_round3_restart_v1.py').read_text()
source=source.replace('from backtests.engine.replay import drive_timestamp_batch','from backtests.engine import replay')
source=source.replace('in drive_timestamp_batch(broker,tuple(batch),(adapter,))','in replay.drive_timestamp_batch(broker,tuple(batch),(adapter,))')
exec(compile(source,__file__,'exec'),{'__name__':'__main__','__file__':__file__})

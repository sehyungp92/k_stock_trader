"""Translate the historical source profile into the strict account profile schema."""
from pathlib import Path
source=Path(__file__).with_name('kalcb_round3_reuse_v1.py').read_text()
source=source.replace("OUT=ROOT/'reused_controls_v1'", "OUT=ROOT/'reused_controls_v2'")
source=source.replace('        checked=validate_case(folder,profile,s)', '''        from backtests.strategies.kalcb.round3_native_io_v1 import profile_for
        for field in ('manifest','plan','policy'):
            assert bound(profile[field]['path'])['sha256']==profile[field]['sha256']
        profile=profile_for({**profile,'initial':100000000},folder,s)
        checked=validate_case(folder,profile,s)''')
exec(compile(source,__file__,'exec'),{'__name__':'__main__','__file__':__file__})

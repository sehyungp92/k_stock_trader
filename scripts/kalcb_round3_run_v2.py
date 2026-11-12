"""Use the canonical round path with its large artifacts stored on E:."""
from pathlib import Path
source=Path(__file__).with_name('kalcb_round3_run_v1.py').read_text()
source=source.replace('import argparse,json,os,sys,threading,time,traceback','import argparse,json,os,sys,threading,time,traceback,subprocess')
source=source.replace('    canonical.mkdir(parents=True,exist_ok=False)', '''    artifact_root=out/'round_artifacts';artifact_root.mkdir(exist_ok=False)
    def quote_ps(value):return "'"+str(value).replace("'","''")+"'"
    command='New-Item -ItemType Junction -Path '+quote_ps(canonical)+' -Value '+quote_ps(artifact_root)+' | Out-Null'
    subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    if canonical.resolve()!=artifact_root.resolve():raise ValueError('Canonical artifact junction does not resolve to this attempt')''')
exec(compile(source,__file__,'exec'),{'__name__':'__main__','__file__':__file__})

"""Verify actual IS-to-OOS warm-up observations, not just file availability."""
from __future__ import annotations
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import olr_round1_optimization_v1 as study
n=study.n
OUT=study.OUT/'history_check'


def history_digest(path, boundary):
    h=hashlib.sha256();count=0;days=set()
    with sqlite3.connect('file:'+path.as_posix()+'?mode=ro',uri=True) as c:
        for row in c.execute('SELECT day,symbol,ts,o,h,l,c,v FROM bars WHERE day<? ORDER BY day,symbol,ts',(boundary,)):
            h.update((json.dumps(row,separators=(',',':'))+'\n').encode())
            count+=1;days.add(row[0])
    return {'rows':count,'sessions':len(days),'first':min(days),'last':max(days),'ordered_observation_sha256':h.hexdigest(),'dates':sorted(days)}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    infos={w:n.read(n.generation_folder(study.PARENT,w)/'ready.json') for w in ('train','oos')}
    profiles={}
    for w,info in infos.items():
        files=Path(info['files']);database=files/info['study']['database']
        profiles[w]=history_digest(database,'2026-04-01')
        assert n.sha(database)==info['study']['database_hash']
        profiles[w]['database_sha256']=info['study']['database_hash']
        print({'window':w,'prior_rows':profiles[w]['rows'],'sessions':profiles[w]['sessions']},flush=True)
    assert profiles['train']['ordered_observation_sha256']==profiles['oos']['ordered_observation_sha256']
    assert profiles['train']['rows']==profiles['oos']['rows']
    n.write(OUT/'observation_continuity.json',{'status':'passed','asof_utc':datetime.now(timezone.utc).isoformat(),
        'profiles':profiles,'all_pre_april1_observations_identical':True,
        'is_sessions_carried':sum('2025-08-07'<=d<='2026-03-31' for d in profiles['oos']['dates']),
        'august_is_boundary_is_not_feature_reset':True,'is_prior_intraday_sessions':sum(d<'2025-08-07' for d in profiles['train']['dates'])})
    # Reuse the independent raw-parquet reference verifier in a new evidence
    # directory; never overwrite the previous review's receipts.
    n.write(OUT/'generations_v10/oos/ready.json',infos['oos'])
    n.write(OUT/'protocol.json',n.read(study.ROOT/'protocol.json'))
    command=[sys.executable,'-B','-u',str(REPO/'scripts/olr_continuous_history_verification.py'),str(OUT)]
    subprocess.run(command,cwd=REPO,check=True)
    verified=n.read(OUT/'continuous_history_verification.json')
    assert verified['status']=='passed' and not verified['future_observations_changed_previous_features']
    n.write(OUT/'completion.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),
        'observation_continuity_sha256':n.sha(OUT/'observation_continuity.json'),
        'rolling_feature_verification_sha256':n.sha(OUT/'continuous_history_verification.json'),
        'status':'passed','remaining_feature_issue':'Current-afternoon panel requires the separately registered continuous-panel feature treatment; original prior-session warm-up is already continuous.'})


if __name__=='__main__':main()

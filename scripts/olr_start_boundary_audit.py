"""Audit historical readiness without changing the active research generation.

Warm-history builds here are causal feature diagnostics, never performance
results. Any changed selections require a separately registered audited replay.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')
    temp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def event(out, stage, **values):
    row={'utc':datetime.now(timezone.utc).isoformat(),'stage':stage,**values}
    write(out/'status.json',row); print(json.dumps(row),flush=True)


def economic(rows):
    return {r['trade_date']:[{k:v for k,v in c.items()
        if k not in ('metadata','source_fingerprint','source_artifact_hash')} for c in r['candidates']] for r in rows}


def compare(left,right):
    a,b=economic(left),economic(right)
    dates=sorted(set(a)|set(b))
    changes=[]
    for day in dates:
        if a.get(day)!=b.get(day):
            la,lb=a.get(day,[]),b.get(day,[])
            changes.append({'day':day,'before':la,'after':lb,
                'added_symbols':sorted({c['symbol'] for c in lb}-{c['symbol'] for c in la}),
                'removed_symbols':sorted({c['symbol'] for c in la}-{c['symbol'] for c in lb})})
    return {'changed_candidate_economics_dates':len(changes),'changes':changes,
        'before_candidate_rows':sum(map(len,a.values())),'after_candidate_rows':sum(map(len,b.values())),
        'warning':'Feature diagnostic only; selection changes are not audited return improvements.'}


def audit(root):
    protocol=read(root/'protocol.json')
    out=root/'start_boundary_audit'
    out.mkdir(parents=True,exist_ok=True)
    infos={w:read(root/protocol['generation_directories'][w]/w/'ready.json') for w in ('train','oos')}
    oos=infos['oos']; engine=Path(oos['files'])
    sys.path.insert(0,str(engine));os.chdir(engine)
    import yaml
    from backtests.core.input_manifest import InputManifest
    from backtests.strategies.olr import research_sweep as research
    from strategy_olr.research import _prior_rows, _row_date
    from scripts.portfolio_forward_olr import build

    profile={}
    for window,info in infos.items():
        event(out,'profiling_history',window=window)
        files=Path(info['files']);study=info['study']
        snapshots=read(files/study['olr_snapshots'])
        first,last=map(date.fromisoformat,protocol['windows'][window])
        scoped=[s for s in snapshots if str(first)<=s['trade_date']<=str(last)]
        manifest=InputManifest(study['manifest'])
        contract=info['manifest']['evidence']['shared_market_inputs']
        with manifest.activate():
            symbols=[str(s).zfill(6) for s in yaml.safe_load(manifest.read(files/contract['universe']))['symbols']]
            data_root=files/contract['daily_data_root']
            tables={'daily':research._load_daily_rows(data_root,symbols,last),
                'flow':research._load_flow_rows(data_root,symbols,last),
                'foreign':research._load_foreign_flow_rows(data_root,symbols,last),
                'institutional':research._load_institutional_flow_rows(data_root,symbols,last),
                'index':research._load_index_rows(data_root,last)}
        history={}
        for kind,table in tables.items():
            rows=[]
            for symbol,values in table.items():
                prior=_prior_rows(values,first)
                rows.append({'symbol':symbol,'prior_count':len(prior),
                    'first':str(_row_date(prior[0])) if prior else None,
                    'last':str(_row_date(prior[-1])) if prior else None})
            history[kind]={'symbols':rows,'min_prior_count':min(r['prior_count'] for r in rows),
                'max_prior_count':max(r['prior_count'] for r in rows)}
        reasons=Counter();excluded=Counter();months=Counter()
        for snapshot in scoped:
            m=snapshot['metadata'];month=snapshot['trade_date'][:7]
            excluded[month]+=len(m.get('feature_prefix_exclusions',[]));months[month]+=len(snapshot['candidates'])
            for stage in ('build_rejected_symbols','rejected_symbols','afternoon_rejected_symbols'):
                for rs in (m.get(stage) or {}).values():reasons.update(rs if isinstance(rs,list) else [rs])
        database=files/study['database']
        with sqlite3.connect('file:'+database.as_posix()+'?mode=ro',uri=True) as c:
            days=[r[0] for r in c.execute('SELECT DISTINCT day FROM bars ORDER BY day')]
        pm=read(files/study['kalcb_policies']/'policies_manifest.json')
        profile[window]={'window':[str(first),str(last)],'daily_and_flow_history':history,
            'database_first':days[0],'database_last':days[-1],
            'prior_intraday_sessions':sum(d<str(first) for d in days),
            'policy_evaluation_from':pm['evaluation_from'],'policy_evaluation_through':pm['evaluation_through'],
            'snapshot_first':snapshots[0]['trade_date'],'snapshot_last':snapshots[-1]['trade_date'],
            'prescore_candidate_rows':sum(len(s['candidates']) for s in snapshots if s['trade_date']<str(first)),
            'entry_snapshot_sessions':len(scoped),'candidate_rows_by_month':dict(months),
            'prefix_exclusions_by_month':dict(excluded),'rejection_reasons':dict(reasons),
            'daily_min_history_gate':60,'insufficient_history_rejections':reasons['insufficient_history']}
    write(out/'history_profile.json',profile)

    event(out,'assembling_diagnostic_history')
    original=Path(oos['files'])/oos['study']['database']
    train=Path(infos['train']['files'])/infos['train']['study']['database']
    diagnostic=out/'history_extended.sqlite'
    if diagnostic.exists():
        raise ValueError('Preserve existing diagnostic generation; do not overwrite it')
    shutil.copy2(original,diagnostic)
    with sqlite3.connect(str(diagnostic),uri=True) as c:
        c.execute('ATTACH DATABASE ? AS historical',('file:'+train.as_posix()+'?mode=ro',))
        overlap=c.execute('SELECT COUNT(*) FROM bars a JOIN historical.bars b ON a.symbol=b.symbol AND a.ts=b.ts WHERE a.day<?',('2026-04-01',)).fetchone()[0]
        mismatch=c.execute('SELECT COUNT(*) FROM bars a JOIN historical.bars b ON a.symbol=b.symbol AND a.ts=b.ts WHERE a.day<? AND (a.o!=b.o OR a.h!=b.h OR a.l!=b.l OR a.c!=b.c OR a.v!=b.v)',('2026-04-01',)).fetchone()[0]
        if mismatch:raise ValueError('Historical overlap observations differ; source repair required')
        prior=[r[0] for r in c.execute('SELECT DISTINCT day FROM historical.bars WHERE day<? ORDER BY day DESC LIMIT 61',('2026-04-01',))][::-1]
        # The two audited DBs use different physical column orders. Match names,
        # and insert only dates strictly before the original DB's first session.
        earliest=c.execute('SELECT MIN(day) FROM bars').fetchone()[0]
        c.execute('CREATE UNIQUE INDEX boundary_bars_symbol_ts ON bars(symbol,ts)')
        before=c.total_changes
        c.execute('INSERT INTO bars (day,symbol,ts,o,h,l,c,v) SELECT h.day,h.symbol,h.ts,h.o,h.h,h.l,h.c,h.v FROM historical.bars h WHERE h.day>=? AND h.day<?',(prior[0],earliest))
        inserted=c.total_changes-before
        if c.execute('SELECT COUNT(*) FROM bars WHERE day!=substr(ts,1,10) OR length(symbol)!=6').fetchone()[0]:
            raise ValueError('Diagnostic merge changed date/symbol identity')
        c.commit()
    write(out/'history_assembly.json',{'train_database':str(train),'train_hash':sha(train),
        'oos_database':str(original),'oos_hash':sha(original),'diagnostic_database':str(diagnostic),
        'diagnostic_hash':sha(diagnostic),'overlap_bar_count':overlap,'overlap_mismatches':mismatch,
        'inserted_prior_bars':inserted,'prior_days':prior,
        'restriction':'All added observations precede April1; original OOS bars untouched. Diagnostic is not registered for native performance.'})
    baseline=read(engine/oos['study']['olr_snapshots'])
    requested=[date.fromisoformat(s['trade_date']) for s in baseline]
    request=read(root/'registrations/oos/ce14d8377b9e67ac6_100000000/candidate.json')
    original_request=read(engine/oos['study']['olr_request'])
    original_request['mutations']={k:v for k,v in request['mutations'].items() if not k.startswith('capacity_')}
    results={}
    previous=baseline
    for count in (21,61):
        event(out,'building_history_sensitivity',prior_sessions=count)
        days=[date.fromisoformat(d) for d in prior[-count:]]+requested
        built=build(oos['study']['manifest'],diagnostic,original_request,days)
        rows=[built[d].to_json_dict() for d in requested]
        write(out/f'prior_{count}_snapshots.json',rows)
        results[str(count)]={'against_original':compare(baseline,rows),'against_previous_history':compare(previous,rows)}
        write(out/'sensitivity.json',results)
        previous=rows
    write(out/'completion.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),
        'performance_established':False,'results':{k:{'changed_dates':v['against_original']['changed_candidate_economics_dates'],
        'changed_dates_vs_previous':v['against_previous_history']['changed_candidate_economics_dates']} for k,v in results.items()}})
    event(out,'history_sensitivity_complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();audit(args.root)

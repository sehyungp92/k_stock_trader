"""0035 bounded funded diagnostic, using a private frozen research generation."""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
from datetime import datetime,timedelta
import gzip
import json
import math
import os
from pathlib import Path
import sqlite3
from statistics import median
import subprocess
import sys
import traceback
import psutil
import yaml
import olr_selective_half_continuation_v1 as shared

REPO=shared.REPO
PARENT=shared.PARENT
ROOT=Path('E:/CodexResearch/k_stock_trader/olr_continuing_research_20261016/relative_context_v2')
REUSED_INPUTS=ROOT.parent/'relative_context_v1'
CASES=[('control',0.),('half',.5),('full',1.)]
read,sha,write,status,now,sig=shared.read,shared.sha,shared.write,shared.status,shared.now,shared.sig


def private_text(files,name,content,manifest):
    path=files/name
    assert path.resolve().is_relative_to(files.resolve())
    if path.exists():path.unlink()  # Break this new generation's hard link first.
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:f.write(content)
    compile(content,str(path),'exec')
    manifest['files'][name]=sha(path)


def features(files,study):
    universe=[str(s).zfill(6) for s in yaml.safe_load((files/'config/universe_103.yaml').read_text(encoding='utf8'))['symbols']]
    assert len(universe)==103
    con=sqlite3.connect((files/study['database']).as_uri()+'?mode=ro',uri=True)
    con.execute('PRAGMA query_only=ON')
    window=study['configuration']['date_range']
    days=[r[0] for r in con.execute('SELECT DISTINCT day FROM bars WHERE day>=? AND day<=? ORDER BY day',
        (window['start'],window['end']))]
    # Holding runoff is part of the policy even after candidate snapshots end.
    assert days[0]==window['start'] and days[-1]==window['end'] and len(days)==156
    source_rows=defaultdict(dict)
    for symbol in universe:
        rows=con.execute("SELECT day,ts,o,c,v FROM bars WHERE symbol=? AND ts>=? AND ts<=? AND substr(ts,12,5) BETWEEN '09:00' AND '09:25' ORDER BY ts",
            (symbol,window['start']+'T00:00:00+09:00',window['end']+'T23:59:00+09:00')).fetchall()
        for day,stamp,opening,close,volume in rows:
            source_rows[day].setdefault(symbol,[]).append((stamp,opening,close,volume))
    values={};supported=defaultdict(int)
    for day in days:
        opening=datetime.fromisoformat(day+'T09:00:00+09:00')
        labels=[(opening+timedelta(minutes=5*i)).isoformat() for i in range(6)]
        own={}
        for symbol in universe:
            rows=source_rows[day].get(symbol,[])
            if [r[0] for r in rows]!=labels or sum(r[3] for r in rows)<=0:
                supported['unsupported_own_prefix']+=1;continue
            assert rows[0][1]>0 and rows[-1][2]>0
            own[symbol]=dict(opening=rows[0][1],close=rows[-1][2],own_return=rows[-1][2]/rows[0][1]-1,
                source_last_label=labels[-1],available_at=(opening+timedelta(minutes=30)).isoformat(),
                observed_volume=sum(r[3] for r in rows))
        rows={}
        for symbol,value in own.items():
            peers=[v['own_return'] for s,v in own.items() if s!=symbol]
            if len(peers)<68:
                supported['unsupported_peer_coverage']+=1;continue
            rows[symbol]=dict(**value,peer_count=len(peers),market_median=median(peers))
            supported['supported']+=1
        values[day+'T09:30:00+09:00']=rows
    con.close()
    assert supported['supported']>0
    # Cross-check the independently extracted, already accepted 0034 contexts.
    review=read(ROOT.parent/'context_population_v1/native_coverage.json');checks=0
    for row in review:
        if row['clock']!='09:30' or not row['context']['available']:continue
        c=row['context'];f=values[row['decision_at']].get(row['symbol'])
        if c['market_peers']<68:continue
        assert f is not None
        assert all(math.isclose(f[x],c[y],abs_tol=1e-12) for x,y in [('own_return','own_return'),('market_median','market_peer_median'),('close','close')])
        checks+=1
    assert checks>0
    return dict(version='relative-context-v1',universe=universe,rows=values,coverage=dict(supported),
        evaluation_days=days,independent_context_checks=checks,
        source_database_sha256=sha(files/study['database']),universe_sha256=sha(files/'config/universe_103.yaml'),
        label_scope='Completed 09:00-09:25 prefixes only; no held-state or outcome selection')


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    old=read(PARENT/'native_engines/train/ready.json');src=Path(old['files'])
    files=root/'native_engines/train/files';files.mkdir(parents=True)
    manifest=copy.deepcopy(old['manifest']);manifest.pop('manifest_hash',None)
    cached=read(REUSED_INPUTS/'native_engines/train/ready.json')
    cached_receipt=read(REUSED_INPUTS/'freeze_receipt.json')
    assert cached_receipt['parent_ready_sha256']==sha(PARENT/'native_engines/train/ready.json')
    for i,(name,digest) in enumerate(manifest['files'].items()):
        # Reuse v1's completed authentication of unchanged immutable inputs.
        # Re-read the small original sources altered in that private generation.
        source=Path(cached['files'])/name if cached['manifest']['files'].get(name)==digest else src/name
        if cached['manifest']['files'].get(name)!=digest:assert sha(source)==digest,name
        path=files/name;path.parent.mkdir(parents=True,exist_ok=True);os.link(source,path)
        if i%2000==0:status(root,'linking_authenticated_inputs',files=i)
    source_hash,completed_case,_=shared.frozen_imports(root)
    parent_hash=source_hash(src);assert source_hash(files)==parent_hash==old['study']['source_generation_hash']
    spec=REPO/'docs/optimisation/olr/experiments/olr-exp-0035.md'
    write(root/'protocol.json',dict(at=now(),experiment='olr-exp-0035',design=str(spec),design_sha256=sha(spec),
        runner_sha256=sha(__file__),parent_source=parent_hash,cases=CASES,concurrency=1,
        planned_wall_minutes=90,new_data=False,classification='bounded funded diagnostic',
        advancement='Economic and causal assessment required; no automatic further cases'))
    private_text(files,'strategy_olr/core/relative_context.py',(REPO/'scripts/olr_relative_context_rule_v1.py').read_text(encoding='utf8'),manifest)
    private_text(files,'tests/strategy_olr/test_relative_context.py',(REPO/'scripts/olr_relative_context_tests_v1.py').read_text(encoding='utf8'),manifest)
    name='strategy_olr/core/logic.py';text=(files/name).read_text(encoding='utf8')
    text=text.replace("role.get('reason') == 'weak_followthrough_partial'", "role.get('reason') in {'weak_followthrough_partial','relative_context_reduce'}")
    old_anchor='    if config.capacity_weak_partial_fraction:\n        from .weak_partial import maybe_reduce\n'
    assert text.count(old_anchor)==1
    text=text.replace(old_anchor,"    if state.meta.get('relative_context_v1'):\n        from .relative_context import maybe_reduce as context_reduce\n        reduced=context_reduce(state, symbol_state, position, bar, config, exit_plan, risk)\n        if reduced is not None:return reduced\n"+old_anchor)
    private_text(files,name,text,manifest)
    name='backtests/strategies/olr/runner.py';text=(files/name).read_text(encoding='utf8')
    anchor='        result = step_olr_batch(self.state, selected, self.config, snapshot, self._portfolio_view(broker),\n'
    assert text.count(anchor)==1
    text=text.replace(anchor,"        relative=getattr(self,'relative_context_v1',None)\n        if relative is not None:\n            available=(stamp+timedelta(minutes=5)).isoformat()\n            self.state.meta['relative_context_v1']=dict(fraction=relative['fraction'],\n                feature_sha256=relative['feature_sha256'],rows=relative['features']['rows'].get(available,{}))\n"+anchor)
    private_text(files,name,text,manifest)
    name='scripts/portfolio_synergy_study.py';text=(files/name).read_text(encoding='utf8')
    anchor='    if args.standalone:\n        adapters = tuple(a for a in adapters if a.strategy_id == args.standalone)\n'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+"    if study.get('relative_context_settings'):\n        settings=bound('relative_context_settings')\n        if settings['fraction'] not in (0.,.5,1.):raise ValueError('Unregistered contextual action')\n        context_features=bound('relative_context_features')\n        if settings['fraction']:\n            next(a for a in adapters if a.strategy_id=='OLR').relative_context_v1=dict(\n                fraction=settings['fraction'],features=context_features,feature_sha256=digest(generation.root/study['relative_context_features']))\n")
    private_text(files,name,text,manifest)
    study=copy.deepcopy(old['study']);study['code']=str(files)
    for key in ('economic_policy_settings','drawdown_reserve_contract','structural_entry_settings'):study.pop(key,None)
    f=features(files,study);feature_name='research/relative_context/features.json'
    write(files/feature_name,f);manifest['files'][feature_name]=sha(files/feature_name)
    study['relative_context_features']=feature_name
    generation=source_hash(files);study['source_generation_hash']=generation;manifest['source_generation_hash']=generation
    write(root/'native_engines/train/ready.json',dict(files=str(files),study=study,manifest=manifest,interval_qualification=old['interval_qualification']))
    write(root/'freeze_receipt.json',dict(at=now(),parent_source=parent_hash,source=generation,
        changes={n:h for n,h in manifest['files'].items() if h!=old['manifest']['files'].get(n)},
        feature_coverage=f['coverage'],feature_evaluation_days=len(f['evaluation_days']),independent_context_checks=f['independent_context_checks'],
        features_sha256=sha(files/feature_name),parent_ready_sha256=sha(PARENT/'native_engines/train/ready.json'),
        reused_authentication=dict(root=str(REUSED_INPUTS),freeze_receipt_sha256=sha(REUSED_INPUTS/'freeze_receipt.json'),
            scope='Unchanged immutable inputs authenticated in v1; altered sources restored from original parent; full source hash rechecked.')))
    tests=['tests/strategy_olr/test_relative_context.py','tests/strategy_olr/test_precision_v3.py',
        'tests/strategy_olr/test_opening_risk_v2.py','tests/strategy_olr/test_structural_service.py','tests/strategy_olr/test_economic_evidence.py']
    report=root/'focused_tests.xml'
    cmd=[sys.executable,'-B','-m','pytest',*tests,'-q','-p','no:cacheprovider','--basetemp='+str(root/'tmp/pytest'),'--junitxml='+str(report)]
    result=subprocess.run(cmd,cwd=files,env=shared.env_for(root),capture_output=True,text=True)
    (root/'focused_tests.log').write_text(result.stdout+result.stderr,encoding='utf8')
    if result.returncode:raise RuntimeError('Frozen boundary tests failed: '+str(root/'focused_tests.log'))
    import xml.etree.ElementTree as ET
    suite=ET.parse(report).getroot().find('testsuite');assert all(suite.get(k)=='0' for k in ('failures','errors','skipped'))
    assert source_hash(files)==generation and source_hash(src)==parent_hash
    assert completed_case(PARENT/'cases/train/native_control')['valid']
    write(root/'preflight.json',dict(at=now(),source=generation,parent_source=parent_hash,tests_passed=True,
        tests=int(suite.get('tests')),test_report=str(report),test_report_sha256=sha(report),
        protocol_sha256=sha(root/'protocol.json'),runner_sha256=sha(__file__),features_sha256=sha(files/feature_name)))
    status(root,'ready_for_disabled_parity',tests=int(suite.get('tests')),queued=[x[0] for x in CASES])


def launch(root,tag,fraction):
    ready=read(root/'native_engines/train/ready.json');files=Path(ready['files'])
    s=copy.deepcopy(ready['study']);m=copy.deepcopy(ready['manifest']);out=root/'cases/train'/tag
    if out.exists():raise FileExistsError(str(out))
    prefix='research/relative_context/'+tag+'/'
    values=[('structural_settings','structural.json',dict(mode='control',interval_service=True)),
        ('funded_probe_settings','probe.json',dict(version='first-native-fade-funded-probe-v1',action='hold',control_reference=None)),
        ('funded_population_settings','population.json',dict(version='monthly-native-funded-population-v1',cell=None,catalog_reference=None)),
        ('relative_context_settings','context.json',dict(fraction=fraction,version='relative-context-v1'))]
    for key,name,value in values:
        s[key]=prefix+name;write(files/s[key],value);m['files'][s[key]]=sha(files/s[key])
    s['candidates']['candidate']['olr_capacity'].update(capacity_structural_partial=False,capacity_weak_partial_fraction=0.)
    m['evidence']['qualification_path']=ready['interval_qualification']
    s['manifest']=str(files.parent/(tag+'.manifest.json'));s['configuration']['input_manifest_path']=s['manifest']
    s['spec_path']=prefix+'spec.json'
    write(files/s['spec_path'],{k:v for k,v in s.items() if k not in ('code','manifest','spec_path','source_generation_hash')})
    m['files'][s['spec_path']]=sha(files/s['spec_path']);write(s['manifest'],dict(**m,manifest_hash=sig(m)))
    sp=root/'runtime/train'/tag/'study.json';write(sp,s)
    cmd=[sys.executable,'-B','-u',str(files/'scripts/portfolio_synergy_study.py'),'replay','--study',str(sp),
        '--recipe','candidate','--scenario','baseline','--standalone','OLR','--output',str(out),'--full-audit']
    log=root/'logs'/(tag+'.log');log.parent.mkdir(exist_ok=True)
    with log.open('x',encoding='utf8') as f:
        p=subprocess.Popen(cmd,cwd=files,env=shared.env_for(root),stdout=f,stderr=subprocess.STDOUT,creationflags=0x08000000)
    value=dict(pid=p.pid,created_at=psutil.Process(p.pid).create_time(),command=cmd,log=str(log),case=str(out),at=now())
    write(sp.parent/'process.json',value)
    return p,value


def run(root):
    source_hash,completed_case,contrast=shared.frozen_imports(root)
    pre=read(root/'preflight.json');ready=read(root/'native_engines/train/ready.json')
    assert pre['tests_passed'] and sha(__file__)==pre['runner_sha256']
    assert sha(root/'protocol.json')==pre['protocol_sha256'] and sha(pre['test_report'])==pre['test_report_sha256']
    assert source_hash(Path(ready['files']))==pre['source']
    summaries={}
    for i,(tag,fraction) in enumerate(CASES):
        status(root,'running',case=tag,accepted=list(summaries),queued=[x[0] for x in CASES[i+1:]])
        p,identity=launch(root,tag,fraction)
        if p.wait():raise RuntimeError('Replay failed: '+identity['log'])
        case=Path(identity['case']);valid=completed_case(case)
        assert valid['valid'] and read(case/'funded_checkpoint_receipt.json')['no_action_continuation_equal']
        assert read(case/'funded_population_receipt.json')['audit']['pass_']
        if tag=='control':
            parent=PARENT/'cases/train/native_control'
            checks={name:read(case/name)==read(parent/name) for name in ['equity.json','daily.json','trades.json','minute_nav_evidence.json']}
            write(root/'disabled_parity.json',dict(at=now(),checks=checks,pass_=all(checks.values())))
            if not all(checks.values()):raise ValueError('Disabled economic parity failed; no treatment launch')
        events=[]
        with gzip.open(case/'native_decisions.jsonl.gz','rt',encoding='utf8') as stream:
            for line in stream:
                row=json.loads(line)
                if row.get('reason') in ('relative_context_reduce','relative_context_exit'):events.append(row)
        if fraction and not events:raise ValueError('No treatment activation')
        summaries[tag]=shared.economic_summary(case)
        summaries[tag]['context_decisions']=events
        write(root/'assessments'/(tag+'.json'),summaries[tag])
        status(root,'case_accepted',case=tag,accepted=list(summaries),queued=[x[0] for x in CASES[i+1:]])
    comparisons={}
    for left,right in [('control','half'),('control','full'),('half','full')]:
        a,b=summaries[left],summaries[right]
        accounting=contrast(read(Path(a['case'])/'trades.json'),read(Path(b['case'])/'trades.json'))
        months=defaultdict(float);dates=defaultdict(float)
        for row in accounting['per_opportunity']:months[row['day'][:7]]+=row['delta_pnl'];dates[row['day']]+=row['delta_pnl']
        comparisons[left+'_to_'+right]=dict(delta={k:b[k]-a[k] for k in ['return_pct','full_dd_pct','common_dd_pct',
            'maximum_currency_decline','meaningful','meaningful_days','buy_notional','winning_wealth','losing_wealth',
            'longest_underwater_calendar_days','mean_economic_event_exposure']},accounting=accounting,
            entry_month_attribution=dict(months),entry_date_attribution=dict(dates),
            excluding_best_entry_date=sum(dates.values())-max(dates.values(),default=0))
    write(root/'assessment.json',dict(at=now(),summaries=summaries,comparisons=comparisons,economic_objective_achieved=False,
        next='Review intervention, recoveries, funding and admission. No automatic qualification or successor variant.'))
    status(root,'awaiting_economic_and_causal_assessment',accepted=list(summaries),queued=[],
        next='0035 requires substantive interpretation; diagnostic completion is not assignment closure.')


def start(root):
    pre=read(root/'preflight.json');assert pre['tests_passed'] and sha(__file__)==pre['runner_sha256']
    if (root/'controller_process.json').exists():raise FileExistsError('Inspect existing worker, do not duplicate')
    log=root/'controller.log';cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'run','--root',str(root)]
    with log.open('x',encoding='utf8') as f:
        p=subprocess.Popen(cmd,cwd=REPO,env=shared.env_for(root),stdout=f,stderr=subprocess.STDOUT,creationflags=0x08000000)
    value=dict(pid=p.pid,created_at=psutil.Process(p.pid).create_time(),command=cmd,log=str(log),at=now())
    write(root/'controller_process.json',value);print(json.dumps(value))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','start','run']);parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    try:globals()[args.phase](args.root)
    except BaseException as exc:
        if args.root.exists():status(args.root,'needs_repair',error=repr(exc),traceback=traceback.format_exc())
        raise

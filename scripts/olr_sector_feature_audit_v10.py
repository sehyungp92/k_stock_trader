"""Source-only causal sector feature audit; no decisions or payoff labels."""
from collections import defaultdict
from dataclasses import asdict
from datetime import date,datetime,time,timedelta,timezone
from pathlib import Path
from statistics import fmean
import argparse
import difflib
import hashlib
import importlib.util
import json
import math
import os
import pickle
import sys
import tempfile

REPO=Path(__file__).resolve().parents[1]
BASE=Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010')
ROOT=BASE/'structural_v6/sector_feature_audit_v1'
OUT=REPO/'data/backtests/analysis/olr_structural_20261012/sector_feature_audit_v1'
INVENTORY=BASE/'structural_v6/opportunity_v1'
NAMES=['scripts/olr_sector_feature_audit_v10.py','tests/strategy_olr/test_sector_feature_audit_v10.py',
       'scripts/olr_opportunity_clock_v7.py','scripts/olr_structural_native_test_isolated_v6.py']


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def signature(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(obj,stream,sort_keys=True,indent=2,default=str,allow_nan=False);stream.write('\n')
def utc():return datetime.now(timezone.utc).isoformat()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def near(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12),(a,b)


def qualify_day(bars,day,clock):
    from strategy_common.clock import KST
    from strategy_common.execution_session import session_cutoff
    cutoff=datetime.combine(day,session_cutoff(day,time(14,30),anchor='close'),KST)
    opening=datetime.combine(day,session_cutoff(day,time(9),anchor='open'),KST)
    prefix={key:tuple(b for b in values if b.is_completed and opening<=b.timestamp and b.timestamp.date()==day and b.timestamp+timedelta(minutes=5)<=cutoff)
            for key,values in bars.items() if key[0]==day}
    qualified,excluded=clock.close_anchored_feature_bars(prefix,time(14,30))
    return qualified,excluded,cutoff


def causal_members(bars,sectors,histories,day):
    from strategy_common.sector_intraday import member_from_bars,score_sector_members,cutoff_label_for
    members=[];details={}
    for symbol in sorted(sectors):
        current=bars.get((day,symbol),())
        if not current:continue
        volume=math.fsum(max(float(b.volume),0) for b in current)
        history=histories.get(symbol,[])[-20:]
        assert all(prior_day<day for prior_day,_ in history)
        baseline=math.fsum(v for _,v in history)/len(history) if history else None
        ratio=volume/max(baseline,1.) if baseline is not None else None
        native_ratio=ratio if ratio is not None else 1.
        member=member_from_bars(symbol,sectors[symbol],day,current,rel_volume=native_ratio)
        if member is None:continue
        members.append(member)
        details[symbol]=dict(volume=volume,bar_count=len(current),history=[[str(d),v] for d,v in history],
            prior_count=len(history),prior_mean=baseline,relative_volume=ratio,
            native_missing_baseline_default=native_ratio if ratio is None else None,
            oldest_age_days=(day-history[0][0]).days if history else None,
            latest_age_days=(day-history[-1][0]).days if history else None,ret=member.ret)
    panel=score_sector_members(members,trade_date=day,cutoff_label=cutoff_label_for(time(14,30)))
    return members,details,panel


def peer_expectation(members,details,symbol):
    target=next(m for m in members if m.symbol==symbol)
    peers=[m for m in members if m.symbol!=symbol and m.sector==target.sector]
    market=[m for m in members if m.symbol!=symbol]
    weight=1. if len(peers)>=3 else len(peers)/(len(peers)+3.)
    avg=lambda group,field,default:fmean(getattr(m,field) for m in group) if group else default
    rel=weight*avg(peers,'rel_volume',1.)+(1-weight)*avg(market,'rel_volume',1.)
    participation=lambda group:sum(m.ret>0 and m.rel_volume>=1 for m in group)/len(group) if group else 0.
    part=weight*participation(peers)+(1-weight)*participation(market)
    contributors=peers if weight==1 else market
    supported=bool(contributors) and all(details[m.symbol]['prior_count']>0 for m in contributors)
    return dict(relative_volume=rel,participation=part,effective_count=len(peers),weight=weight,
        supported=supported,qualified_relative_volume=rel if supported else None,
        contributing_peers=[m.symbol for m in contributors])


def update_history(histories,members,day):
    for member in members:
        if member.volume>0:
            history=histories.setdefault(member.symbol,[])
            assert not history or history[-1][0]<day
            history.append((day,member.volume));del history[:-20]


def imports(guard):
    result={}
    for name,mod in list(sys.modules.items()):
        if name.split('.')[0] not in guard.roots or not getattr(mod,'__file__',None):continue
        path=Path(mod.__file__).resolve();assert path.is_relative_to(guard.files),name
        relative=path.relative_to(guard.files).as_posix();assert guard.manifest.get(relative)==sha(path)
        result[relative]=sha(path)
    return result


def setup():
    root_temp=ROOT/'runtime_tmp';root_temp.mkdir(parents=True,exist_ok=True)
    os.environ['TEMP']=os.environ['TMP']=tempfile.tempdir=str(root_temp)
    reg=read(INVENTORY/'registration.json')
    assert sha(INVENTORY/'registration.json')=='f650f1ad030173ef93f62d1830816d6114798777db9375a6f62f4ce99dfb6b78'
    ready=read(reg['source']['ready']);files=Path(ready['files']).resolve()
    # Bind imported constructors and cache provenance, without replaying any old study.
    for path in (reg['source']['ready'],str(Path(reg['source']['cache'])/'index.json')):
        assert sha(path)==reg['inputs'][path]
    assert sha(REPO/'scripts/olr_structural_native_test_isolated_v6.py')==reg['files']['scripts/olr_structural_native_test_isolated_v6.py']
    guard_mod=load('sector_feature_import_guard',REPO/'scripts/olr_structural_native_test_isolated_v6.py')
    guard=guard_mod.NativeOnly(files,ready['manifest']['files']);sys.meta_path.insert(0,guard)
    sys.path[:]=[str(files)]+[p for p in sys.path if not Path(p or os.getcwd()).resolve().is_relative_to(REPO)]
    os.chdir(files)
    assert sha(INVENTORY/'clock_repair.json')=='6c317879ab0b104ef20f83e6a56d23040f13e7e8923b46f4b43b11fb142715fd'
    repair=read(INVENTORY/'clock_repair.json')
    assert sha(REPO/'scripts/olr_opportunity_clock_v7.py')==repair['files']['scripts/olr_opportunity_clock_v7.py']
    clock=load('sector_feature_clock',REPO/'scripts/olr_opportunity_clock_v7.py')
    return reg,ready,guard,clock


def freeze(reg,ready,guard,clock):
    import xml.etree.ElementTree as ET
    tests=read(ROOT/'test_receipt.json')
    assert tests['exit_code']==0
    for path,digest in tests['files'].items():assert sha(path)==digest,path
    suites=list(ET.parse(ROOT/'tests.xml').getroot().iter('testsuite'))
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts['tests']>=10 and not any(counts[k] for k in ('failures','errors','skipped'))
    old=BASE/'continuous_panel_v1';prior=read(old/'feature_validation_receipts.json')
    train=prior['windows']['train'];assert sha(train['path'])==train['sha256']
    index_path=Path(reg['source']['cache'])/'index.json';index=read(index_path)
    paths=[REPO/n for n in NAMES]+[ROOT/n for n in ('tests.xml','tests.log','test_receipt.json')]
    paths += [INVENTORY/'registration.json',INVENTORY/'clock_repair.json',Path(reg['source']['ready']),
        Path(ready['study']['manifest']),index_path,Path(index['parent_receipt']),
        old/'feature_validation_receipts.json',Path(train['path']),old/'cache_sources/train/source.json']
    protocol=dict(utc=utc(),version='causal-sector-source-contract-v1',
        scope='215 existing IS context sessions /103 fixed symbols, May12 2025 through March31 2026; no payoff, selection, score retuning or OOS data.',
        hypothesis='Single-day panel construction discards the previous-volume denominator; native history API already exists. Audit source-qualified causal use and explicit unknown support before any economic proposal.',
        constructor='Unchanged native preceding20 POSITIVE completed-prefix volumes, arithmetic mean (not median), denominator floor1; leave-one-out sector peers, native 3-member/3.0 shrinkage. Missing baseline remains null in qualified fields; native default1 is recorded separately.',
        clocks='Close-anchored14:30 completion, including Jan2 late open; missing required bars excluded, no interpolation. All raw cached bars must match authenticated database slices before filtering.',
        comparisons='Current single-day native panel; streaming causal history adapter; unchanged public native history constructor; existing source-bound static panel. Independently rederive own ratios, peer relative volume/participation and prior20 lists. Report all missingness, cold starts, stale history, legacy differences, and ordinary/Jan2 changes.',
        constraints='No new source materialization, minute origin reconstruction, old feature builder/collector/validator rerun, trading replay, candidate change or economic advancement. Correctness/nondegeneracy is not alpha or permission to adopt the previously failed uncalibrated substitution.',
        tests=counts,source=reg['source'],bindings={str(p):sha(p) for p in paths},native_test_imports=tests['imports'])
    write(ROOT/'protocol.json',protocol);write(OUT/'protocol.json',protocol)
    print('Source-only feature protocol frozen; no feature audit outcomes computed.')


def authenticate():
    protocol=read(ROOT/'protocol.json')
    for path,digest in protocol['bindings'].items():assert sha(path)==digest,path
    return protocol


def audit(reg,ready,guard,clock):
    authenticate();assert not (ROOT/'ledger.json').exists() and not (ROOT/'completion.json').exists()
    from backtests.core.input_manifest import InputManifest
    from scripts.kalcb_executed_value_study import connect,load_day
    from strategy_common.sector_intraday import build_sector_intraday_panel,AFTERNOON_CUTOFF
    cache=Path(reg['source']['cache']);index=read(cache/'index.json')
    sig=index.pop('index_signature');assert signature(index)==sig
    parent=read(index['parent_receipt']);assert sha(index['parent_receipt'])==index['parent_receipt_sha256']
    assert parent['inputs']==index['inputs'] and parent['identity']==index['identity']['base']
    files=Path(ready['files']);generation=InputManifest(ready['study']['manifest'])
    for name,digest in index['inputs'].items():assert ready['manifest']['files'].get(name)==sha(files/name)==digest,name
    def payload(name):
        assert sha(cache/name)==index['files'][name],name
        with (cache/name).open('rb') as stream:return pickle.load(stream)
    static=payload('static.pickle');symbols=sorted(static['symbols']);sectors={s:static['sectors'][s] for s in symbols}
    assert len(symbols)==103 and all(sectors[s] not in ('','UNKNOWN') for s in symbols)
    days=[date.fromisoformat(d) for d in index['identity']['base']['context_dates']]
    assert len(days)==215 and days[-1]==date(2026,3,31)
    database=files/ready['study']['database'];assert sha(database)==index['identity']['base']['database_sha256']==ready['study']['database_hash']
    connection=connect({'database':str(database)})
    histories={};rows=[];all_bars={};day_receipts=[];stream_features={}
    with generation.activate():
        for day in days:
            name=str(day)+'.pickle';snapshot,_,cached_daily,bars=payload(name)
            assert snapshot.trade_date==day
            actual=defaultdict(list)
            for b in load_day(connection,str(day),ready['study']['database_hash'],set(symbols)):actual[day,b.symbol].append(b)
            normalize=lambda mapping:{s:[b.to_json_dict() for b in mapping.get((day,s),())] for s in symbols}
            assert normalize(actual)==normalize(bars),str(day)
            qualified,excluded,cutoff=qualify_day({(day,s):bars.get((day,s),()) for s in symbols},day,clock)
            all_bars.update(qualified)
            legacy=build_sector_intraday_panel(qualified,sectors,trade_dates=(day,),cutoff=AFTERNOON_CUTOFF)
            members,details,stream=causal_members(qualified,sectors,histories,day)
            stream_features.update(stream.features_by_key)
            for symbol in symbols:
                detail=details.get(symbol);old=legacy.feature_for(day,symbol,sector=sectors[symbol])
                value=stream.feature_for(day,symbol,sector=sectors[symbol])
                cached=static['panel'].feature_for(day,symbol,sector=sectors[symbol])
                expected=peer_expectation(members,details,symbol) if detail else None
                if expected:
                    expected_peers=[s for s in symbols if s!=symbol and sectors[s]==sectors[symbol]]
                    expected.update(fixed_sector_peer_count=len(expected_peers),observed_market_peer_count=len(members)-1,
                        missing_sector_peers=sorted(set(expected_peers)-set(details)),fixed_market_peer_count=len(symbols)-1)
                    expected['complete_fixed_peer_coverage']=(not expected['missing_sector_peers'] if expected['weight']==1 else len(members)==len(symbols))
                    near(value.rel_volume,expected['relative_volume']);near(value.participation,expected['participation'])
                    near(value.shrinkage_weight,expected['weight']);assert value.effective_count==expected['effective_count']
                    near(old.rel_volume,1.)
                rows.append(dict(day=str(day),symbol=symbol,sector=sectors[symbol],available_at=cutoff.isoformat(),
                    source_complete=detail is not None,source_exclusion=symbol+':'+str(day) in excluded,
                    own=detail,peer_support=expected,legacy_native=asdict(old),history_native=asdict(value),
                    existing_static=asdict(cached),existing_static_relvol_changed=not math.isclose(value.rel_volume,cached.rel_volume,rel_tol=1e-12,abs_tol=1e-12)))
            update_history(histories,members,day)
            day_receipts.append(dict(day=str(day),payload_sha256=index['files'][name],raw_database_parity=True,
                prefix_hash=signature({s:[b.to_json_dict() for b in qualified.get((day,s),())] for s in symbols}),
                complete_names=len(details),excluded=excluded))
            print(json.dumps(dict(phase='causal_feature_audit',day=str(day),rows=len(rows))),flush=True)
        native_all=build_sector_intraday_panel(all_bars,sectors,trade_dates=days,cutoff=AFTERNOON_CUTOFF)
        assert set(native_all.features_by_key)==set(stream_features)
        for key,value in stream_features.items():
            actual=native_all.features_by_key[key]
            for field in ('symbol','sector','trade_date','cutoff_label'):assert getattr(actual,field)==getattr(value,field)
            for field in ('score_pct','ret','breadth','rel_volume','participation','shrinkage_weight','raw_score','member_score'):
                near(getattr(actual,field),getattr(value,field))
            assert actual.effective_count==value.effective_count
    connection.close()
    assert len(rows)==215*103
    write(ROOT/'ledger.json',rows)
    write(ROOT/'completion.json',dict(utc=utc(),protocol_sha256=sha(ROOT/'protocol.json'),ledger_sha256=sha(ROOT/'ledger.json'),
        imports=imports(guard),days=day_receipts,source_inputs=index['inputs'],public_constructor_parity=True,
        source_database_sha256=sha(database),no_payoff_files_read=True,new_native_replays=0,promotion=False))


def summarize(rows):
    train=[r for r in rows if r['day']>='2025-08-07'];supported=[r for r in train if r['peer_support'] and r['peer_support']['supported']]
    return dict(rows=len(rows),training_rows=len(train),training_days=len({r['day'] for r in train}),
        complete_training_rows=sum(r['source_complete'] for r in train),
        missing_training_rows=sum(not r['source_complete'] for r in train),
        own_missing_baseline=sum(r['own'] is not None and r['own']['relative_volume'] is None for r in train),
        full20_own_history=sum(r['own'] is not None and r['own']['prior_count']==20 for r in train),
        source_qualified_peer_rows=len(supported),
        fully_observed_fixed_peer_rows=sum(r['peer_support']['complete_fixed_peer_coverage'] for r in supported),
        nonunit_qualified_peer_rows=sum(abs(r['peer_support']['qualified_relative_volume']-1)>1e-12 for r in supported),
        old_static_changed_rows=sum(r['existing_static_relvol_changed'] for r in train),
        old_static_changed_dates=sorted({r['day'] for r in train if r['existing_static_relvol_changed']}),
        latest_prior_older_than_20_calendar_days=sum(r['own'] is not None and r['own']['latest_age_days'] is not None and r['own']['latest_age_days']>20 for r in train),
        jan2=dict(rows=sum(r['day']=='2026-01-02' for r in rows),complete=sum(r['day']=='2026-01-02' and r['source_complete'] for r in rows),
                  static_changed=sum(r['day']=='2026-01-02' and r['existing_static_relvol_changed'] for r in rows)),
        feature_contract_verified=True,selection_changed=False,payoff_measured=False,portfolio_experiment_eligible=False,promotion=False)


def verify_ledger(rows):
    # A separate scalar pass reconstructs prior lists and peer blend from the
    # source-bound daily ledger; it does not use the native constructor verdict.
    history=defaultdict(list)
    for day in sorted({r['day'] for r in rows}):
        current=[r for r in rows if r['day']==day];present={r['symbol']:r for r in current if r['own']}
        for symbol,r in present.items():
            own=r['own'];prior=history[symbol][-20:];assert own['history']==prior
            assert own['prior_count']==len(prior)
            if prior:
                near(own['prior_mean'],sum(v for _,v in prior)/len(prior))
                near(own['relative_volume'],own['volume']/max(sum(v for _,v in prior)/len(prior),1))
                assert own['latest_age_days']==(date.fromisoformat(day)-date.fromisoformat(prior[-1][0])).days
                assert own['oldest_age_days']==(date.fromisoformat(day)-date.fromisoformat(prior[0][0])).days
            else:assert own['relative_volume'] is None and own['prior_mean'] is None
            peers=[v for s,v in present.items() if s!=symbol and v['sector']==r['sector']]
            market=[v for s,v in present.items() if s!=symbol]
            weight=1 if len(peers)>=3 else len(peers)/(len(peers)+3)
            avg=lambda values:sum(v['own']['relative_volume'] if v['own']['relative_volume'] is not None else 1 for v in values)/len(values) if values else 1
            near(r['history_native']['rel_volume'],weight*avg(peers)+(1-weight)*avg(market))
            participation=lambda values:sum(v['own']['ret']>0 and (v['own']['relative_volume'] if v['own']['relative_volume'] is not None else 1)>=1 for v in values)/len(values) if values else 0
            near(r['history_native']['participation'],weight*participation(peers)+(1-weight)*participation(market))
            near(r['history_native']['shrinkage_weight'],weight)
            assert r['history_native']['effective_count']==len(peers)
            contributors=peers if weight==1 else market
            support=bool(contributors) and all(v['own']['prior_count']>0 for v in contributors)
            assert r['peer_support']['supported']==support
            if not support:assert r['peer_support']['qualified_relative_volume'] is None
            else:near(r['peer_support']['qualified_relative_volume'],r['history_native']['rel_volume'])
            expected_peers=[v['symbol'] for v in current if v['symbol']!=symbol and v['sector']==r['sector']]
            assert r['peer_support']['missing_sector_peers']==sorted(set(expected_peers)-set(present))
        for symbol,r in present.items():
            if r['own']['volume']>0:history[symbol].append([day,r['own']['volume']])


def report():
    authenticate();done=read(ROOT/'completion.json')
    assert done['ledger_sha256']==sha(ROOT/'ledger.json') and done['protocol_sha256']==sha(ROOT/'protocol.json')
    rows=read(ROOT/'ledger.json');verify_ledger(rows);summary=summarize(rows)
    write(OUT/'summary.json',summary)
    write(OUT/'selected_research_policy.json',dict(selected_id='c4f49be01ed09b8dc',promoted=False,live_activation=False,summary_sha256=sha(OUT/'summary.json')))
    lines=['OLR causal sector-volume feature contract audit',
        'Source-only audit. No selections, payoff labels, threshold changes or policy promotion.',
        'History adapter matched unchanged native full-history constructor. Independent ratios and leave-one-out peer blends passed.',
        'The existing denominator is the mean of preceding20 positive complete-prefix volumes, not a median. Single-day calls have no preceding rows and supply native default1.',
        'Source-qualified fields preserve unknown baselines and peer support as null; native neutral/default values remain separately visible. Qualified ratios describe observed peers; full fixed-universe peer coverage is a separate field.',
        json.dumps(summary,indent=2),
        'The existing static panel is an authenticated reference, not automatically adopted; clock/source differences are retained in the ledger.',
        'The previous uncalibrated history-feature trading substitution failed. Correctness and nondegeneracy do not establish alpha or authorize repeating it.',
        'Next economic hypothesis requires separate preregistration and chronological net-of-stress evidence; rejected score-only cohort and rank-calibration model stay closed.',
        'Limitations: fixed universe/taxonomy, source-completeness exclusions, positive-volume-conditioned prior window, calendar age variation, sparse auctions/inferred clocks and conservative execution proxies. April-July repeatedly inspected development, not independent validation; unused here.']
    content='\n\n'.join(lines)+'\n'
    with (OUT/'report.txt').open('x',encoding='utf-8') as stream:stream.write(content)
    from html import escape
    with (OUT/'report.html').open('x',encoding='utf-8') as stream:stream.write('<!doctype html><meta charset="utf-8"><title>OLR feature audit</title><style>body{max-width:1000px;margin:40px auto;font:16px system-ui;line-height:1.6}pre{white-space:pre-wrap}</style><pre>'+escape(content)+'</pre>')
    bundle=OUT/'review_bundle';bundle.mkdir();patch=[]
    for name in NAMES[:2]:
        source=REPO/name;target=bundle/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        patch.extend(difflib.unified_diff([],source.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile='b/'+name))
    (bundle/'implementation.patch').write_text(''.join(patch),encoding='utf-8')
    paths=[OUT/n for n in ('summary.json','report.txt','report.html','selected_research_policy.json','review_bundle/implementation.patch')]
    write(OUT/'report_receipt.json',dict(utc=utc(),completion_sha256=sha(ROOT/'completion.json'),files={str(p):sha(p) for p in paths}))
    print(json.dumps(summary))


def verify():
    authenticate();assert not (OUT/'delivery_checks.json').exists()
    receipt=read(OUT/'report_receipt.json');review=read(OUT/'delivery_review.json')
    assert receipt['completion_sha256']==sha(ROOT/'completion.json') and review['report_receipt_sha256']==sha(OUT/'report_receipt.json')
    for path,digest in {**receipt['files'],**review['files']}.items():assert sha(path)==digest,path
    done=read(ROOT/'completion.json');assert done['ledger_sha256']==sha(ROOT/'ledger.json')
    ready=read(read(ROOT/'protocol.json')['source']['ready'])
    for name,digest in {**done['imports'],**done['source_inputs']}.items():assert sha(Path(ready['files'])/name)==digest,name
    rows=read(ROOT/'ledger.json');verify_ledger(rows);assert summarize(rows)==read(OUT/'summary.json')
    write(OUT/'delivery_checks.json',dict(utc=utc(),passed=True,protocol_sha256=sha(ROOT/'protocol.json'),
        completion_sha256=sha(ROOT/'completion.json'),report_receipt_sha256=sha(OUT/'report_receipt.json'),
        summary_sha256=sha(OUT/'summary.json'),profile_sha256=sha(OUT/'selected_research_policy.json'),promotion=False))
    print('Feature source/constructor/independent arithmetic and artifact bindings verified.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['tests','freeze','audit','report','verify']);phase=parser.parse_args().phase
    if phase in ('report','verify'):
        globals()[phase]();return
    reg,ready,guard,clock=setup()
    if phase=='tests':
        sys.modules['sector_feature_runtime']=sys.modules[__name__]
        import pytest
        rc=pytest.main([str(REPO/NAMES[1]),'-q','--junitxml='+str(ROOT/'tests.xml')])
        write(ROOT/'test_receipt.json',dict(exit_code=rc,imports=imports(guard),files={str(REPO/n):sha(REPO/n) for n in NAMES}))
        raise SystemExit(rc)
    if phase=='freeze':freeze(reg,ready,guard,clock)
    else:audit(reg,ready,guard,clock)


if __name__=='__main__':main()

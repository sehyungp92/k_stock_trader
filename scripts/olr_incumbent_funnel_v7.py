"""Explain saved incumbent admissions with the unchanged native signal rule.

This is descriptive source/decision reconciliation, not a new entry candidate
or a payoff study of the rejected score-map cohort.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import gzip
import importlib.util
import json
import os
import sys


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


def main():
    repo=Path(__file__).resolve().parents[1]
    c=load('incumbent_funnel_contract',repo/'scripts/olr_opportunity_v7.py')
    out=c.OUT/'incumbent_funnel';out.mkdir(exist_ok=False)
    inventory=c.read(c.OUT/'inventory_checks.json');assert inventory['passed']
    assert inventory['ledger_sha256']==c.sha(c.ROOT/'opportunities_train.json')
    ready_path=c.ROOT.parent/'native_controlled/native_engines/train/ready.json'
    ready=c.read(ready_path);files=Path(ready['files']).resolve()
    case=c.ROOT.parent/'native_controlled/cases/cbd557f1059b42dc0/train/100000000_baseline'
    completion=c.read(case/'completion.json');assert completion['complete']
    paths=[ready_path,c.OUT/'inventory_checks.json',c.ROOT/'opportunities_train.json',case/'completion.json',
           case/'trades.json',case/'native_decisions.jsonl.gz',case/'request.json',Path(__file__)]
    for name in ('trades.json','native_decisions.jsonl.gz','request.json'):
        assert completion['artifacts'][name]==c.sha(case/name)
    database=files/ready['study']['database']
    assert c.sha(database)==ready['study']['database_hash']==ready['manifest']['files'][ready['study']['database']]
    c.write(out/'protocol.json',dict(utc=c.utc(),purpose='Descriptive incumbent-only signal-to-admission reconciliation; no parameter change or payoff estimation.',
        already_observed=dict(selected_candidates=356,filled_lifecycles=185,no_same_day_entry_event=145),
        scope='Original nominal IS control only. Recompute frozen entry signal over each completed prefix, reconcile with saved native admissions and known position intervals. No score-only rejected candidate is evaluated.',
        source_inputs={str(p):c.sha(p) for p in paths},database_sha256=c.sha(database),
        performance_fields_used=False,new_portfolio_replays=0,independent_validation=False))
    isolation=load('incumbent_native_isolation',repo/'scripts/olr_structural_native_test_isolated_v6.py')
    guard=isolation.NativeOnly(files,ready['manifest']['files']);sys.meta_path.insert(0,guard)
    sys.path[:]=[str(files)]+[p for p in sys.path if not Path(p or os.getcwd()).resolve().is_relative_to(repo)]
    os.chdir(files)
    from backtests.core.input_manifest import InputManifest
    from strategy_olr.models import OLRDailyCandidate
    from strategy_olr.config import OLRConfig
    from strategy_olr.core.logic import _entry_plan
    from strategy_olr.core.precision_v3 import entry_plan
    from strategy_olr.execution import find_olr_entry_signal
    from scripts.kalcb_executed_value_study import connect,load_day
    request=c.read(files/ready['study']['olr_request'])
    cfg=OLRConfig.from_mapping(request['config'],request['mutations'])
    plan=entry_plan(_entry_plan(cfg),cfg)
    snapshots=c.read(files/ready['study']['olr_snapshots'])
    selected={s['trade_date']:s for s in snapshots if '2025-08-07'<=s['trade_date']<='2026-03-31'}
    trades=c.read(case/'trades.json');events=defaultdict(list)
    with gzip.open(case/'native_decisions.jsonl.gz','rt',encoding='utf8') as stream:
        for line in stream:
            e=json.loads(line)
            if e['decision_code'] in ('ENTRY_SUBMITTED','ENTRY_SKIPPED','ENTRY_DEFERRED'):
                events[e['timestamp'][:10],e['symbol']].append(e)
    connection=connect({'database':str(database)});rows=[]
    generation=InputManifest(ready['study']['manifest'])
    with generation.activate():
        for day,snapshot in selected.items():
            wanted={x['symbol'] for x in snapshot['candidates']};bars=defaultdict(list)
            for bar in load_day(connection,day,ready['study']['database_hash'],wanted):bars[bar.symbol].append(bar)
            for index,raw in enumerate(snapshot['candidates']):
                symbol=raw['symbol'];candidate=OLRDailyCandidate.from_json_dict(raw);series=tuple(bars[symbol])
                signal=find_olr_entry_signal(series,candidate,plan,require_fill_bar=False)
                signal_time=None
                if signal is not None:
                    prefix=series[:signal.signal_index+1]
                    causal=find_olr_entry_signal(prefix,candidate,plan,require_fill_bar=False)
                    assert causal is not None and causal.signal_index==signal.signal_index and causal.reason==signal.reason
                    signal_time=series[signal.signal_index].timestamp+timedelta(minutes=5)
                matched=[t for t in trades if t['symbol']==symbol and t['entry_decision_time'][:10]==day]
                assert len(matched)<=1
                admission=events[day,symbol]
                held=bool(signal_time is not None and any(t['symbol']==symbol and datetime.fromisoformat(t['entry_fill_time'])<=signal_time<datetime.fromisoformat(t['exit_fill_time']) for t in trades))
                if matched:reason='filled_lifecycle'
                elif admission:reason='recorded_admission_without_fill'
                elif index>=int(cfg.overnight_slot_count):reason='outside_original_slots'
                elif signal is None:reason='no_original_confirmation'
                elif held:reason='existing_position_at_first_confirmation'
                else:reason='unresolved_native_state_or_prior_attempt'
                if admission:
                    assert signal_time is not None,'Admission lacks original confirmation'
                rows.append(dict(day=day,symbol=symbol,rank=index+1,disposition=reason,
                    native_signal_time=str(signal_time) if signal_time else None,position_open_at_first_signal=held,
                    admissions=[dict(timestamp=x['timestamp'],code=x['decision_code'],reason=x['reason']) for x in admission],
                    filled=bool(matched),native_prefix_signal_exact=signal is not None))
    connection.close()
    assert len(rows)==356 and sum(r['filled'] for r in rows)==185
    imports={}
    for name,module in list(sys.modules.items()):
        if name.split('.')[0] not in guard.roots or not getattr(module,'__file__',None):continue
        p=Path(module.__file__).resolve();assert p.is_relative_to(files);key=p.relative_to(files).as_posix()
        assert ready['manifest']['files'][key]==c.sha(p);imports[key]=c.sha(p)
    counts=dict(Counter(r['disposition'] for r in rows))
    c.write(out/'funnel.json',dict(rows=rows,counts=counts,selected_dates=len({r['day'] for r in rows}),
        filled_dates=len({r['day'] for r in rows if r['filled']}),
        interpretation='Descriptive first-confirmation reconciliation. Unresolved state is explicit, not assigned to an invented reason. A signal or unfilled candidate is not an alpha label. Existing holdings do not create an independent new position. No outcomes or alternative thresholds were evaluated.'))
    c.write(out/'receipt.json',dict(utc=c.utc(),protocol_sha256=c.sha(out/'protocol.json'),funnel_sha256=c.sha(out/'funnel.json'),
        imports=imports,rows=356,filled_lifecycles=185,new_portfolio_replays=0,performance_fields_used=False))
    print(json.dumps(dict(counts=counts,output=str(out)),ensure_ascii=True))


if __name__=='__main__':main()

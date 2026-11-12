"""Saved-lane recollection, diagnostic report, and post-review verification."""
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from html import escape
import argparse
import difflib
import gzip
import importlib.util
import json
import math
import os
import sqlite3
import sys


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


def near(a,b):
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-5),(a,b)


def restore_journal_types(journal):
    """Restore only the native Fill datetime erased by JSON default=str.

    Audit code calls isoformat on the native datetime; leaving its serialized
    space-separated string would create a false capacity-offer key mismatch.
    The saved bytes and every other journal field remain untouched.
    """
    for event in journal:
        if event['event']=='fill':
            event['fill']['timestamp']=datetime.fromisoformat(event['fill']['timestamp'])
    return journal


def summarize(rows, design):
    grouped=defaultdict(list)
    for row in rows:grouped[row['day'],row['symbol']].append(row)
    paired={key for key,value in grouped.items() if len(value)==4 and all(r['status']=='matured' for r in value)}
    cells=[];support=[]
    for lo,hi in design['blocks']:
        keys={key for key in grouped if lo<=key[0]<=hi}
        # A block endpoint can only use labels that had matured by that date.
        matured={key for key in keys&paired if all(r['available_for_training_after'][:10]<=hi for r in grouped[key])}
        days={key[0] for key in matured}
        support.append(dict(start=lo,end=hi,opportunities=len(keys),paired_mature=len(matured),
            paired_coverage=len(matured)/len(keys) if keys else 0,paired_days=len(days),
            passed=len(days)>=10 and len(matured)>=.8*len(keys)))
        for scenario in design['scenarios']:
            for budget in design['target_budgets']:
                sample=[r for key in sorted(matured) for r in grouped[key] if r['scenario']==scenario and r['budget']==budget]
                daily=defaultdict(list)
                for row in sample:daily[row['day']].append(row['net_bps'])
                mean=sum(sum(v)/len(v) for v in daily.values())/len(daily) if daily else None
                wins=sorted((r['net_pnl'] for r in sample if r['net_pnl']>0),reverse=True)
                cells.append(dict(start=lo,end=hi,scenario=scenario,budget=budget,n=len(sample),days=len(daily),
                    date_weighted_net_bps=mean,mean_budget_net_bps=sum(r['budget_net_bps'] for r in sample)/len(sample) if sample else None,
                    mean_filled_notional=sum(r['filled_notional'] for r in sample)/len(sample) if sample else None,
                    max_target_drawdown_pct=max((r['fixed_target_risk']['max_drawdown_over_target_pct'] for r in sample),default=None),
                    positive_pnl_top5_share=sum(wins[:5])/sum(wins) if wins else None,
                    winning_labels=sum(r['net_pnl']>0 for r in sample)))
    small=[r for r in cells if r['scenario']=='combined' and r['budget']==min(design['target_budgets'])]
    gates=dict(each_block_paired_support=all(r['passed'] for r in support),
        each_block_small_stressed_net_positive=all(r['date_weighted_net_bps'] is not None and r['date_weighted_net_bps']>0 for r in small))
    pairs=[]
    winner_retention=[]
    for scenario in design['scenarios']:
        for key in sorted(paired):
            by_budget={r['budget']:r for r in grouped[key] if r['scenario']==scenario}
            a,b=[by_budget[n] for n in sorted(by_budget)]
            delta=b['filled_notional']-a['filled_notional']
            pairs.append(dict(day=key[0],symbol=key[1],scenario=scenario,
                acquired_delta=delta,pnl_delta=b['net_pnl']-a['net_pnl'],
                additional_filled_capital_net_bps=10000*(b['net_pnl']-a['net_pnl'])/delta if delta>0 else None,
                unit_payoff_change_bps=b['net_bps']-a['net_bps']))
        paired_rows=[{r['budget']:r for r in grouped[key] if r['scenario']==scenario} for key in sorted(paired)]
        winners=[v for v in paired_rows if v[min(design['target_budgets'])]['net_pnl']>0]
        kept=[v for v in winners if v[max(design['target_budgets'])]['net_pnl']>0]
        small_profit=lambda v:v[min(design['target_budgets'])]['net_pnl']
        winner_retention.append(dict(scenario=scenario,small_target_winners=len(winners),large_target_still_profitable=len(kept),
            small_winner_pnl_share_retained=sum(map(small_profit,kept))/sum(map(small_profit,winners)) if winners else None,
            scope='Paired mature diagnostic lanes only; not retained-policy whole-winner preservation.'))
    descriptive=[]
    for field in ('rank','incumbent_disposition'):
        for value in sorted({str(r.get(field,'unavailable')) for r in rows}):
            for scenario in design['scenarios']:
                for budget in design['target_budgets']:
                    sample=[r for r in rows if (r['day'],r['symbol']) in paired and str(r.get(field,'unavailable'))==value
                            and r['scenario']==scenario and r['budget']==budget]
                    daily=defaultdict(list)
                    for r in sample:daily[r['day']].append(r['net_bps'])
                    descriptive.append(dict(field=field,value=value,scenario=scenario,budget=budget,n=len(sample),days=len(daily),
                        date_weighted_net_bps=sum(sum(v)/len(v) for v in daily.values())/len(daily) if daily else None,
                        warning='Descriptive, selected sample; no subgroup exclusion or size rule inferred.'))
    return dict(lanes=len(rows),opportunities=len(grouped),sample_dates=len({r['day'] for r in rows}),
        dispositions=dict(Counter(r['status'] for r in rows)),paired_complete_opportunities=len(paired),
        chronological_support=support,cells=cells,paired_size_changes=pairs,gates=gates,
        paired_winner_retention=winner_retention,descriptive_rank_and_funnel=descriptive,
        separate_allocation_calibration_may_be_considered=all(gates.values()),
        portfolio_experiment_eligible=False,promotion=False,independent_validation=False,
        note='Fixed requests isolate funding/forecast sizing feedback while keeping native loss limits and physical execution. No added portfolio frequency or funded portfolio returns are measured. Missing outcomes are not zero-return trades.')


def collect(c,design,ready,guard):
    from backtests.core.input_manifest import InputManifest
    from backtests.engine.olr_minute_source_v5 import load_minute_day
    from backtests.engine.olr_minute_audit_v5 import audit_minute_capacity,audit_minute_clock
    from backtests.engine.olr_minute_v5 import MinuteParticipationScenario,fixed_exit_plan
    from scripts.kalcb_executed_value_study import load_day
    from strategy_olr.config import OLRConfig
    from dataclasses import asdict,replace
    done=c.read(c.ROOT/'labels_complete.json');assert done['complete']
    assert done['frozen_harness_sha256']==c.sha(c.ROOT/'frozen_harness.json')
    assert done['source_proofs_sha256']==c.sha(c.ROOT/'source_proofs.json')
    proof=c.read(c.ROOT/'source_proofs.json')
    generation=InputManifest(ready['study']['manifest']);files=generation.root
    assert proof['unchanged_source_manifest_sha256']==c.sha(ready['study']['manifest'])
    definition=c.read(files/ready['study']['minute_source_definition'])
    for label,digest in proof['source_inputs'].items():
        path=files/definition['aliases'][label] if label in definition['aliases'] else Path(label)
        if not path.is_absolute():path=files/path
        path=path.resolve()
        assert path.is_relative_to(files.resolve()), label
        assert generation.digest(path)==digest==c.sha(path),label
    for name,item in done['imports'].items():
        path=Path(item['path']);assert c.sha(path)==item['sha256']
        overlay=name.replace('.','/')+'.py'
        if overlay in design['overlays']:assert item['sha256']==design['overlays'][overlay]['sha256']
        else:assert ready['manifest']['files'][path.relative_to(files).as_posix()]==item['sha256']
    connections=[]
    for key in ('minute_database','canonical_database'):
        path=files/definition[key];assert c.sha(path)==generation.digest(path)
        connections.append(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True))
    request=c.read(files/ready['study']['olr_request'])
    cfg=replace(OLRConfig.from_mapping(request['config'],request['mutations']),**ready['study']['candidates']['candidate']['olr_capacity'])
    plan=asdict(fixed_exit_plan(cfg))
    cache={};rows=[]
    funnel=c.read(c.INVENTORY/'incumbent_funnel/funnel.json')['rows']
    disposition={(r['day'],r['symbol']):r['disposition'] for r in funnel}
    wanted={(x['day'],x['symbol'],s,b) for x in design['cohort'] for s in design['scenarios'] for b in design['target_budgets']}
    seen=set()
    with generation.activate():
        for index,item in enumerate(done['lanes']):
            lane=Path(item['path']);assert lane.resolve().is_relative_to((c.ROOT/'lanes').resolve())
            assert c.sha(lane/'result.json')==item['result_sha256']
            assert c.sha(lane/'receipt.json')==item['receipt_sha256']
            receipt=c.read(lane/'receipt.json');row=c.read(lane/'result.json')
            for path,key in (('result.json','result_sha256'),('execution_audit.json','audit_sha256'),('journal.jsonl.gz','journal_sha256')):
                assert c.sha(lane/path)==receipt[key]
            assert receipt['frozen_harness_sha256']==done['frozen_harness_sha256']
            key=(row['day'],row['symbol'],row['scenario'],row['budget']);assert key in wanted and key not in seen;seen.add(key)
            assert c.signature({k:proof['sessions'][k] for k in row['source_sessions']})==row['source_proof_hash']==receipt['source_proof_hash']
            with gzip.open(lane/'journal.jsonl.gz','rt',encoding='utf-8') as stream:journal=restore_journal_types([json.loads(line) for line in stream])
            fine=[];coarse=[]
            for scope in row['source_sessions']:
                if scope not in cache:
                    symbol,day=scope.split(':');raw=load_minute_day(connections[0],day,{symbol})
                    old=load_day(connections[1],day,ready['study']['database_hash'],{symbol})
                    cache[scope]=([b.to_json_dict() for b in raw],[b.to_json_dict() for b in old])
                    assert c.signature(cache[scope][0])==proof['sessions'][scope]['fine_hash']
                    assert c.signature(cache[scope][1])==proof['sessions'][scope]['coarse_hash']
                a,b=cache[scope];fine.extend((r,) for r in a);coarse.extend((r,) for r in b)
            cash=100_000_000.;quantity=0;acquired=cost=proceeds=0.;active={};fills=[];marks={}
            for event in journal:
                kind=event['event']
                if kind in ('opening_marks','closing_marks'):marks.update(event['prices'])
                elif kind in ('submit','amend'):active[event['order']['order_id']]=dict(event['order'])
                elif kind in ('cancel','expire','reject'):active.pop(event['order']['order_id'],None)
                elif kind=='fill':
                    fill=event['fill'];q=fill['qty'];p=fill['price'];meta=fill['metadata'];buy=fill['side']=='BUY'
                    slip=design['scenarios'][row['scenario']][1]
                    if event['order']['order_type']=='CLOSE_AUCTION':slip+=float(event['order']['metadata'].get('auction_adverse_bps',0))
                    near(p,meta['base_price']*(1+(slip if buy else -slip)/10000))
                    fee=q*p*(2 if buy else 22)/10000;near(fee,meta['fee']);near(cash,event['cash_before'])
                    cash+=(-q*p if buy else q*p)-fee;near(cash,event['cash_after'])
                    quantity+=q if buy else -q;assert quantity>=0
                    acquired+=q*p if buy else 0;proceeds+=q*p if not buy else 0;cost+=fee
                    order=active[fill['order_id']];order['qty']-=q
                    if not order['qty']:active.pop(fill['order_id'])
                    fills.append(SimpleNamespace(**fill))
            near(acquired,row['filled_notional']);near(cost,row['total_fees']);assert quantity==row['residual_quantity']
            assert len(active)==row['pending_orders']
            if row['status']=='matured':
                assert quantity==0 and not active and acquired>0
                near(proceeds-acquired-cost,row['net_pnl']);near(10000*row['net_pnl']/acquired,row['net_bps'])
            else:assert row['net_pnl'] is None and row['net_bps'] is None
            view=SimpleNamespace(journal=journal,expected_batches=fine,coarse_expected_batches=coarse,
                capacity_policy=MinuteParticipationScenario(design['scenarios'][row['scenario']][0],row['scenario']),
                initial_equity=100_000_000.,equity_curve=[cash+quantity*marks.get(row['symbol'],0)],
                fills=fills,positions={row['symbol']:SimpleNamespace(qty=quantity)} if quantity else {},
                orders=[SimpleNamespace(**o) for o in active.values()],minute_fast_exit=False,
                minute_exit_plan=plan,minute_exit_priority=cfg.capacity_exit_priority,
                minute_clock_schedule=[{k:e[k] for k in ('kind','label','available_at','delivered_by','symbols')}
                    for e in journal if e['event']=='coarse_observation_v5'])
            cap=audit_minute_capacity(view);clock=audit_minute_clock(view)
            assert not cap['errors'] and not clock['errors'],(key,cap['errors'],clock['errors'])
            saved=c.read(lane/'execution_audit.json')
            assert saved['execution_validity_pass'] and not saved['audit_receipt']['errors']
            assert not saved['audit_pass'],'Isolated lanes must not masquerade as whole-portfolio qualification'
            reduced={k:v for k,v in row.items() if k not in ('nav','decisions','native_requests')}
            reduced['incumbent_disposition']=disposition[row['day'],row['symbol']]
            peak=100_000_000.;common_dollars=0.
            for point in row['nav']['common_original_grid']:
                peak=max(peak,point['equity']);common_dollars=max(common_dollars,peak-point['equity'])
            reduced['fixed_target_risk']={**reduced['fixed_target_risk'],
                'common_coarse_drawdown_krw':common_dollars,
                'common_coarse_drawdown_over_target_pct':100*common_dollars/row['budget'],
                'fine_dummy_account_drawdown_pct':row['nav']['all_event_drawdown_pct'],
                'coarse_dummy_account_drawdown_pct':row['nav']['common_original_drawdown_pct']}
            rows.append(reduced)
            if index%40==0:print(json.dumps(dict(phase='independent_saved_lane_recollection',lanes=index+1)),flush=True)
    for connection in connections:connection.close()
    assert seen==wanted and len(rows)==design['max_lanes']
    return rows,summarize(rows,design)


def report(c,design,ready,guard):
    rows,summary=collect(c,design,ready,guard)
    c.write(c.OUT/'results.json',dict(summary=summary,lanes=rows))
    c.write(c.OUT/'selected_research_policy.json',dict(selected_id=design['retained'],promoted=False,
        diagnostic_only=True,live_activation=False,results_sha256=c.sha(c.OUT/'results.json')))
    lines=['OLR fixed-incumbent signal and execution diagnostic',
        'Original policy remains selected: '+design['retained']+'. No portfolio promotion or live activation.',
        f"{summary['opportunities']} incumbent signals on {summary['sample_dates']} dates; {summary['lanes']} independent lanes.",
        'Targets: KRW1m and KRW20m; nominal and combined execution stress. Original confirmation and exit thresholds, native loss limits and physical execution retained.',
        'Allocation intervention: fixed request replaces rank size; forecast request clipping and recycling disabled only inside the diagnostic. No assertion that these request limits should be removed in a trading policy.',
        'Dispositions: '+json.dumps(summary['dispositions']),
        'Paired block results require all four lanes matured by the block end. Unfilled/unsupported/unresolved outcomes are missing, not zero.',
        '', 'Block | Scenario | Target KRW | Paired N/days | Date-weighted net bps | Mean actual notional | Largest target-normalized drawdown']
    for r in summary['cells']:
        fmt=lambda x:'missing' if x is None else f'{x:.3f}'
        lines.append(f"{r['start']} to {r['end']} | {r['scenario']} | {r['budget']:,} | {r['n']}/{r['days']} | {fmt(r['date_weighted_net_bps'])} | {fmt(r['mean_filled_notional'])} | {fmt(r['max_target_drawdown_pct'])}%")
    lines+=['','Support: '+json.dumps(summary['chronological_support']), 'Registered gates: '+json.dumps(summary['gates']),
        'Separate allocation calibration may be considered: '+str(summary['separate_allocation_calibration_may_be_considered']),
        'These are isolated opportunities, not simultaneous self-funding trades, portfolio returns, or increased trading frequency. The account has idle cash for funding isolation; target-normalized dollar drawdown is reported separately from its diluted account DD.',
        'Paired winner retention: '+json.dumps(summary['paired_winner_retention']),
        'Both fine and common coarse drawdowns, plus the original dummy-account drawdowns, are retained for every lane in results.json. Exact-rank and original-funnel-category comparisons are descriptive and also included there.',
        'No fitted allocation or new alpha rule was selected. Paired size changes are controlled execution diagnostics, not summed portfolio counterfactuals. Existing filled winners are not used to label missing trades.',
        'Source/verification: authenticated original inputs, independent native reparsing of every consumed minute origin and exact coarse aggregation, saved-ledger cash/cost recollection, independent capacity and clock audit against authenticated minute/coarse database slices. Generic whole-portfolio source qualification remains explicitly unverified for isolated lanes.',
        'Limitations: fixed universe and taxonomy; sparse auction history; inferred archive clocks; conservative capacity/price proxies; five-session administrative censoring; selection on existing confirmation; overlapping opportunities; April-July repeatedly inspected development, not independent validation and not consumed here.',
        'A successor still must pass every original/guard/retained and matched fine-control return/DD/frequency/notional/stress/capped gate, both DD grids and actual history continuity.']
    text='\n'.join(lines)+'\n'
    with (c.OUT/'report.txt').open('x',encoding='utf-8') as f:f.write(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,scenario in zip(axes,design['scenarios']):
        for i,budget in enumerate(design['target_budgets']):
            cells=[r for r in summary['cells'] if r['scenario']==scenario and r['budget']==budget]
            vals=[float('nan') if r['date_weighted_net_bps'] is None else r['date_weighted_net_bps'] for r in cells]
            ax.bar([x+(i-.5)*.35 for x in range(3)],vals,width=.35,label=f'KRW {budget/1e6:g}m target')
        ax.set_xticks(range(3),['Aug-Oct','Nov-Dec','Jan-Mar']);ax.axhline(0,color='black',linewidth=.8)
        ax.set_title(scenario);ax.set_ylabel('Date-weighted net bps on actual filled notional');ax.legend()
    fig.suptitle('Existing confirmed signals: paired mature opportunities\nDiagnostic lanes, not portfolio returns')
    fig.savefig(c.OUT/'results.png',dpi=150);plt.close(fig)
    with (c.OUT/'report.html').open('x',encoding='utf-8') as f:
        f.write('<!doctype html><meta charset="utf-8"><title>OLR fixed-signal diagnostic</title><style>body{max-width:1200px;margin:32px auto;font:16px system-ui;line-height:1.5}pre{white-space:pre-wrap}img{width:100%}</style><h1>OLR fixed-signal diagnostic</h1><img src="results.png"><pre>'+escape(text)+'</pre><p><a href="selected_research_policy.json">Selected profile</a> | <a href="review_bundle/implementation.patch">Implementation patch</a></p>')
    bundle=c.OUT/'review_bundle';bundle.mkdir(exist_ok=False);patch=[]
    for name in c.NAMES+[str(Path(__file__).relative_to(c.REPO)).replace('\\','/'),
                         'tests/strategy_olr/test_fixed_signal_reporting_v8.py']:
        path=c.REPO/name;target=bundle/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
        patch.extend(difflib.unified_diff([],path.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile='b/'+name))
    for name,item in design['overlays'].items():
        old=Path(ready['files'])/name;new=Path(item['path'])
        patch.extend(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(True),new.read_text(encoding='utf-8').splitlines(True),fromfile='a/'+name,tofile='b/diagnostic_overlay/'+name))
    (bundle/'implementation.patch').write_text(''.join(patch),encoding='utf-8')
    paths=[c.OUT/n for n in ('results.json','selected_research_policy.json','report.txt','report.html','results.png','review_bundle/implementation.patch')]
    c.write(c.OUT/'report_receipt.json',dict(utc=c.utc(),files={str(p):c.sha(p) for p in paths},
        label_completion_sha256=c.sha(c.ROOT/'labels_complete.json'),report_code_sha256=c.sha(__file__),
        saved_lane_recollection_pass=True,imports=c.imports(guard)))
    print(json.dumps(summary['gates']))


def verify(c,design,ready,guard,reviewed):
    assert reviewed,'Visual review must precede final verification'
    assert not (c.OUT/'delivery_checks.json').exists(),'Never rerun a successful verifier'
    receipt=c.read(c.OUT/'report_receipt.json');review=c.read(c.OUT/'delivery_review.json')
    assert receipt['report_code_sha256']==c.sha(__file__)
    for path,digest in receipt['files'].items():assert c.sha(path)==digest,path
    assert review['report_receipt_sha256']==c.sha(c.OUT/'report_receipt.json')
    for path,digest in review['files'].items():assert c.sha(path)==digest,path
    rows,summary=collect(c,design,ready,guard)
    assert c.read(c.OUT/'results.json')==dict(summary=summary,lanes=rows)
    c.write(c.OUT/'delivery_checks.json',dict(utc=c.utc(),passed=True,
        report_receipt_sha256=c.sha(c.OUT/'report_receipt.json'),review_sha256=c.sha(c.OUT/'delivery_review.json'),
        results_sha256=c.sha(c.OUT/'results.json'),profile_sha256=c.sha(c.OUT/'selected_research_policy.json'),
        code_sha256=c.sha(__file__),independent_saved_arithmetic_recollection=True,promotion=False))
    print('Fixed-signal diagnostic report verified; original policy retained.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['report','verify']);parser.add_argument('--chart-reviewed',action='store_true');args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    c=load('fixed_signal_report_contract',repo/'scripts/olr_fixed_signal_v8.py');c.temp()
    design,ready=c.authenticate();files=Path(ready['files']).resolve()
    guard=c.NativeOverlay(files,ready['manifest']['files'],design['overlays']);sys.meta_path.insert(0,guard)
    sys.path[:]=[str(files)]+[p for p in sys.path if not Path(p or os.getcwd()).resolve().is_relative_to(repo)];os.chdir(files)
    if args.phase=='report':report(c,design,ready,guard)
    else:verify(c,design,ready,guard,args.chart_reviewed)

"""Read-only diagnosis of v3 failures; no new policy, fills or parameter search."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import gzip,hashlib,json,sys

REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'data/backtests/analysis/olr_precision_diagnosis_20261011'
PRIOR=REPO/'data/backtests/analysis/olr_precision_20261011'
ROOT=Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010')
FROZEN=Path.home()/'.codex/research/olr_precision_v3/oos/inputs/files'
SOURCES={}

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def read(path):
    SOURCES[str(path)]=sha(path)
    return json.loads(Path(path).read_text(encoding='utf-8'))

def auth(folder):
    comp=read(folder/'completion.json');assert comp['complete']
    for name,digest in comp['artifacts'].items():assert sha(folder/name)==digest,(folder,name)
    assert read(folder/'execution_audit.json')['audit_pass']

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=read(PRIOR/'results.json');profile=read(PRIOR/'selected_research_policy.json')
    assert profile['results_sha256']==sha(PRIOR/'results.json')
    rows={r['label']:r for r in data['candidates']};trades={}
    for label,r in rows.items():
        trades[label]={}
        for w in ('train','oos'):
            folder=Path(r['windows'][w]['folder']);auth(folder)
            trades[label][w]=read(folder/'trades.json')
    controls=trades['selected_control'];tail=trades['liquidation_tail_es90']
    key=lambda x:(x['symbol'],x['entry_decision_time'])
    counts={};all_tail=[]
    for w in ('train','oos'):
        c=Counter();cohorts=defaultdict(list)
        for trade in tail[w]:
            m=trade['route_metadata'];ev=m['liquidation_tail_evidence']
            atr_fraction=1.25*m['risk_per_share']/m['estimated_entry_price']
            active=ev['fraction']>atr_fraction
            peer_only=active and ev['symbol_tail_mean']<=atr_fraction
            approved=m['allocation_before_redistribution']+m['allocation_redistributed_qty']
            row={'window':w,**{k:trade[k] for k in ('symbol','entry_decision_time','qty','net_pnl')},
                 'rank':m['candidate_rank'],'tail_active':active,'peer_only_activation':peer_only,
                 'own_complete_all_scenarios':ev['own_incomplete_scenarios']==0,
                 'peer_has_incomplete':ev['peer_incomplete_scenarios']>0,
                 'risk_binding':m['allocation_binding_limit'].endswith('loss_budget'),
                 'own_fraction':ev['symbol_tail_mean'],'peer_fraction':ev['peer_tail_mean'],
                 'blended_fraction':ev['fraction'],'atr_fraction':atr_fraction,
                 'quoted_notional':ev['notional_upper'],'approved_notional':approved*m['estimated_entry_price'],
                 'quote_to_approved_ratio':ev['notional_upper']/(approved*m['estimated_entry_price'])}
            all_tail.append(row)
            c['trades']+=1;c['tail_above_atr']+=active;c['peer_only_above_atr']+=peer_only
            c['peer_only_risk_binding']+=peer_only and row['risk_binding']
            c['own_incomplete']+=not row['own_complete_all_scenarios']
            c['peer_incomplete']+=row['peer_has_incomplete']
            c['tail_active_own_complete_peer_incomplete']+=active and row['own_complete_all_scenarios'] and row['peer_has_incomplete']
        counts[w]=dict(c)
    # Reproduce frozen quotes exactly, then vary only hypothetical requested
    # notional to expose the estimator's size feedback. No execution inference.
    sys.path.insert(0,str(FROZEN))
    from strategy_olr.core.liquidation_tail_v3 import KEY,evidence,scenario
    from strategy_olr.core.opening_tail import tail_mean
    SOURCES[str(FROZEN/'strategy_olr/core/liquidation_tail_v3.py')]=sha(FROZEN/'strategy_olr/core/liquidation_tail_v3.py')
    panel_path=FROZEN/'research/precision_v3/liquidation_observations.json'
    manifest=read(FROZEN.parent/'manifest.json')
    assert sha(panel_path)==manifest['files']['research/precision_v3/liquidation_observations.json']
    panel=read(panel_path);byday=defaultdict(list)
    for r in panel['observations']:byday[r['day']].append(r)
    examples=[]
    targets=[('train','000720','2026-01-16'),('train','001040','2025-09-05'),
             ('oos','034020','2026-04-23'),('oos','000660','2026-04-13'),
             ('oos','042660','2026-05-27'),('oos','403870','2026-06-15')]
    for w,symbol,day in targets:
        trade=next(x for x in tail[w] if x['symbol']==symbol and x['entry_decision_time'].startswith(day))
        m=trade['route_metadata'];ev=m['liquidation_tail_evidence'];through=ev['through']
        days=[d for d in panel['sessions'] if d<=through][-60:]
        h={'version':3,'through':through,'sessions':days,'observations':[r for d in days for r in byday[d]]}
        state=SimpleNamespace(meta={KEY:h});at=datetime.fromisoformat(trade['entry_decision_time'])
        quote=lambda notional:evidence(state,symbol,m['sector'],at,notional,ev['participation'])
        exact=quote(ev['notional_upper']);assert exact==ev,(w,symbol)
        approved=m['allocation_before_redistribution']+m['allocation_redistributed_qty']
        actual=quote(approved*m['estimated_entry_price'])
        peers=[r for r in h['observations'] if r['sector']==m['sector'] and r['symbol']!=symbol]
        if len(peers)<20:peers=[r for r in h['observations'] if r['symbol']!=symbol]
        peer_stats=[]
        for peer in sorted({r['symbol'] for r in peers}):
            px=[scenario(r,ev['notional_upper'],ev['participation']) for r in peers if r['symbol']==peer]
            peer_stats.append({'symbol':peer,'rows':len(px),'incomplete':sum(x['unfilled_shares']>0 for x in px),
                               'tail_mean':tail_mean([x['fraction'] for x in px])})
        curve=[]
        for fraction in (.1,.25,.5,.75,1.):
            q=quote(ev['notional_upper']*fraction)
            curve.append({'requested_notional_fraction':fraction,'own':q['symbol_tail_mean'],'peer':q['peer_tail_mean'],'blend':q['fraction']})
        examples.append({'window':w,'symbol':symbol,'entry':day,'stored_evidence_exact':True,'sector':m['sector'],
             'proposed_quote':ev,'approved_qty':approved,'native_qty':m['allocation_native_qty'],
             'approved_notional':approved*m['estimated_entry_price'],'requote_at_approved_size':actual,
             'atr_fraction':1.25*m['risk_per_share']/m['estimated_entry_price'],'peer_symbols':peer_stats,'curve':curve,
             'interpretation':'Estimator sensitivity only. Requoting approved quantity is not a feasible portfolio or proposed final allocation; joint budgets must be solved self-consistently.'})
    del panel,byday
    # Reconcile actual cash, reservations and portfolio marks for already-valid
    # signals that did not enter. This does not fabricate a fill or trade label.
    prior=read(REPO/'data/backtests/analysis/olr_next_targeted_20261011/diagnosis.json')
    funding=[]
    for w in ('train','oos'):
        base=ROOT/'remaining_drawdown_v2'/w;auth(base)
        snapshot_rel='research/portfolio_current/olr_snapshots.json' if w=='train' else 'research/olr_continuous_history/olr_snapshots.json'
        snapshot_path=FROZEN.parents[2]/w/'inputs/files'/snapshot_rel
        snapshot_manifest=read(FROZEN.parents[2]/w/'inputs/manifest.json')
        assert sha(snapshot_path)==snapshot_manifest['files'][snapshot_rel]
        snapshots=read(snapshot_path)
        candidates={(r['trade_date'],c['symbol']):c for r in snapshots for c in r['candidates']}
        signals=[x for x in prior['windows'][w]['signal_rows'] if x['status']=='signal_but_not_submitted' and not x['already_held_at_signal']]
        targets_by_stamp=defaultdict(list)
        for x in signals:targets_by_stamp[x['signal_time']].append(x)
        cash=1e8;positions=defaultdict(int);marks={};observations={};closing={};latest_entry={}
        journal=base/'native_journal.jsonl.gz';SOURCES[str(journal)]=sha(journal)
        for line in gzip.open(journal,'rt',encoding='utf-8'):
            r=json.loads(line)
            if r['event']=='submit' and r['order']['side']=='BUY':latest_entry[r['order']['symbol']]=r['order']['metadata']
            if r['event']=='fill':
                assert abs(cash-r['cash_before'])<1e-5
                f=r['fill'];signed=f['qty'] if f['side']=='BUY' else -f['qty']
                expected=cash-signed*f['price']-f['metadata']['fee'];assert abs(expected-r['cash_after'])<1e-5
                cash=r['cash_after'];positions[f['symbol']]+=signed
            # Journal mark events contain only symbols observed in this batch;
            # the native broker retains the last mark for a halted/missing name.
            if r['event'] in ('opening_marks','closing_marks'):marks.update(r['prices'])
            if r['event']=='closing_marks':closing[r['timestamp'][:10]]=cash+sum(q*marks[s] for s,q in positions.items() if q)
            if r['event']=='strategy_state':
                reserved=sum(x['reserved_notional'] for x in r['pending_entries'].values())
                nav=cash+sum(q*marks[s] for s,q in r['positions'].items());risk_by_sector=defaultdict(float)
                for s,q in list(r['positions'].items())+[(x['symbol'],x['remaining_qty']) for x in r['pending_entries'].values()]:
                    m=latest_entry[s];risk_by_sector[m['sector']]+=q*m['scenario_loss_budget']['unit_loss']
                observations[r['timestamp']]={'cash':cash,'reserved':reserved,'unreserved_cash':cash-reserved,'positions':r['positions'],'pending_entries':r['pending_entries'],'pending_exits':r['pending_exits'],'prices':dict(marks),
                    'nav':nav,'total_risk_used':sum(risk_by_sector.values()),'risk_by_sector':dict(risk_by_sector)}
        daily=read(base/'daily.json');error=max(abs(closing[x['day']]-x['equity']) for x in daily);assert error<1e-5
        for x in signals:
            state=observations[x['signal_time']];future=[(stamp,v) for stamp,v in observations.items() if stamp[:10]==x['date'] and x['signal_time']<=stamp and stamp[11:16]<='15:10']
            best=max(future,key=lambda z:z[1]['unreserved_cash']) if future else (x['signal_time'],state)
            symbol=x['symbol'];candidate=candidates[x['date'],symbol]
            def headroom(v):return min(.04*v['nav'],.06*v['nav']-v['risk_by_sector'].get(candidate['sector'],0),.08*v['nav']-v['total_risk_used'])
            best_risk=max(future,key=lambda z:headroom(z[1])) if future else (x['signal_time'],state)
            funding.append({'window':w,**x,**{k:state[k] for k in ('cash','reserved','unreserved_cash','positions','pending_entries','pending_exits','nav','total_risk_used','risk_by_sector')},
                'candidate_sector':candidate['sector'],'candidate_daily_atr':candidate['daily_atr'],
                'minimum_unit_risk_before_costs':1.25*candidate['daily_atr'],
                'total_risk_remaining':.08*state['nav']-state['total_risk_used'],
                'sector_risk_remaining':.06*state['nav']-state['risk_by_sector'].get(candidate['sector'],0),
                'minimum_risk_headroom':headroom(state),
                'can_budget_one_share_before_costs':headroom(state)>=1.25*candidate['daily_atr'],
                'max_later_minimum_risk_headroom_before_1510':headroom(best_risk[1]),'max_risk_headroom_time':best_risk[0],
                'mark_price':state['prices'][symbol],'can_afford_one_share_at_mark':state['unreserved_cash']>=state['prices'][symbol],
                'max_later_unreserved_cash_before_1510':best[1]['unreserved_cash'],'max_cash_time':best[0],
                'cash_ledger_reconstruction_max_error':error,
                'interpretation':'Post-decision actual account state, not the unexported rejection reason. Later free cash is observational and requires fresh signal/risk/capacity checks.'})
    control=rows['selected_control'];struct=rows['structural_partial_half']
    peak=struct['windows']['oos']['intraday_drawdown']['peak_nav'];trough=struct['windows']['oos']['intraday_drawdown']['trough_nav']
    dd_target=control['windows']['oos']['drawdown_pct']-.75
    required_trough=peak*(1-dd_target/100)
    gate_gap={'fixed_peak_dd_target_pct':dd_target,'structural_peak':peak,'structural_trough':trough,
        'additional_trough_wealth_needed_if_peak_unchanged':required_trough-trough,
        'equivalent_peak_reduction_if_trough_unchanged':peak-trough/(1-dd_target/100),
        'interpretation':'Arithmetic distance to the registered DD gate, not a proposed trade or return forecast. Native changes can move both endpoints.'}
    marginal={}
    for w in ('train','oos'):
        s=struct['windows'][w];c=rows['risk_plus_late_confirmation']['windows'][w]
        marginal[w]={'adding_late_to_structural_return_delta_pp':c['return_pct']-s['return_pct'],
                     'adding_late_to_structural_dd_delta_pp':c['drawdown_pct']-s['drawdown_pct'],
                     'adding_late_to_structural_trades_delta':c['trades']-s['trades']}
    output={'created_utc':datetime.now(timezone.utc).isoformat(),'native_policy_performance_generated':False,
        'baseline_binding_verified':True,'tail_activation_counts':counts,'tail_admitted_rows':all_tail,
        'estimator_examples':examples,'existing_valid_signal_funding':funding,'drawdown_gate_gap':gate_gap,
        'late_extension_increment_on_structural':marginal,
        'original_funnel':{w:{k:prior['windows'][w]['funnel'][k] for k in ('snapshot_days','empty_days','empty_daily_stage','empty_afternoon_stage','final_candidates','afternoon_sole_reasons')} for w in ('train','oos')},
        'source_sha256':SOURCES,'script_sha256':sha(__file__),
        'limits':['Descriptive estimator and actual-ledger attribution; not counterfactual portfolio performance.',
                  'No candidate identity exclusions, new thresholds, gate changes, or modified frozen inputs.',
                  'IS/OOS repeatedly inspected. New future evidence is needed for independent validation.']}
    (OUT/'diagnosis.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'counts':counts,'gate_gap':gate_gap,'examples':[{'symbol':x['symbol'],'entry':x['entry'],'upper_blend':x['proposed_quote']['fraction'],'requote_blend':x['requote_at_approved_size']['fraction'],'peers':x['peer_symbols']} for x in examples],
       'oos_funding':[{k:x[k] for k in ('date','symbol','unreserved_cash','mark_price','max_later_unreserved_cash_before_1510','max_cash_time')} for x in funding if x['window']=='oos']},indent=2),flush=True)

if __name__=='__main__':main()

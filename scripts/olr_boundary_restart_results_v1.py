"""Authenticate restart evidence and compare the two declared history policies."""
from __future__ import annotations
from collections import Counter
import copy
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import olr_boundary_restart_study_v1 as b
s=b.s;n=b.n


def audit(c,w,scenario='baseline',capped=False):
    f=b.output(c,w,scenario,capped);proof=s.forensic.authenticate(f)
    if c['history_policy']=='literal_carry' and w=='oos':
        previous=b.output(c,'train',scenario,capped);parent=n.read(previous/'capacity_terminal_state.json')
        actual=n.read(f/'capacity_restart_receipt.json')
        assert actual['parent_checkpoint_sha256']==n.sha(previous/'capacity_terminal_state.json')
        assert actual['parent_completion_sha256']==n.sha(previous/'completion.json')
        assert actual['restored_state']==parent['capacity_history_v1']
        assert actual['state_hash']==n.sig(parent['capacity_history_v1'])
        assert actual['through']=='2026-03-31' and actual['restart_date']=='2026-04-01'
        assert actual['all_samples_verified_against_oos_retained_ohlcv'] and actual['same_configuration_and_scenario']
        proof.update(literal_is_terminal_state_restored=True,verified_prior_observations=actual['verified_observations'],
            first_and_last_retained_timestamps=[min(t for hist in actual['restored_state'].values() for rows in hist.values() for t,v in rows),
                max(t for hist in actual['restored_state'].values() for rows in hist.values() for t,v in rows)])
    elif c['history_policy']=='broad_declared_universe':
        actual=n.read(f/'capacity_history_inputs.json');expected=n.read(b.ROOT/f'boundary_validation_{w}.json')
        assert actual['seeded_state_hash']==expected['seeded_state_hash']
        assert actual['seeded_state']==expected['seeded_state']
        assert actual['seed_rows']==expected['prior_auction_rows']
        assert len(actual['seed_days'])==expected['prior_sessions']
        proof.update(broad_restart_equals_uninterrupted_stream=True,prior_sessions=len(actual['seed_days']),
            prior_observations=actual['seed_rows'],universe_symbols=len(actual['seeded_state']))
    trades=n.read(f/'trades.json')
    proof['entry_auction_sample_counts']=dict(Counter(len(t['route_metadata']['capacity_forecast']['exit']['samples']) for t in trades))
    return proof


def pairs():
    original={w:n.read(s.PARENT/'cases'/s.forensic.BASE/w/'100000000_baseline/assessment.json') for w in ('train','oos')};rows=[]
    for c in n.read(b.ROOT/'candidate_plan.json')['candidates']:
        row={'candidate_id':c['id'],'label':c['label'],'history_policy':c['history_policy'],'mutations':c['mutations']}
        for w in ('train','oos'):
            f=b.output(c,w)
            if not (f/'assessment.json').exists():break
            a=n.read(f/'assessment.json')
            if not a.get('qualified'):break
            proof=audit(c,w)
            row[w]={k:a[k] for k in ('return_pct','drawdown_pct','trades')}
            row[w].update(return_gain_pp=a['return_pct']-original[w]['return_pct'],
                drawdown_change_pp=a['drawdown_pct']-original[w]['drawdown_pct'],folder=str(f),authentication=proof)
        if not all(w in row for w in ('train','oos')):continue
        row['both_returns_improved']=all(row[w]['return_gain_pp']>=0 for w in ('train','oos'))
        row['return_gate']=row['train']['return_gain_pp']>=-3 and row['oos']['return_gain_pp']>=2 and row['oos']['return_pct']>0
        row['panel_corrected']=bool(c['mutations'].get(b.x.PANEL_KEY))
        row['performance_eligible']=row['return_gate'] and all(row[w]['drawdown_change_pp']<=2 for w in ('train','oos'))
        row['eligible']=row['performance_eligible'] and row['panel_corrected']
        rows.append(row)
    # Preserve the economic gates. An optional scoring-formula repair is not
    # allowed to make the mandatory history handoff contingent on accepting a
    # material IS deterioration. This fallback is explicitly registered.
    amendment=b.ROOT/'feature_adoption_amendment.json'
    if amendment.exists() and not any(v['eligible'] for v in rows):
        for v in rows:
            if v['history_policy']=='literal_carry' and v['performance_eligible']:
                v['eligible']=True
                v['compatibility_scoring_fallback']=True
    n.write(s.OUT/'boundary_restart_paired_results.json',rows)
    # Checkpoint-only source changes must preserve IS economics exactly.
    for label,oldroot,oldid in (('literal_carry_original',s.PARENT,s.forensic.BASE),('literal_carry_guard_1p0',s.ROOT,'c8d12020033107bb0')):
        c=next(c for c in n.read(b.ROOT/'candidate_plan.json')['candidates'] if c['label']==label)
        new=b.output(c,'train');old=oldroot/'cases'/oldid/'train/100000000_baseline'
        if not (new/'assessment.json').exists():continue
        s.forensic.authenticate(new)
        fields=('symbol','qty','entry_decision_time','entry_fill_time','entry_price','exit_fill_time','exit_price','gross_pnl','commission','net_pnl','exit_reason','mfe','mae','r_multiple')
        economic=lambda p:[{k:t.get(k) for k in fields} for t in n.read(p/'trades.json')]
        assert economic(new)==economic(old) and n.read(new/'daily.json')==n.read(old/'daily.json') and n.read(new/'equity.json')==n.read(old/'equity.json')
        n.write(s.OUT/(label+'_is_exact_parity.json'),{'pass':True,'original':str(old),'new':str(new),
            'trade_economics_exact':True,'every_daily_balance_exact':True,'every_intraday_mark_exact':True})
    return rows


def matched_control():
    plan=n.read(b.ROOT/'candidate_plan.json')
    existing=next((c for c in plan['candidates'] if c['label']=='literal_carry_full_panel'),None)
    if existing:return existing
    m=copy.deepcopy(plan['baseline']);m[b.x.PANEL_KEY]=True;key=n.sig(m);policy='literal_carry'
    c={'id':'c'+n.sig({'mutations':m,'history_policy':policy})[:16],'label':'literal_carry_full_panel','history_policy':policy,
        'mutations':m,'mutations_hash':key,'aliases':[],'kind':'matched_history_control',
        'paths':[list(p) for p,_,_ in n.mutation_edits(plan['baseline'],m)]}
    n.write(b.ROOT/'matched_panel_control_registration.json',{'registered_utc':s.utc(),'candidate':c,
        'purpose':'Complete the two-by-two panel/guard attribution if the literal full-panel guard is a finalist; no new tuned parameter.'})
    for name in ('candidate_plan.json','mechanism_plan.json'):
        p=n.read(b.ROOT/name);p['candidates'].append(c);n.write(b.ROOT/name,p)
    return c


if __name__=='__main__':
    rows=pairs();print([{k:r[k] for k in ('label','train','oos','eligible')} for r in rows])

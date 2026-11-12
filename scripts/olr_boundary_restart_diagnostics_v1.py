"""Authenticated calendar, payoff, and uncertainty diagnostics; no optimization."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import olr_boundary_restart_results_v1 as r
b=r.b;s=b.s;n=b.n


def describe(folder):
    receipt=s.forensic.authenticate(folder)
    result=n.read(folder/'result.json');daily=n.read(folder/'daily.json');trades=n.read(folder/'trades.json')
    spec=importlib.util.spec_from_file_location('olr_payoff_accounting',REPO/'data/backtests/analysis/olr_exit_diagnosis_20261010/analyze.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    summary=module.summary(trades)
    months={};previous=100_000_000.
    for row in daily:
        month=row['day'][:7]
        if month not in months:months[month]={'start_equity':previous,'first_day':row['day'],'sessions':0}
        months[month].update(last_day=row['day'],end_equity=row['equity'])
        months[month]['sessions']+=1;previous=row['equity']
    for key,v in months.items():
        v.update(return_pct=100*(v['end_equity']/v['start_equity']-1),pnl=v['end_equity']-v['start_equity'],
            entered_trades=sum(t['entry_fill_time'].startswith(key) for t in trades))
    return {'result':{k:result[k] for k in ('return_pct','drawdown_pct','trades','sessions','audit_status')},
        'authentication':receipt,'payoff':summary,'calendar':months,'folder':str(folder)}


def main():
    out={'asof_utc':s.utc(),'original':{},'candidates':{},'comparisons':{},
         'uncertainty_method':'10,000 paired circular daily block bootstrap samples; fixed seed 20261010; blocks 1/3/5/10 sessions. Descriptive conditional sensitivity, not new independent data or a multiple-testing adjustment.'}
    original={w:s.PARENT/'cases'/s.forensic.BASE/w/'100000000_baseline' for w in ('train','oos')}
    for w,p in original.items():out['original'][w]=describe(p)
    plan=n.read(b.ROOT/'candidate_plan.json');catalog={c['id']:c for c in plan['candidates']}
    out['existing_1p2_stress_reference']={
        'basis':'Earlier original-history source generation; direct authenticated evidence, not relabeled as a new literal-restart OOS result.',
        **{w:describe(s.PRIOR/'cases/c272fe0390df855bd'/w/'100000000_combined') for w in ('train','oos')}}
    out['additional_is_exact_parity']={}
    fields=('symbol','qty','entry_decision_time','entry_fill_time','entry_price','exit_fill_time','exit_price','gross_pnl','commission','net_pnl','exit_reason','mfe','mae','r_multiple')
    for label,oldid in (('literal_carry_original_scoring_guard_0.8','cbd14aff8c9557723'),('literal_carry_original_scoring_guard_1.2','c272fe0390df855bd')):
        c=next((v for v in catalog.values() if v['label']==label),None)
        if c is None or not (b.output(c,'train')/'assessment.json').exists():continue
        new=b.output(c,'train');old=s.PRIOR/'cases'/oldid/'train/100000000_baseline'
        s.forensic.authenticate(new);s.forensic.authenticate(old)
        economic=lambda folder:[{k:t.get(k) for k in fields} for t in n.read(folder/'trades.json')]
        assert economic(old)==economic(new) and n.read(old/'daily.json')==n.read(new/'daily.json') and n.read(old/'equity.json')==n.read(new/'equity.json')
        out['additional_is_exact_parity'][label]={'pass':True,'original':str(old),'new':str(new),'every_trade_daily_balance_and_intraday_mark_equal':True}
    selected=n.read(s.OUT/'boundary_restart_qualification.json').get('selected') if (s.OUT/'boundary_restart_qualification.json').exists() else None
    for c in catalog.values():
        windows={}
        for scenario,capped in (('baseline',False),('combined',False),('baseline',True)):
            for w in ('train','oos'):
                folder=b.output(c,w,scenario,capped)
                if not (folder/'assessment.json').exists() or not n.read(folder/'assessment.json').get('qualified'):continue
                windows.setdefault(scenario+('_capped' if capped else ''),{})[w]=describe(folder)
        if windows:out['candidates'][c['id']]={'label':c['label'],'history_policy':c['history_policy'],'windows':windows}
    # Bootstrap only the eventual finalist plus the main 1R corrected-panel
    # policy, not every attempted configuration as a significance ranking.
    targets={c['id'] for c in catalog.values() if c['label'] in ('literal_carry_guard_1p0','literal_carry_full_panel_guard_1p0')}
    if selected:targets.add(selected['candidate_id'])
    for cid in targets:
        c=catalog[cid]
        for w in ('train','oos'):
            target=b.output(c,w)
            if not (target/'assessment.json').exists():continue
            refs={'original':original[w]}
            control_label=('literal_carry_full_panel' if c['mutations'].get(b.x.PANEL_KEY) else 'literal_carry_original') if c['history_policy']=='literal_carry' else 'broad_declared_universe_full_panel'
            matched=next((v for v in catalog.values() if v['label']==control_label),None)
            if matched and (b.output(matched,w)/'assessment.json').exists():refs['matched_continuous_control']=b.output(matched,w)
            for name,p in refs.items():
                out['comparisons'].setdefault(cid,{}).setdefault(w,{})[name]={
                    'paired_block_bootstrap':s.forensic.bootstrap(n.read(p/'daily.json'),n.read(target/'daily.json')),
                    'issuer_day_attribution':s.forensic.contribution(n.read(p/'trades.json'),n.read(target/'trades.json'))}
    n.write(s.OUT/'boundary_restart_diagnostics.json',out)
    print({'candidates':len(out['candidates']),'selected':selected['candidate_id'] if selected else None,'output':str(s.OUT/'boundary_restart_diagnostics.json')})


if __name__=='__main__':main()

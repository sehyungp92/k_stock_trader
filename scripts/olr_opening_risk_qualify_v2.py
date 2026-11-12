"""Serial, resumable qualification of the registered opening-risk mechanisms."""
from __future__ import annotations
import copy
from pathlib import Path
import traceback
from scripts import olr_opening_risk_study_v2 as t
from scripts import olr_opening_risk_prepare_v2 as preparation
from scripts import olr_targeted_risk_results_v1 as metrics
from scripts import olr_boundary_restart_results_v1 as history
n,s,b=t.n,t.s,t.b
WINDOWS=('train','oos')


def audit(c,w,scenario='baseline',capped=False):
    proof=history.audit(c,w,scenario,capped)
    folder=b.output(c,w,scenario,capped)
    trades=n.read(folder/'trades.json')
    assert len(trades)==len({(x['symbol'],x['entry_decision_time']) for x in trades}), 'Partial exit legs must collapse into unique entries'
    if c['mutations']['capacity_opening_tail']:
        receipt=n.read(folder/'opening_tail_restart_receipt.json')
        assert receipt['strictly_prior_sessions_only'] and receipt['independent_source_prefix_equal']
        if w=='oos':
            parent=n.read(b.output(c,'train',scenario,capped)/'capacity_terminal_state.json')
            assert receipt['actual_is_terminal_restored']
            assert receipt['state_hash']==n.sig(parent['opening_tail_history_v1'])
            assert receipt['through']=='2026-03-31' and receipt['restart_date']=='2026-04-01'
        proof['opening_tail_restart']=receipt
        for trade in trades:
            ev=trade['route_metadata']['opening_tail_evidence']
            assert ev['through']<trade['entry_decision_time'][:10]
    return proof


def evaluate(c,w,scenario='baseline',capped=False):
    if (t.ROOT/'STOP.json').exists():raise RuntimeError('Registered study STOP before next replay')
    folder=b.output(c,w,scenario,capped)
    # An interrupted outer collector can leave fully authenticated native output.
    # Recover its assessment without rerunning or modifying any native artifact.
    if (folder/'completion.json').exists() and not (folder/'assessment.json').exists():
        s.forensic.authenticate(folder)
        a=n.read(folder/'result.json');native=n.read(folder/'execution_audit.json')
        assert a['execution_validity_pass'] and native['audit_pass'] and a['audit_status']=='passed'
        assert not a['terminal_inventory'] and not a['terminal_pending']
        runtime=t.ROOT/'runtime'/c['id']/w/(scenario+('_capped' if capped else ''))/'study.json'
        n.verify_window_contract(t.ROOT,w,n.read(runtime),folder)
        a.update(candidate_id=c['id'],label=c['label'],window=w,capital=100_000_000,capped=capped,
            history_policy=c['history_policy'],output=str(folder),qualified=True)
        n.write(folder/'assessment.json',a)
    result=b.evaluate(c,w,scenario,capped)
    audit(c,w,scenario,capped)
    if c['label']=='selected_control':preparation.parity(c,w,scenario,capped)
    return result


def reference_data():
    old=n.read(t.previous.OUT/'selected_research_policy.json')
    assert old['results_sha256']==n.sha(t.previous.OUT/'results.json')
    data=n.read(t.previous.OUT/'results.json')
    guard=next(r for r in data['candidates'] if r['label']=='guard_control')
    return {'original':data['original_saved'],'original_stress':data['original_stress'],
            'guard':guard,'prior_results_sha256':old['results_sha256']}


def gates(a,z,control,refs,stress=False):
    g=n.read(t.ROOT/'registration.json')['gates']
    group='stress' if stress else 'windows'
    guard=refs['guard'][group]['train'];original=refs['original_stress' if stress else 'original']['train']
    ca,cz=control[group]['train'],control[group]['oos']
    tests={'is_original_return':a['return_pct']>=original['return_pct']-g['is_return_loss_max_pp_original'],
        'is_guard_return':a['return_pct']>=guard['return_pct']-g['is_return_loss_max_pp_guard'],
        'is_selected_return':a['return_pct']>=ca['return_pct']-g['is_return_loss_max_pp_selected'],
        'is_guard_drawdown':a['drawdown_pct']<=guard['drawdown_pct']+g['is_dd_increase_max_pp_guard']}
    if stress:
        tests.update(oos_return=z['return_pct']>=cz['return_pct'],oos_drawdown=z['drawdown_pct']<cz['drawdown_pct'])
    else:
        tests.update(oos_return=z['return_pct']>=cz['return_pct']+g['oos_return_improvement_min_pp_selected'],
            oos_drawdown=z['drawdown_pct']<=cz['drawdown_pct']-g['oos_dd_reduction_min_pp_selected'],
            is_frequency=a['trades']>=g['is_trades_min'],oos_frequency=z['trades']>=g['oos_trades_min'])
    return tests


def collect():
    t.activate();catalog=n.read(t.ROOT/'candidate_plan.json')['candidates'];refs=reference_data()
    rows=[]
    for c in catalog:
        row={'id':c['id'],'label':c['label'],'mutations':c['mutations'],'windows':{},'stress':{},'capped':{}}
        for scenario,capped,group in [('baseline',False,'windows'),('combined',False,'stress'),('baseline',True,'capped')]:
            for w in WINDOWS:
                folder=b.output(c,w,scenario,capped)
                if (folder/'assessment.json').exists():
                    row[group][w]={**metrics.describe(folder),'authentication':audit(c,w,scenario,capped)}
        rows.append(row)
    control=rows[0]
    for row in rows:
        if len(row['windows'])==len(control['windows'])==2:
            row['gates']=gates(row['windows']['train'],row['windows']['oos'],control,refs)
            row['nominal_qualified']=all(row['gates'].values())
            row['trade_comparison']={w:metrics.matched(Path(control['windows'][w]['folder']),Path(row['windows'][w]['folder'])) for w in WINDOWS}
        if len(row['stress'])==len(control['stress'])==2:
            row['stress_gates']=gates(row['stress']['train'],row['stress']['oos'],control,refs,True)
            row['stress_qualified']=all(row['stress_gates'].values())
    return {'registration_sha256':n.sha(t.ROOT/'registration.json'),'references':refs,'candidates':rows,
            'independent_validation':False,'promotion':False}


def frequency(row):
    a,z=row['windows']['train'],row['windows']['oos']
    return (min(a['trades']/186,z['trades']/46),a['trades']/186+z['trades']/46)


def rank(row):
    z=row['windows']['oos'];return(z['return_pct']/z['drawdown_pct'],*frequency(row),row['windows']['train']['return_pct'])


def combination(data):
    eligible=sorted((r for r in data['candidates'][1:4] if r.get('nominal_qualified')),key=rank,reverse=True)
    receipt=t.ROOT/'combination_selection.json'
    ids=[r['id'] for r in eligible[:2]] if len(eligible)>=2 else []
    if receipt.exists():assert n.read(receipt)['parents']==ids
    else:n.write(receipt,{'utc':s.utc(),'parents':ids,'consumed_nominal_results_hash':n.sig(data),
        'rule':'Union of at most two individually nominal-qualified mechanisms; no new thresholds.'})
    if not ids:return None
    plan=n.read(t.ROOT/'candidate_plan.json');base=plan['baseline'];mutations=copy.deepcopy(base)
    for row in eligible[:2]:
        for path,value,deleted in n.mutation_edits(base,row['mutations']):
            assert len(path)==1 and not deleted
            assert mutations[path[0]]==base[path[0]] or mutations[path[0]]==value
            mutations[path[0]]=value
    c={'id':'c'+n.sig({'version':'opening_risk_v2','mutations':mutations})[:16],
        'label':'qualified_pair_union','history_policy':'literal_carry','mutations':mutations,
        'mutations_hash':n.sig(mutations),'aliases':[],'kind':'opening_risk_v2',
        'paths':[list(p) for p,_,_ in n.mutation_edits(base,mutations)]}
    if c['id'] not in {x['id'] for x in plan['candidates']}:
        for name in ('candidate_plan.json','mechanism_plan.json'):
            p=n.read(t.ROOT/name);p['candidates'].append(c);n.write(t.ROOT/name,p)
    return c


def qualify():
    preparation.prepare();t.activate()
    catalog=n.read(t.ROOT/'candidate_plan.json')['candidates'];control=catalog[0]
    for c in catalog[:4]:
        for w in WINDOWS:evaluate(c,w)
    data=collect();combined=combination(data)
    if combined:
        for w in WINDOWS:evaluate(combined,w)
        data=collect()
    eligible=[r for r in data['candidates'][1:] if r.get('nominal_qualified')]
    leaders=list({r['id']:r for r in ([max(eligible,key=rank),max(eligible,key=lambda r:(frequency(r),rank(r)))] if eligible else [])}.values())
    ids=[r['id'] for r in leaders];receipt=t.ROOT/'stress_selection.json'
    if receipt.exists():assert n.read(receipt)['finalists']==ids
    else:n.write(receipt,{'utc':s.utc(),'finalists':ids,'consumed_nominal_results_hash':n.sig(data),
        'rule':'Stress control and at most two nominal-qualified objective leaders: return/DD and balanced frequency.'})
    catalog={c['id']:c for c in n.read(t.ROOT/'candidate_plan.json')['candidates']}
    for c in [control,*[catalog[cid] for cid in ids]]:
        for w in WINDOWS:evaluate(c,w,'combined')
    data=collect();qualified=[r for r in data['candidates'][1:] if r.get('nominal_qualified') and r.get('stress_qualified')]
    selected=max(qualified,key=rank) if qualified else data['candidates'][0]
    for cid in dict.fromkeys([control['id'],selected['id']]):
        for w in WINDOWS:evaluate(catalog[cid],w,'baseline',True)
    data=collect()
    n.write(t.OUT/'results.json',data)
    selected=next(r for r in data['candidates'] if r['id']==selected['id'])
    profile={'status':'qualified_research_successor' if qualified else 'retain_selected_no_qualified_upgrade',
      'selected_id':selected['id'],'selected_label':selected['label'],'mutations':catalog[selected['id']]['mutations'],
      'results_sha256':n.sha(t.OUT/'results.json'),'registration_sha256':n.sha(t.ROOT/'registration.json'),
      'source_generation_hashes':n.read(t.ROOT/'preparation_complete.json')['sources'],
      'history_policy':'literal_carry','promotion':False,'independent_validation':False,
      'scope':'Bounded, development-only native comparisons. Retain the current selected profile if no tested successor meets every gate.'}
    n.write(t.OUT/'selected_research_policy.json',profile)
    n.write(t.OUT/'qualified_alternatives.json',{'policies':[{'candidate_id':r['id'],'label':r['label'],
       'mutations':catalog[r['id']]['mutations'],'nominal':r['windows'],'stress':r['stress']} for r in qualified]})
    n.write(t.ROOT/'qualification_complete.json',{'utc':s.utc(),'selected_id':selected['id'],
        'results_sha256':n.sha(t.OUT/'results.json'),'profile_sha256':n.sha(t.OUT/'selected_research_policy.json')})
    print(profile,flush=True)


if __name__=='__main__':
    try:qualify()
    except Exception:
        n.write(t.ROOT/'qualification_failure.json',{'utc':s.utc(),'traceback':traceback.format_exc()})
        raise

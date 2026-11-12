"""Serial qualification with frozen rules and authenticated restart recovery."""
import copy
from pathlib import Path
import traceback
from scripts import olr_precision_study_v3 as t
from scripts import olr_precision_prepare_v3 as preparation
from scripts import olr_opening_risk_qualify_v2 as inherited
n,s,b=t.n,t.s,t.b
_audit=inherited.audit
_gates=inherited.gates

def reference_data():
    profile=n.read(t.previous.OUT/'selected_research_policy.json')
    assert profile['results_sha256']==n.sha(t.previous.OUT/'results.json')
    return n.read(t.previous.OUT/'results.json')['references']

def audit(c,w,scenario='baseline',capped=False):
    proof=_audit(c,w,scenario,capped)
    if c['mutations']['capacity_liquidation_tail']:
        folder=b.output(c,w,scenario,capped);receipt=n.read(folder/'liquidation_restart_receipt.json')
        assert receipt['strictly_prior_sessions_only'] and receipt['independent_source_prefix_equal']
        if w=='oos':
            parent=n.read(b.output(c,'train',scenario,capped)/'capacity_terminal_state.json')
            assert receipt['actual_is_terminal_restored'] and receipt['through']=='2026-03-31'
            assert receipt['state_hash']==n.sig(parent['liquidation_tail_history_v3'])
        proof['liquidation_restart']=receipt
        for trade in n.read(folder/'trades.json'):
            ev=trade['route_metadata']['liquidation_tail_evidence']
            assert ev['through']<trade['entry_decision_time'][:10]
            # Real filled price may differ from buffered estimate; verify request
            # upper envelope against approved quantity at the actual decision.
            meta=trade['route_metadata']
            approved=meta.get('allocation_before_redistribution',trade['qty'])+meta.get('allocation_redistributed_qty',0)
            assert approved*meta['estimated_entry_price']<=ev['notional_upper']+1e-5
    return proof

def gates(a,z,control,refs,stress=False):
    out=_gates(a,z,control,refs,stress)
    if stress:
        out.update(is_frequency=a['trades']>=control['stress']['train']['trades'],
                   oos_frequency=z['trades']>=control['stress']['oos']['trades'])
    else:
        g=n.read(t.ROOT/'registration.json')['gates']
        out.update(is_entry_days=a['entry_days']>=g['is_entry_days_min'],oos_entry_days=z['entry_days']>=g['oos_entry_days_min'])
    return out

def economics(folder):
    keys=('symbol','qty','entry_decision_time','entry_fill_time','entry_price','exit_fill_time','exit_price','net_pnl')
    return [{k:r[k] for k in keys} for r in n.read(folder/'trades.json')]

def combination(data):
    control=data['candidates'][0];risks=[]
    for row in data['candidates'][1:3]:
        active=any(economics(Path(row['windows'][w]['folder']))!=economics(Path(control['windows'][w]['folder'])) for w in ('train','oos'))
        if active:risks.append(row)
    def score(row):
        preserve=all(v for k,v in row['gates'].items() if k in ('is_original_return','is_guard_return','is_selected_return','is_guard_drawdown'))
        return (preserve,control['windows']['oos']['drawdown_pct']-row['windows']['oos']['drawdown_pct'],
                row['windows']['train']['return_pct'],row['windows']['oos']['return_pct'])
    risk=max(risks,key=score) if risks else None
    entry=next(r for r in data['candidates'] if r['label']=='late_confirmation_9')
    ids=[risk['id'],entry['id']] if risk else []
    receipt=t.ROOT/'combination_selection.json'
    if receipt.exists():assert n.read(receipt)['parents']==ids
    else:n.write(receipt,{'utc':s.utc(),'parents':ids,'active_risk_candidates':[r['id'] for r in risks],
                         'consumed_nominal_results_hash':n.sig(data),'rule':n.read(t.ROOT/'registration.json')['gates']['combination']})
    if not risk:return None
    plan=n.read(t.ROOT/'candidate_plan.json');base=plan['baseline'];changes={}
    for row in (risk,entry):
        for path,value,deleted in n.mutation_edits(base,row['mutations']):
            assert len(path)==1 and not deleted
            if path[0] in changes:assert changes[path[0]]==value
            changes[path[0]]=value
    c=t.candidate('risk_plus_late_confirmation',base,changes)
    if c['id'] not in {r['id'] for r in plan['candidates']}:
        for name in ('candidate_plan.json','mechanism_plan.json'):
            p=n.read(t.ROOT/name);p['candidates'].append(c);n.write(t.ROOT/name,p)
    return c

def configure():
    inherited.t=t;inherited.preparation=preparation
    inherited.audit=audit;inherited.gates=gates;inherited.reference_data=reference_data;inherited.combination=combination
    t.activate()

def collect():configure();return inherited.collect()
def rank(row):return inherited.rank(row)

def qualify():
    configure()
    receipt=n.read(t.OUT/'test_receipt.json')
    assert receipt['passed']>=50 and receipt['failures']==0 and receipt['skipped']==0 and receipt['exit_code']==0
    assert len(receipt['frozen_sources'])==2*len(t.PATCHED)
    for name,expected in receipt['test_hashes'].items():assert n.sha(t.REPO/name)==expected,name
    for name,expected in receipt['frozen_sources'].items():assert n.sha(Path(name))==expected,name
    inherited.qualify()

if __name__=='__main__':
    try:qualify()
    except Exception:
        n.write(t.ROOT/'qualification_failure.json',{'utc':s.utc(),'traceback':traceback.format_exc()});raise

"""Independent cash/quantity/chronology/normal-equation review of V43."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import argparse,gzip,hashlib,json,math,statistics

ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(x) for x in f]
def save(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,allow_nan=False)
def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-6):raise ValueError(f'Mismatch {a} != {b}')

def main(source,out):
    done=read(source/'completion.json');assert done['complete']
    protocol=read(source/'protocol.json');bindings=protocol['bindings']+protocol['programs']+done['outputs']
    for b in bindings:assert sha(b['path'])==b['sha256'],b['path']
    out.mkdir(exist_ok=False)
    save(out/'protocol.json',{'scope':'Independent reconstruction from original fills/orders, paired maturity, exact session support, fitted normal equations and every forecast. No fits are selected or changed.',
         'source_completion':{'path':str(source/'completion.json'),'sha256':sha(source/'completion.json')},'program':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)}})
    targets=rows(source/'targets.jsonl.gz');lookup={(r['parent'],r['scenario'],r['id']):r for r in targets}
    assert len(lookup)==len(targets)
    cash_checks=0;zero_checks=0;fee_total=0.;accounts=[]
    for parent in ('control','control_prior125'):
      for period in ('is','oos'):
       for scenario in ('baseline','combined'):
        folder=ROOT/f'v25_pending_funding_screen/cases/{parent}/{period}/{scenario}'
        fills=read(folder/'fills.json');exposures=read(folder/'observer_evidence.json')['entry_exposures']
        exposed={e['order_id']:e for e in exposures};active={};lifetimes=[];seen=set()
        for f in fills:
            eid=f['metadata']['execution_id'];assert eid not in seen;seen.add(eid)
            symbol=f['symbol'];qty=f['qty'];fee=f['metadata']['fee'];fee_total+=fee
            assert qty>0 and int(qty)==qty
            if f['side']=='BUY':
                if symbol not in active:
                    e=exposed[f['order_id']]
                    active[symbol]={'oid':f['order_id'],'qty':0,'live':0,'net':0.,'route':e['action']['metadata']['entry_route']}
                p=active[symbol];p['qty']+=qty;p['live']+=qty;p['net']-=qty*f['price']+fee
            else:
                p=active[symbol];assert qty<=p['live'];p['live']-=qty;p['net']+=qty*f['price']-fee;p['exit']=f['timestamp']
                if not p['live']:lifetimes.append(p);del active[symbol]
        assert not active
        lifetime={p['oid']:p for p in lifetimes if p['route']=='first30_open_anchor'}
        wanted=[r for r in targets if r['parent']==parent and r['period']==period and r['scenario']==scenario]
        assert len(wanted)==sum(e['action']['metadata']['entry_route']=='first30_open_anchor' for e in exposures)
        seen_orders=set()
        for r in wanted:
            t=r['native'];oid=t['order_id'];assert oid not in seen_orders;seen_orders.add(oid)
            e=exposed[oid];den=e['action']['qty']*e['action']['metadata']['entry_price_ref']
            close(den,t['planned_notional_krw']);assert t['submitted_quantity']==e['action']['qty']
            p=lifetime.get(oid)
            if p:
                close(p['net'],t['net_pnl_krw']);close(p['net']/den*10000,r['funded']);assert p['qty']==t['filled_quantity']
                assert datetime.fromisoformat(r['mature_at'])>=datetime.fromisoformat(p['exit']);cash_checks+=1
            else:
                assert r['funded']==0 and t['net_pnl_krw']==0 and t['filled_quantity']==0;zero_checks+=1
        accounts.append({'parent':parent,'period':period,'scenario':scenario,'decisions':len(wanted),'lifetimes':len(lifetime),'all_cash_quantity_targets_match':True})
    calendar=read(ROOT.parent/'kalcb_aligned_walk_forward_20261010/v3/fit/calendar.json')['sessions']
    models=read(source/'models.json');forecast=read(source/'forecasts.json');normal_checks=0
    models_by_key={}
    for p in models:
        parent=p['parent'];scenario=p['scenario'];first=p['first'];target=p['target'];models_by_key[parent,scenario,target,first]=p
        allowed=set(calendar[:max(0,calendar.index(first)-1)]);cutoff=datetime.fromisoformat(first+'T00:00:00+09:00')
        history=[r for r in targets if r['parent']==parent and r['scenario']==scenario and r['day'] in allowed and r['target_valid'] and datetime.fromisoformat(r['mature_at'])<cutoff]
        assert p['mature_sessions']==len({r['day'] for r in history}) and p['issuer_days']==len(history)
        if p['cells']:assert p['mature_sessions']>=60 and p['issuer_days']>=100
        for key,c in p['cells'].items():
            rs=[r for r in history if f"{r['origin']}/q{r['quantity_scale']:g}"==key]
            assert [r['id'] for r in sorted(rs,key=lambda r:(r['day'],r['id']))]==c['training_ids']
            counts=defaultdict(int)
            for r in rs:counts[r['day']]+=1
            assert len(rs)>=20 and len(counts)>=8 and len(rs)==c['observations'] and len(counts)==c['sessions']
            weight=[len(rs)/len(counts)/counts[r['day']] for r in rs]
            mean=sum(w*r['prediction'] for w,r in zip(weight,rs))/sum(weight)
            scale=max(1.,math.sqrt(sum(w*(r['prediction']-mean)**2 for w,r in zip(weight,rs))/sum(weight)))
            close(mean,c['mean']);close(scale,c['scale'])
            intercept=c['intercept_standardized'];slope=c['slope_delta'];assert -1<=slope<=0
            gi=100*intercept;gs=100*slope
            for w,r in zip(weight,rs):
                z=(r['prediction']-mean)/scale;res=(r[target]-r['prediction'])/scale
                err=intercept+slope*z-res;gi+=w*err;gs+=w*z*err
            close(gi,0)
            if -1<slope<0:close(gs,0)
            elif slope==0:assert gs<=1e-6
            else:assert gs>=-1e-6
            normal_checks+=1
    for f in forecast:
        r=lookup[f['parent'],f['scenario'],f['id']];p=models_by_key[f['parent'],f['scenario'],f['target'],f['first']]
        c=p['cells'].get(f"{r['origin']}/q{r['quantity_scale']:g}");x=r['prediction']
        support=c is not None and c['min']<=x<=c['max'];assert f['supported']==support
        expected=x+c['scale']*c['intercept_standardized']+c['slope_delta']*(x-c['mean']) if support else x
        close(expected,f['value'])
    # Independent reconciliation of V42 final forecasts and original vetoes.
    rec=rows(ROOT/'v42_decision_value_reconciliation_v2/reconciled_decisions.jsonl.gz')
    old=ROOT.parent/'kalcb_selective_recovery_20261010';proofs=read(old/'v6_qualification/build/proofs.json')
    others={r['action_id']:r for r in rows(old/'v2/build/decision_audit.jsonl.gz') if r['family']=='decay60'}
    v26={r['action_id']:r for r in rows(ROOT/'v26_opening_decision_audit_v2/decisions.jsonl.gz')}
    for x in rec:
        v=x['value'];r=v26[v['action_id']];o=others[v['action_id']]
        p=next(p for p in proofs['100.0' if x['parent']=='control' else '125.0'] if p['valid_from']==v['calibration_valid_from'])
        field='calibrated_core' if v['origin']=='frozen_core' else 'calibrated_support'
        own=int(min(r[field].values())>0);other=int(min(o[field].values())>0)
        key=f"{v['origin']}/q{v['scale']:g}/self{own}_other{other}/{r['descriptor']['tier']}"
        support=all(p['cohorts'][s+'/'+key]['supported'] for s in ('baseline','combined'))
        values={s:r[field][s]+p['cohorts'][s+'/'+key]['offset_bps'] for s in ('baseline','combined')}
        for s in values:close(values[s],v['final'][s])
        assert v['supported']==support and v['retained']==(not(support and min(values.values())<=0))
    for b in bindings:assert sha(b['path'])==b['sha256'],b['path']
    save(out/'review.json',{'passed':True,'accounts':accounts,'independent_cash_lifetimes':cash_checks,'retained_zero_fill_targets':zero_checks,
        'all_fill_fees_checked_krw':fee_total,'models':len(models),'supported_cell_normal_equations':normal_checks,
        'forecast_values':len(forecast),'v42_final_decisions_independently_reconstructed':len(rec),
        'scope':'No original lifetime net is asserted to be recovered portfolio PnL. No fresh native source/MTM audit claimed; authenticated V25 audit reused.'})
    save(out/'completion.json',{'complete':True,'study_complete':False,'review_sha256':sha(out/'review.json')})
    print(json.dumps(read(out/'review.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.source,a.output)

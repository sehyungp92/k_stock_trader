"""Independent C9-C direct-target/other-account and chronological attribution.

This is descriptive accounting of complete policies, not isolated causal effects.
It does not fit, select, alter criteria, start a worker or promote a policy.
"""
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify


def routes(folder):
    result={}
    for f in read(folder/'fills.json'):
        if f['side']!='BUY':continue
        key=f['symbol']+'/'+f['metadata']['signal_bar'];route=f['metadata']['entry_route']
        if key in result and result[key]!=route:raise ValueError('Ambiguous opportunity route')
        result[key]=route
    return result


def partitions(a,b,ar,br,signals):
    groups=defaultdict(lambda:dict(net_delta_krw=0.,control_pnl_krw=0.,treatment_pnl_krw=0.,
        shared=0,added=0,removed=0,quantity_krw=0.,unit_krw=0.,added_pnl_krw=0.,removed_pnl_krw=0.))
    details=[]
    for key in sorted(set(a)|set(b)):
        symbol,clock=key.split('/',1);day=clock[:10];signal=signals.get(day+'/'+symbol)
        active=signal and signal['predictions']['baseline'] < -23 and signal['predictions']['combined'] < -33
        direct=bool(active and (ar.get(key)=='first30_open_anchor' or br.get(key)=='first30_open_anchor'))
        label='signal_opening' if direct else 'other_account';g=groups[label]
        av=a.get(key,{});bv=b.get(key,{});an=av.get('net_pnl_krw',0.);bn=bv.get('net_pnl_krw',0.)
        g['net_delta_krw']+=an-bn;g['control_pnl_krw']+=bn;g['treatment_pnl_krw']+=an
        if key in a and key in b:
            g['shared']+=1;g['quantity_krw']+=(av['qty']-bv['qty'])*bv['net_pnl_per_share']
            g['unit_krw']+=av['qty']*(av['net_pnl_per_share']-bv['net_pnl_per_share'])
        elif key in a:g['added']+=1;g['added_pnl_krw']+=an
        else:g['removed']+=1;g['removed_pnl_krw']+=bn
        details.append(dict(key=key,group=label,signal=signal if direct else None,
            control=bv,treatment=av,delta_krw=an-bn))
    for g in groups.values():
        if abs(g['quantity_krw']+g['unit_krw']+g['added_pnl_krw']-g['removed_pnl_krw']-g['net_delta_krw'])>.01:
            raise ValueError('Target partition does not reconcile')
    return dict(groups),details


def main(root):
    done=read(root/'completion.json')
    if not done['complete'] or done['accounts']!=10:raise ValueError('Incomplete registered matrix')
    verify([done['assessment'],done['registration']]);protocol=read(root/'protocol.json')
    verify(protocol['programs']+protocol['inputs']+[protocol['signal_map']])
    out=root/'target_attribution_v1.json'
    if out.exists():raise ValueError('Preserve completed attribution')
    assessment=read(root/'assessment.json');signals=read(root/'horizon_map.json');cells={}
    for name,t in assessment['treatments'].items():
        mode,period,s=name.split('/');key=period+'/'+s;parent=Path(protocol['references'][period][s])
        a=t['activity']['opportunities'];b=assessment['control_activity'][key]['opportunities']
        groups,details=partitions(a,b,routes(root/'cases'/mode/period/s),routes(parent),signals)
        delta=sum(g['net_delta_krw'] for g in groups.values())
        if abs(delta-t['activity_attribution']['net_pnl_delta_krw'])>.01:raise ValueError('Policy difference mismatch')
        month={}
        for m in sorted({r['key'][7:14] for r in details}):
            rr=[r for r in details if r['key'][7:14]==m]
            month[m]={g:sum(r['delta_krw'] for r in rr if r['group']==g) for g in ('signal_opening','other_account')}
        control=assessment['controls'][key]
        cells[name]={'return_delta_pp':t['account']['metrics']['return_pct']-control['metrics']['return_pct'],
            'net_delta_krw':delta,'activity_delta':t['activity_attribution']['summary_delta'],
            'money_dd_delta_krw':t['mtm']['max_money_dd_krw']-control['mtm']['max_money_dd_krw'],
            'percentage_dd_delta_pp':t['mtm']['dd_pct']-control['mtm']['dd_pct'],
            'control_recovery':control['mtm'],'treatment_recovery':t['mtm'],
            'groups':groups,'monthly':month,'opportunities':details,
            'activation':read(root/'cases'/mode/period/s/'activation_review.json')}
    write(out,{'at_utc':datetime.now(timezone.utc).isoformat(),'program':bound(__file__),
        'inputs':[bound(root/'completion.json'),done['assessment'],done['registration'],protocol['signal_map']],
        'cells':cells,'scope':'Exact funded-path arithmetic, not causal marginal attribution. Signal-opening versus other-account includes compounding and altered future states. Current decision signals are retained beside every favourable/adverse target outcome; no outcome-derived executable rule. Reused development; no automatic qualification or successor experiment.'})
    print(json.dumps({n:{k:v[k] for k in ('return_delta_pp','net_delta_krw','activity_delta','money_dd_delta_krw','percentage_dd_delta_pp','groups')} for n,v in cells.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)

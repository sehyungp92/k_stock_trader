"""C9-D: existing opening beliefs on original population-vetoed admissions."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,os,sys,time,traceback
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import joblib
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify
from scripts.kalcb_entry_lifecycle_panel_v2 import lines
from scripts.kalcb_funded_horizon_v1 import opening_features,PREP,INFORMATION,ALIGN,AUDIT

DESIGN=REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0033-restoration-design.md'


def summarise(rows):
    valid=[r for r in rows if r['label_valid']]
    return dict(opportunities=len(rows),issuer_days=len({r['base_id'] for r in rows}),sessions=len({r['day'] for r in rows}),
        valid=len(valid),invalid=len(rows)-len(valid),missing_prefix=sum(not r['prefix_supported'] for r in rows),
        full_lifetime_krw=sum(r['full_krw'] for r in valid),earned_to35_krw=sum(r['earned_krw'] for r in valid if r['prefix_supported']),
        remaining_after35_krw=sum(r['remaining_krw'] for r in valid if r['prefix_supported']),
        positive_wealth_krw=sum(max(0,r['full_krw']) for r in valid),negative_wealth_krw=sum(min(0,r['full_krw']) for r in valid),
        profitable=sum(r['full_krw']>0 for r in valid),profitable_material=sum(r['full_krw']>=10000 and r['filled_notional']>=1000000 for r in valid),
        donor_overlap_sessions=len({r['day'] for r in rows if r['same_day_donor']}),
        fraction_counts=dict(Counter(str(r['quantity_scale']) for r in rows)))


def run(root):
    import psutil
    root.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    source=PREP/'funded_horizon_v1';c=read(source/'completion.json');verify([c['assessment'],c['registration']])
    b=read(INFORMATION/'completion.json');verify([b['registration']]+b['outputs'])
    a=read(PREP/'entry_lifecycle_v2/completion.json');verify([a['registration'],a['panel'],a['assessment']])
    audit_binding=next(v for v in read(AUDIT/'completion.json')['artifacts'] if Path(v['path']).name=='decisions.jsonl.gz');verify([audit_binding])
    inputs=[bound(source/'completion.json'),c['assessment'],c['registration'],bound(source/'target_attribution_v1.json'),
        bound(source/'horizon_map.json'),a['panel'],a['registration'],b['registration'],*b['outputs'],audit_binding]
    prepared={}
    for cell in read(ALIGN/'protocol.json')['cells']:
        if cell['kind']!='opening':continue
        p=Path(cell['prepared']);meta=read(p/'prepared.json');binding=dict(path=str(p/'opportunities.jsonl'),sha256=meta['opportunities_sha256'])
        verify([binding]);inputs+=[binding,bound(p/'prepared.json')]
        for r in lines(p/'opportunities.jsonl'):prepared[r['opportunity_id']]=r
    registration=dict(at_utc=datetime.now(timezone.utc).isoformat(),program=bound(__file__),design=bound(DESIGN),
        inputs=inputs,feature_program=bound(REPO/'scripts/kalcb_funded_horizon_v1.py'),native_cases=0,fits=0)
    write(root/'registration.json',registration);me=psutil.Process();write(root/'process.json',dict(pid=me.pid,created_epoch=me.create_time(),command=me.cmdline()))
    proof=read(INFORMATION/'model_support.json');models={i:joblib.load(INFORMATION/f'models_{i:02}.joblib') for i in range(len(proof))}
    audit=list(lines(AUDIT/'decisions.jsonl.gz'));targets={};missing=Counter()
    all_removed=[r for r in audit if r['original_parent_selected'] and 'control' not in r['selected']]
    for r in all_removed:
        choices=[i for i,p in enumerate(proof) if p['origin']<=r['day']]
        if not choices:missing[r['period']+'_warmup']+=1;continue
        i=choices[-1]
        if proof[i]['observed_through']>=r['day']:raise ValueError('Future model')
        x=opening_features(prepared[r['base_id']],r['quantity_scale'],r)
        pred={s:float(models[i][s]['tree_B']['tree'].predict(np.array([x]))[0]) for s in ('baseline','combined')}
        targets[r['action_id']]=dict(audit=r,features=x,predictions=pred,origin=proof[i]['origin'],positive=all(v>0 for v in pred.values()))
    common=0
    for r in read(INFORMATION/'renewal_predictions.json'):
        if r['action_id'] not in targets:continue
        pred=targets[r['action_id']]['predictions']
        if any(abs(pred[s]-r['predictions']['tree_B'][s])>1e-8 for s in pred):raise ValueError('Prediction bridge mismatch')
        common+=1
    donors={r['day'] for r in read(source/'horizon_map.json').values() if r['predictions']['baseline'] < -23 and r['predictions']['combined'] < -33}
    ar=read(a['registration']['path']);days={t['audit']['day'] for t in targets.values()}
    jobs=[j for j in ar['jobs'] if j['day'] in days]
    declared={str(Path(v['path']).resolve()):v for v in ar['inputs']}
    event_bindings=[declared[str(Path(j['events']).resolve())] for j in jobs]
    verify(event_bindings);write(root/'event_input_registration.json',dict(inputs=event_bindings,registration=bound(root/'registration.json')))
    notionals={}
    for j in jobs:
        for e in lines(j['events']):
            if e['opportunity_id'] in targets:
                notionals[e['opportunity_id'],j['scenario']]=sum(f['qty']*f['price'] for f in e['fills'] if f['side']=='BUY')
    rows=[]
    for p in lines(PREP/'entry_lifecycle_v2/panel.jsonl.gz'):
        t=targets.get(p['action_id'])
        if not t:continue
        r=t['audit'];label=r['labels'][p['scenario']]
        full=label['net_value_equity_bps']*10000 if label['label_valid'] else None
        if full is not None and abs(full-p['net_reconstructed_krw'])>.02:raise ValueError('Lifetime label mismatch')
        filled=notionals[p['action_id'],p['scenario']]
        rows.append(dict(action_id=p['action_id'],base_id=p['base_id'],day=p['day'],period=p['period'],scenario=p['scenario'],
            quantity_scale=p['quantity_scale'],prefix_supported=p['prefix_supported'],label_valid=label['label_valid'],
            physical_risk_pass=r['physical_risk_pass'],capacity_pass=r['capacity_pass'],full_krw=full,
            earned_krw=p['marked_net_at35_krw'] if p['prefix_supported'] else None,
            remaining_krw=p['remaining_net_after35_krw'] if p['prefix_supported'] else None,filled_notional=filled,
            same_day_donor=p['day'] in donors,features=t['features'],predictions=t['predictions'],positive=t['positive'],
            original_rank=r['descriptor']['rank'],origin=t['origin']))
    cells={}
    for period in ('is','oos'):
        for s in ('baseline','combined'):
            rr=[r for r in rows if r['period']==period and r['scenario']==s]
            for group in ('all_removed','positive','nonpositive'):
                cc=[r for r in rr if group=='all_removed' or r['positive']==(group=='positive')]
                cells[period+'/'+s+'/'+group]=dict(total=summarise(cc),months={m:summarise([r for r in cc if r['day'].startswith(m)]) for m in sorted({r['day'][:7] for r in cc})})
    write(root/'rows.json',rows);write(root/'assessment.json',dict(cells=cells,unavailable=dict(missing),original_removed=len(all_removed),
        common_predictions_exact=common,native_cases=0,fits=0,qualification=False,
        scope='Isolated original-quantity full native lifetimes; material notional uses authenticated actual BUY fills and100m reference equity. Counts are not funded-policy activity. All reused development. Same-day donor overlap is not joint funding proof.'))
    verify(inputs+[registration['program'],registration['design'],registration['feature_program']])
    write(root/'completion.json',dict(complete=True,at_utc=datetime.now(timezone.utc).isoformat(),elapsed_s=time.monotonic()-started,
        registration=bound(root/'registration.json'),events=bound(root/'event_input_registration.json'),rows=bound(root/'rows.json'),assessment=bound(root/'assessment.json'),economic_objective_achieved=False))
    print(json.dumps(dict(cells={k:v['total'] for k,v in cells.items()},unavailable=dict(missing),common=common),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a.output)
    except Exception as e:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise

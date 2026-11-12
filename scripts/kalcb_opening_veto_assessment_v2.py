"""C10 original assessment with explicit Path normalization at helper boundaries."""
from pathlib import Path
import numpy as np
from scripts import kalcb_opening_veto_qualification_v1 as original
read,write,diag,useful_days=original.read,original.write,original.diag,original.useful_days


def assess(root):
    root=Path(root);p=read(root/'protocol.json');ledger=read(root/'completed_cases.json');candidate=dict(p['reused']);cells={};capital={};rejected={}
    for row in ledger:
        c=row['case']
        if c['kind']=='standard':
            key=c['period']+'/'+c['scenario']
            if row['verdict']['qualified']:candidate[key]=str(root/'cases'/c['id'])
            else:rejected[key]=row['verdict']
    for key,path in candidate.items():
        folder=Path(path);control=Path(p['controls'][key])
        a=diag.account(folder);b=diag.account(control);aa=diag.opportunities(folder);bb=diag.opportunities(control)
        cells[key]=dict(account=a,control=b,metrics=diag.metrics(folder),control_metrics=diag.metrics(control),
            activity=aa,control_activity=bb,path=diag.paired_attribution(a,b),activity_delta=diag.compare(aa,bb),
            profitable_material_issuer_days=useful_days(aa),control_profitable_material_issuer_days=useful_days(bb))
    for row in ledger:
        c=row['case']
        if not c['id'].startswith('capital_'):continue
        s=c['scenario'];initial=c['initial_equity'];old=p['capital_controls'][s];item=dict(verdict=row['verdict'],control_verdict=old['verdict'],initial_equity=initial,matched_qualified=row['verdict']['qualified'] and old['qualified'])
        if row['verdict']['qualified']:
            folder=root/'cases'/c['id'];item.update(metrics=diag.metrics(folder,initial),account=diag.account(folder,initial),activity=diag.opportunities(folder,initial))
        if item['matched_qualified']:
            control=Path(old['folder']);b=diag.account(control,initial);bb=diag.opportunities(control,initial)
            item.update(control_metrics=diag.metrics(control,initial),net_delta_krw=item['account']['net_pnl_krw']-b['net_pnl_krw'],
                activity_delta=diag.compare(item['activity'],bb),path=diag.paired_attribution(item['account'],b))
        capital[s]=item
    od=[v['account']['net_pnl_krw']-v['control']['net_pnl_krw'] for k,v in cells.items() if k.startswith('oos/')]
    iid=[v['account']['net_pnl_krw']-v['control']['net_pnl_krw'] for k,v in cells.items() if k.startswith('is/')]
    fd={pfx:float(np.median([v['profitable_material_issuer_days']-v['control_profitable_material_issuer_days'] for k,v in cells.items() if k.startswith(pfx+'/')])) for pfx in ('is','oos')}
    criteria=dict(all_standard_valid=len(cells)==8 and not rejected,
        component_return_reference=len(cells)==8 and sum(x>=1000000 for x in od)>=3 and float(np.median(iid))>=0,
        capital_portability=len(capital)==4 and all(v['verdict']['qualified'] for v in capital.values()) and sum(v.get('net_delta_krw',0)>0 for v in capital.values() if v['matched_qualified'])>=2,
        useful_frequency_reference=len(cells)==8 and fd['oos']>=3 and fd['is']>=0,median_frequency_change=fd,
        human_economic_risk_chronology_assessment_required=True,economic_objective_achieved=False)
    write(root/'assessment.json',dict(cells=cells,rejected_standard=rejected,capital=capital,criteria=criteria,scope='C10 references; no automatic economic success or promotion. Reused development.'))

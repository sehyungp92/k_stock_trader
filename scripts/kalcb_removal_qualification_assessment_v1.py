"""Read-only assessment of every registered removal prior and stress scenario."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.kalcb_population_assessment_v1 import account,paired_attribution
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify,validate_native


def drawdown_episodes(equity,initial=100_000_000.):
    peak=float(initial);peak_at=None;episode=None;episodes=[]
    for stamp,value in equity:
        value=float(value)
        if value>=peak:
            if episode is not None:
                episode['recovered_at']=stamp;episodes.append(episode);episode=None
            peak=value;peak_at=stamp
        else:
            depth=100*(1-value/peak)
            if episode is None:episode={'peak_at':peak_at,'peak_equity':peak,'trough_at':stamp,
                'trough_equity':value,'drawdown_pct':depth,'recovered_at':None}
            elif depth>episode['drawdown_pct']:
                episode.update(trough_at=stamp,trough_equity=value,drawdown_pct=depth)
    if episode is not None:episodes.append(episode)
    return sorted(episodes,key=lambda e:e['drawdown_pct'],reverse=True)


def assess(root,out):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Preserve prior assessment')
    complete=read(root/'stage_completion.json')
    if not complete['complete'] or complete['fresh_native_cases']!=32:raise ValueError('Qualification batch incomplete')
    protocol=read(root/'protocol.json');study=read(root/'study.json');verify(protocol['programs']+protocol['bindings'])
    records={p.parent.name:read(p) for p in (root/'cases').glob('*/record.json')}
    if set(records)!=set(study['recipes']):raise ValueError('Missing qualification recipes')
    accounts={};trials=[];support=[]
    parents={(p['name'],p['period']):p for p in protocol['parent_profiles']}
    for name,spec in study['recipes'].items():
        verify(list(records[name]['cases'].values()))
        for period in ('is','oos'):
            if spec['prior'] is None:item=parents[spec['parent'],period]
            elif spec['prior']==100:
                item=read(Path(protocol['screening_root'])/'cases'/(spec['parent']+'_population_removals_only')/period/'input_profile.json')
            else:item=read(root/'cases'/name/period/'input_profile.json')
            for scenario in spec['scenarios']:
                folder=root/'cases'/name/period/scenario
                validate_native(folder,item,scenario,save=False)
                a=account(folder);episodes=drawdown_episodes(read(folder/'equity.json'))
                if abs((episodes[0]['drawdown_pct'] if episodes else 0)-a['metrics']['drawdown_pct'])>1e-8:
                    raise ValueError('Drawdown episode mismatch')
                a['largest_MTM_drawdown_episodes']=episodes[:5]
                accounts[name,period,scenario]=a
                if name.startswith('control') and scenario in ('baseline','combined'):
                    bridge=read(folder/'observer_bridge.json')
                    if not bridge['passed']:raise ValueError('Failed observer bridge')
                    summary=read(folder/'intervention_support.json')['summary']
                    support.append({'name':name,'period':period,'scenario':scenario,
                        'summary':{k:v for k,v in summary.items() if k!='execution_checks'},
                        'bindings':[bound(folder/f) for f in ('observer_bridge.json','intervention_support.json')]})
    for name,spec in study['recipes'].items():
        parent='control' if spec['parent']=='expanding' else 'control_expanding_atr5'
        row={'recipe':name,'specification':spec,'cases':{}}
        for period in ('is','oos'):
            for scenario in spec['scenarios']:
                a=accounts[name,period,scenario];b=accounts[parent,period,scenario]
                row['cases'][period+'/'+scenario]={'account':a,'vs_local_parent':paired_attribution(a,b)}
        row['global_IS_sacrifice_pp']={s:accounts['control','is',s]['metrics']['return_pct']-accounts[name,'is',s]['metrics']['return_pct'] for s in ('baseline','combined')}
        row['IS_reporting_bands']={str(b):all(v<=b for v in row['global_IS_sacrifice_pp'].values()) for b in (2,5,10)}
        trials.append(row)
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'program':bound(__file__),
        'inputs':[bound(root/f) for f in ('protocol.json','study.json','stage_completion.json','qualification_ledger.json')],
        'trials':trials,'actual_parent_gate_support':support,
        'scope':'All finite priors and four center stresses; April-July reused development. Conditional paired block uncertainty is not independent or multiplicity-adjusted. Order suppression support is not isolated PnL. Funded fixed-allocation comparison and final decision pending.'}
    write(out/'assessment.json',result)
    print(str(out/'assessment.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assess(a.root,a.output)

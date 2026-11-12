"""Summarize completed native recovery cases without hiding rejected trials."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_full_oos_recovery import ROOT,read,write,bound

def segments(directory):
    daily=read(directory/'daily.json');trades=read(directory/'trades.json');equity=100_000_000.;by_month={};parts={}
    for label,start,end in [('Apr-May','2026-04-01','2026-05-31'),('Jun-Jul','2026-06-01','2026-07-10')]:
        selected=[r for r in daily if start<=r['day']<=end]
        if not selected:continue
        prior=[r['equity'] for r in daily if r['day']<start];initial=prior[-1] if prior else equity
        parts[label]={'continuous_account_return_pct':(selected[-1]['equity']/initial-1)*100,
            'net_pnl_krw':selected[-1]['equity']-initial,'sessions':len(selected),
            'trades':sum(start<=t['entry_fill_time'][:10]<=end for t in trades)}
    for r in daily:
        month=r['day'][:7]
        if month not in by_month:by_month[month]={'initial':equity,'sessions':0}
        by_month[month]['final']=r['equity'];by_month[month]['sessions']+=1;equity=r['equity']
    for v in by_month.values():v['return_pct']=(v['final']/v['initial']-1)*100
    exits=defaultdict(lambda:{'trades':0,'net_pnl_krw':0.})
    for t in trades:
        v=exits[t['exit_reason']];v['trades']+=1;v['net_pnl_krw']+=t['net_pnl']
    return {'parts':parts,'months':by_month,'exit_attribution':dict(exits)}

def summarize(root):
    controls={}
    for s in ('baseline','participation','cost','combined'):
        p=root/'control'/s
        if (p/'completion.json').exists():controls[s]={'result':bound(p/'result.json'),'metrics':read(p/'result.json'),'attribution':segments(p)}
    groups={}
    for group in ('ablation_v3','extension_ablation','admission_round','terminal_round','continuous_round','refit_round','targeted_round'):
        folder=root/group
        if not folder.exists():continue
        records=[]
        for p in sorted((folder/'cases').glob('*/record.json')):
            r=read(p);name=r['recipe'];case=r['cases'];extra={}
            for scenario in ('baseline','combined'):
                path=Path(case['oos_'+scenario]['path']).parent
                extra[scenario]=segments(path)
            records.append({**r,'oos_attribution':extra})
        groups[group]={'study':bound(folder/'study.json'),'trials':records,
            'rejected':[read(p) for p in sorted((folder/'cases').glob('*/rejected_record.json'))]}
    selected=Path('config/research/kalcb_intraday_advantage_20261007/selected_research.json').resolve()
    payload={'scope':'Corrected OOS 2026-04-01 through 2026-07-10; 69 continuous-account sessions. Reused development, not untouched validation.',
        'canonical_unchanged':bound(selected),'controls':controls,'groups':groups,'program':bound(__file__)}
    write(root/'recovery_results.json',payload)
    lines=['KALCB COMPLETE-WINDOW RECOVERY EXPERIMENTS',payload['scope'],
        'Returns and drawdowns are percentages; all accounts start at KRW100m.',
        'DD targets8% baseline/10% combined are preferences, not rejection thresholds.',
        'Refit cases retain selected IS as a historical reference; they are not new IS tests of alternative schedules.','',
        'Case | IS baseline return/DD/trades | IS combined return/DD/trades | OOS baseline return/DD/trades | OOS combined return/DD/trades']
    for group,v in groups.items():
        lines.append('\n'+group)
        for r in v['trials']:
            m=r['metrics'];pieces=[r['recipe']]
            for period in ('is','oos'):
                for scenario in ('baseline','combined'):
                    k=period+'_'+scenario;pieces.append(f"{m[k+'_return']:.4f}/{m[k+'_dd']:.4f}/{m[k+'_trades']}")
            lines.append(' | '.join(pieces))
            for s,a in r['oos_attribution'].items():lines.append('  '+s+' subperiods: '+json.dumps(a['parts']))
        for r in v['rejected']:lines.append('REJECTED '+r['recipe']+': '+json.dumps(r['failures']))
    (root/'recovery_results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'groups':{k:len(v['trials']) for k,v in groups.items()},'control_scenarios':list(controls)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);summarize(p.parse_args().root)

"""Read-only forensic statistics over authenticated OLR native replay artifacts.

Outputs immutable-at-run evidence copies, descriptive diagnostics and charts.
Arithmetic attribution and bootstrap sensitivity are not executable strategies.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PARENT = Path('E:/CodexResearch/k_stock_trader/olr_round1_comprehensive_20261009/full_oos_20260401_20260710/continuous_history_v10')
OUT = REPO/'data/backtests/analysis/olr_round1_oos_review_20261010'
BASE = 'ce14d8377b9e67ac6'
CAPITAL = 100_000_000


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def authenticate(folder):
    complete, result, audit = (read(folder/f) for f in ('completion.json','result.json','execution_audit.json'))
    assert complete['complete'] and audit['audit_pass'] and result['execution_validity_pass']
    assert result['audit_status'] == 'passed' and not result['terminal_inventory']
    assert not result['terminal_pending']
    for name, expected in complete['artifacts'].items():
        assert sha(folder/name) == expected, str(folder/name)
    daily, trades, equity = (read(folder/f) for f in ('daily.json','trades.json','equity.json'))
    nav = np.array([CAPITAL, *[e[1] for e in equity]])
    dd = float(100*np.max(1-nav/np.maximum.accumulate(nav)))
    assert abs(dd-result['drawdown_pct']) < 1e-7
    assert abs(100*(daily[-1]['equity']/CAPITAL-1)-result['return_pct']) < 1e-7
    assert abs(sum(t['net_pnl'] for t in trades) - (daily[-1]['equity']-CAPITAL)) < 0.01
    assert len(trades) == result['trades'] and len(daily) == result['sessions']
    return {'artifact_hashes_pass': True, 'nav_pnl_dd_reconciliation_pass': True,
            'source_generation_hash': complete['source_generation_hash'],
            'completion_sha256': sha(folder/'completion.json')}


def bootstrap(left, right=None, draws=10000):
    def returns(daily):
        nav = np.array([CAPITAL, *[r['equity'] for r in daily]])
        return nav[1:]/nav[:-1]-1
    a = returns(left)
    b = returns(right) if right is not None else None
    if right is not None:
        assert [r['day'] for r in left] == [r['day'] for r in right]
    out = {}
    for block in (1,3,5,10):
        rng = np.random.default_rng(20261010)
        idx = (rng.integers(0,len(a),size=(draws,math.ceil(len(a)/block),1))+np.arange(block)).reshape(draws,-1)[:,:len(a)]%len(a)
        gains = 100*(np.prod(1+a[idx],axis=1)-1)
        if b is not None:
            gains = 100*(np.prod(1+b[idx],axis=1)-1)-gains
        out[str(block)] = {'ci95_pct_or_gain_pp': np.quantile(gains,[.025,.975]).tolist(),
                           'resampled_fraction_positive': float(np.mean(gains>0))}
    return out


def summary(window, folder, snapshots):
    trades, daily, equity, result = (read(folder/f) for f in ('trades.json','daily.json','equity.json','result.json'))
    notional = np.array([t['qty']*t['entry_price'] for t in trades])
    pnl = np.array([t['net_pnl'] for t in trades])
    ret = pnl/notional
    wins, losses = ret[ret>0], ret[ret<0]
    nav = np.array([CAPITAL, *[x[1] for x in equity]])
    drawdowns = 1-nav/np.maximum.accumulate(nav)
    trough = int(np.argmax(drawdowns)); peak = int(np.argmax(nav[:trough+1]))
    stamp = lambda i: equity[i-1][0] if i else 'initial cash'
    months = {}
    for month in sorted({s['trade_date'][:7] for s in snapshots}):
        rows = [s for s in snapshots if s['trade_date'].startswith(month)]
        rejects, sole = defaultdict(Counter), defaultdict(Counter)
        for s in rows:
            for stage in ('rejected_symbols','afternoon_rejected_symbols'):
                for reasons in s['metadata'].get(stage,{}).values():
                    if not isinstance(reasons,list): reasons=[reasons]
                    rejects[stage].update(reasons)
                    if len(reasons)==1:sole[stage].update(reasons)
        months[month] = {
            'entry_sessions':len(rows),
            'daily_candidates':sum(s['metadata'].get('candidate_pool_count',0) for s in rows),
            'final_candidates':sum(len(s['candidates']) for s in rows),
            'candidate_days':sum(bool(s['candidates']) for s in rows),
            'entries':sum(t['entry_fill_time'].startswith(month) for t in trades),
            'regimes':dict(Counter(s['metadata']['market']['regime'] for s in rows)),
            'mean_breadth_pct':float(np.mean([s['metadata']['market']['breadth_pct_above_20dma'] for s in rows])),
            'mean_universe_prior_20d_return_pct':float(np.mean([s['metadata']['market']['avg_20d_return_pct'] for s in rows])),
            'overlapping_rejections':{k:dict(v) for k,v in rejects.items()},
            'sole_recorded_rejections':{k:dict(v) for k,v in sole.items()},
        }
    candidates = [c for s in snapshots for c in s['candidates']]
    features = {}
    for name in ('afternoon_ret','range_atr','sector_intraday_rel_volume','sector_intraday_effective_count','sector_rotation_score'):
        v = [c.get('metadata',{}).get('afternoon_features',{}).get(name) for c in candidates]
        v = [float(x) for x in v if x is not None]
        features[name] = {'count':len(v),'min':min(v),'median':float(np.median(v)),
                          'max':max(v),'unique_values':len(set(v))} if v else {}
    weights = notional/notional.sum()
    # Exact identity: total PnL = N*(mean(size)*mean(return)+population covariance).
    unweighted_component = len(trades)*notional.mean()*ret.mean()
    covariance_component = float(pnl.sum()-unweighted_component)
    assert abs(unweighted_component+covariance_component-pnl.sum()) < 0.01
    day_pnl = defaultdict(float)
    for t in trades:day_pnl[t['entry_fill_time'][:10]]+=t['net_pnl']
    return {
        'window':window,'result':result,
        'entry_turnover_multiple':float(notional.sum()/CAPITAL),
        'entry_notional_mean':float(notional.mean()),'entry_notional_median':float(np.median(notional)),
        'equal_weight_trade_net_return_pct':float(ret.mean()*100),
        'notional_weighted_trade_net_return_pct':float(pnl.sum()/notional.sum()*100),
        'median_trade_net_return_pct':float(np.median(ret)*100),
        'mean_winning_trade_return_pct':float(wins.mean()*100),
        'mean_losing_trade_return_pct':float(losses.mean()*100),
        'notional_return_correlation':float(np.corrcoef(notional,ret)[0,1]),
        'pnl_decomposition_mean_size_times_mean_return':float(unweighted_component),
        'pnl_decomposition_size_return_covariance':covariance_component,
        'pnl_decomposition_caveat':'Exact descriptive identity over executed positions; not equal-weight counterfactual performance.',
        'turnover_weight_effective_count':float(1/(weights@weights)),
        'loss_return_p05_pct':float(np.quantile(ret,.05)*100),
        'top_two_loss_pnl':float(np.sort(pnl)[:2].sum()),
        'top_two_profit_pnl':float(np.sort(pnl)[-2:].sum()),
        'entry_days':len(day_pnl),'best_entry_days':sorted(day_pnl.items(),key=lambda x:-x[1])[:5],
        'worst_entry_days':sorted(day_pnl.items(),key=lambda x:x[1])[:5],
        'bar_mtm_drawdown':{'pct':float(drawdowns[trough]*100),'peak':stamp(peak),'peak_nav':float(nav[peak]),
                            'trough':stamp(trough),'trough_nav':float(nav[trough])},
        'baseline_return_block_bootstrap':bootstrap(daily),'monthly_funnel':months,'selected_features':features,
    }


def contribution(left, right):
    def aggregate(rows):
        grouped = defaultdict(lambda: {'pnl':0., 'notional':0., 'qty':0})
        for t in rows:
            key=t['entry_fill_time'][:10]+'/'+t['symbol']
            v=grouped[key];v['pnl']+=t['net_pnl'];v['notional']+=t['qty']*t['entry_price'];v['qty']+=t['qty']
        return dict(grouped)
    a,b=aggregate(left),aggregate(right)
    rows=[]
    for key in sorted(a.keys()|b.keys()):
        if a.get(key)==b.get(key):continue
        rows.append({'issuer_day':key,'delta_net_pnl':b.get(key,{}).get('pnl',0)-a.get(key,{}).get('pnl',0),
                     'baseline':a.get(key),'treatment':b.get(key)})
    positive=sorted([x['delta_net_pnl'] for x in rows if x['delta_net_pnl']>0],reverse=True)
    return {'changed_issuer_days':len(rows),'delta_net_pnl':sum(x['delta_net_pnl'] for x in rows),
            'top2_positive_contribution_share':sum(positive[:2])/sum(positive) if positive else None,
            'delta_excluding_top2_positive_arithmetic_only':sum(x['delta_net_pnl'] for x in rows)-sum(positive[:2]),
            'largest_positive':sorted(rows,key=lambda x:-x['delta_net_pnl'])[:5],
            'largest_negative':sorted(rows,key=lambda x:x['delta_net_pnl'])[:5]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ('mechanism_diagnostics.json','calendar_subperiods.json','matched_length_is_windows.json',
                 'history_performance_attribution.json','history_control_comparison.json','protocol.json'):
        shutil.copyfile(PARENT/name, OUT/name)
    stats, baselines, provenance = {}, {}, []
    for window in ('train','oos'):
        folder=PARENT/'cases'/BASE/window/'100000000_baseline'
        check=authenticate(folder)
        destination=OUT/'evidence'/window
        destination.mkdir(parents=True,exist_ok=True)
        for name in ('assessment.json','result.json','completion.json','daily.json','trades.json','equity.json','execution_audit.json'):
            shutil.copyfile(folder/name,destination/name)
            provenance.append({'source':str(folder/name),'copy':str(destination/name),'sha256':sha(destination/name)})
        st=read(PARENT/'registrations'/window/(BASE+'_100000000')/'study.json')
        path=Path(st['code'])/st['olr_snapshots']
        days={x['day'] for x in read(folder/'daily.json')}
        snapshots=[s for s in read(path) if s['trade_date'] in days]
        write(destination/'scored_snapshots.json',snapshots)
        provenance.append({'source':str(path),'sha256':sha(path),'scored_snapshot_count':len(snapshots)})
        stats[window]=summary(window,folder,snapshots)
        stats[window]['authentication']=check
        baselines[window]=read(folder/'assessment.json')
    write(OUT/'diagnostics.json',stats)
    comparisons=[]; pairs=defaultdict(dict); excluded=[]
    for root in (PARENT,OUT/'replays'):
        for p in sorted((root/'cases').glob('*/*/100000000_baseline/assessment.json')):
            a=read(p)
            if not a.get('qualified'):
                audit_path=p.parent/'execution_audit.json'
                audit=read(audit_path) if audit_path.exists() else {}
                excluded.append({'path':str(p),'candidate_id':a['candidate_id'],'window':a['window'],
                    'status':a['status'],'label':a['label'],'error':a.get('error'),
                    'audit_status':a.get('audit_status'),
                    'native_audit_errors':audit.get('audit_receipt',{}).get('errors'),
                    'disposition':a.get('disposition','Not ranked: native audit / qualification did not pass.')})
                continue
            a['authentication']=authenticate(p.parent)
            pairs[a['candidate_id']][a['window']]=a
    support={}
    priority={'c1e5c4b5e26e960b8','cff804c36e3d4b7d0','ca18d9a35ba34c1cc','ce0d97e2f019d24c1',
              'c609a66b6ce80f748','cbd14aff8c9557723','c272fe0390df855bd'}
    if (OUT/'replays/interaction_completion.json').exists():
        priority.add(read(OUT/'replays/interaction_completion.json')['candidate_id'])
    for cid,pair in pairs.items():
        if set(pair)!={'train','oos'}:continue
        row={'candidate_id':cid,'label':pair['train']['label']}
        for w,a in pair.items():
            for key in ('return_pct','drawdown_pct','trades'):
                row[w+'_'+key]=a[key]
            row[w+'_gain_pp']=a['return_pct']-baselines[w]['return_pct']
        row['is_preserved_3pp']=row['train_gain_pp']>=-3
        row['oos_uplift_2pp']=row['oos_gain_pp']>=2
        row['activity_preserved']=all(pair[w]['trades']>=baselines[w]['trades'] for w in pair)
        row['drawdown_within_2pp']=all(pair[w]['drawdown_pct']<=baselines[w]['drawdown_pct']+2 for w in pair)
        row['passes_registered_gates']=all(row[k] for k in ('is_preserved_3pp','oos_uplift_2pp','activity_preserved','drawdown_within_2pp'))
        comparisons.append(row)
        if cid in priority:
            support[cid]={}
            for w,a in pair.items():
                baseline_folder=Path(baselines[w]['output']); treatment_folder=Path(a['output'])
                support[cid][w]={
                    'contribution':contribution(read(baseline_folder/'trades.json'),read(treatment_folder/'trades.json')),
                    'paired_block_bootstrap':bootstrap(read(baseline_folder/'daily.json'),read(treatment_folder/'daily.json')),
                }
    comparisons.sort(key=lambda a:-a['oos_return_pct'])
    write(OUT/'diagnostics.json',stats);write(OUT/'comparisons.json',comparisons)
    write(OUT/'candidate_support.json',support);write(OUT/'excluded_trials.json',excluded)
    if (OUT/'replays/stress_completion.json').exists():
        write(OUT/'stress_authentication.json',[
            {'candidate_id':r['candidate_id'],'window':r['window'],**authenticate(Path(r['output']))}
            for r in read(OUT/'replays/stress_results.json')])
    with (OUT/'comparisons.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    write(OUT/'provenance.json',{
        'asof_utc':datetime.now(timezone.utc).isoformat(),'parent':str(PARENT),
        'program_sha256':sha(__file__),'sources':provenance,
        'audited_paired_configurations':len(comparisons),'excluded_assessments':len(excluded),
        'all_comparison_artifacts_authenticated':True,
        'statistical_scope':'Post-selection descriptive statistics. Bootstrap is not selection-adjusted or independent validation.',
        'promotion':False})
    charts(stats,comparisons)
    print(json.dumps({'output':str(OUT),'pairs':len(comparisons),'gate_passers':[r['candidate_id'] for r in comparisons if r['passes_registered_gates']],
        'stats':{w:{k:v for k,v in s.items() if k not in ('result','monthly_funnel','selected_features','authentication')} for w,s in stats.items()}},indent=2))


def charts(stats, comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    colors={'train':'#2563eb','oos':'#dc2626'}
    for w in ('train','oos'):
        daily=read(OUT/'evidence'/w/'daily.json')
        axes[0,0].plot(range(len(daily)+1),[0,*[100*(d['equity']/CAPITAL-1) for d in daily]],label='IS' if w=='train' else 'OOS',color=colors[w],lw=2)
    axes[0,0].set(title='Same initial cash; complete session-close NAV',xlabel='Exchange sessions since window start',ylabel='Return (%)');axes[0,0].legend()
    for w,offset in [('train',-.18),('oos',.18)]:
        d=read(PARENT/'mechanism_diagnostics.json')[w]['cohorts']['rank']
        axes[0,1].bar(np.arange(1,6)+offset,[d.get(str(i),{}).get('net_pnl',0)/1e6 for i in range(1,6)],width=.35,label='IS' if w=='train' else 'OOS',color=colors[w])
    axes[0,1].set(title='Capital concentrated in rank 1; its payoff reverses',xlabel='Entry candidate rank',ylabel='Realized net P&L (KRW million)',xticks=range(1,6));axes[0,1].axhline(0,color='#64748b',lw=.7)
    calendar=read(PARENT/'calendar_subperiods.json')
    months=list(calendar['oos']['months']);vals=[calendar['oos']['months'][m]['return_pct'] for m in months]
    axes[1,0].bar(months,vals,color=['#0d9488' if v>=0 else '#dc2626' for v in vals]);axes[1,0].set(title='OOS deterioration is concentrated in June',ylabel='Calendar-month NAV return (%)');axes[1,0].axhline(0,color='#64748b',lw=.7)
    unique={ (round(x['train_return_pct'],6),round(x['oos_return_pct'],6)):x for x in comparisons}
    for x in unique.values():
        isnew=x['label'].startswith('completed_bar_loss_guard')
        interaction=x['label'].startswith('wider_entry_extension_plus_')
        axes[1,1].scatter(x['train_return_pct'],x['oos_return_pct'],s=65 if isnew or interaction else 25,
                         c='#7c3aed' if isnew else '#0d9488' if interaction else '#64748b',alpha=.8)
        if isnew or interaction:
            label=('wide entry + ' if interaction else '')+x['label'].split('loss')[-1].replace('atr','R')
            axes[1,1].annotate(label,(x['train_return_pct'],x['oos_return_pct']),xytext=(4,4),textcoords='offset points',fontsize=8)
    axes[1,1].axvline(52.0465907665,color='#0d9488',ls='--',lw=1,label='IS return floor 52.05%')
    axes[1,1].axhline(-1.3276524383,color='#0d9488',ls=':',lw=1,label='OOS return floor -1.33%')
    axes[1,1].scatter([55.0465907665],[-3.3276524383],marker='*',s=160,c='#f59e0b',zorder=5,label='Saved round 1')
    axes[1,1].set(title='Audited paired trials (duplicates collapsed)',xlabel='IS return (%)',ylabel='OOS return (%)');axes[1,1].legend(loc='lower right',fontsize=8)
    fig.suptitle('OLR round 1: performance gap and tested mitigations',fontsize=17,fontweight='bold')
    fig.savefig(OUT/'diagnostic_overview.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    main()

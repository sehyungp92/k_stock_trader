"""Audit real feature observations at the scoring boundary without fitting returns."""
from collections import Counter
from datetime import date
import json
from pathlib import Path
from statistics import fmean
import sys
from types import SimpleNamespace


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(root):
    info=read(root/'generations_v10/oos/ready.json')
    files=Path(info['files']);study=info['study']
    sys.path.insert(0,str(files))
    import os
    os.chdir(files)
    import pandas as pd
    from backtests.core.input_manifest import InputManifest
    from strategy_common.source_quality import complete_feature_bars
    from strategy_common.sector_intraday import (AFTERNOON_CUTOFF,completed_session_bars,
        _relative_cutoff_volume_map,load_canonical_sector_map,member_from_bars,
        score_sector_members,cutoff_label_for)

    boundary=date(2026,4,1);is_start=date(2025,8,7);previous=date(2026,3,31)
    contract=info['manifest']['evidence']['shared_market_inputs']
    volumes={};target_bars={};excluded=[]
    with InputManifest(study['manifest']).activate():
        sectors=load_canonical_sector_map({'sector_map_path':str(files/contract['sector_map_path'])})
        for symbol,names in sorted(contract['intraday_files'].items()):
            for name in names:
                frame=pd.read_parquet(files/name)
                frame=frame[frame.timestamp.dt.date<=boundary]
                for day,group in frame.groupby(frame.timestamp.dt.date):
                    bars=tuple(SimpleNamespace(timestamp=r.timestamp.to_pydatetime(),open=r.open,
                        high=r.high,low=r.low,close=r.close,volume=r.volume,is_completed=True)
                        for r in group.itertuples(index=False))
                    accepted,rejected=complete_feature_bars({(day,symbol):bars},AFTERNOON_CUTOFF)
                    excluded.extend(rejected)
                    if not accepted:continue
                    prefix=completed_session_bars(bars,day,cutoff=AFTERNOON_CUTOFF)
                    if prefix:
                        volumes[day,symbol]=sum(max(float(b.volume),0.) for b in prefix)
                        if day in (previous,boundary):target_bars[day,symbol]=prefix
        continuous=_relative_cutoff_volume_map(volumes)
        reset_volumes={k:v for k,v in volumes.items() if k[0]>=previous}
        reset=_relative_cutoff_volume_map(reset_volumes)
        prefix_only=_relative_cutoff_volume_map({k:v for k,v in volumes.items() if k[0]<boundary})
        assert all(continuous[k]==v for k,v in prefix_only.items()), 'Later observations affected prior features'
        coverage=[]
        for symbol in sorted(contract['intraday_files']):
            for label,start in (('is',is_start),('oos',boundary)):
                prior=sorted((d,v) for (d,s),v in volumes.items() if s==symbol and d<start and v>0)
                key=(start,symbol);expected=None
                if key in volumes:
                    expected=volumes[key]/max(fmean(v for _,v in prior[-20:]),1.) if prior else 1.
                    assert continuous[key]==expected, 'Boundary history does not equal uninterrupted rolling history'
                coverage.append({'window':label,'symbol':symbol,'positive_complete_prior_count':len(prior),
                    'used_prior_dates':[str(d) for d,_ in prior[-20:]],'feature_available':key in volumes,
                    'continuous_relative_volume':expected,'old_reset_relative_volume':reset.get(key)})
        # Build an independent sector reference from the continuous per-symbol
        # series, and compare actual reconstructed previous-session metadata.
        panels={}
        for target in (previous,boundary):
            members=[member_from_bars(s,sectors.get(s,'UNKNOWN'),d,bars,rel_volume=continuous[d,s])
                for (d,s),bars in target_bars.items() if d==target]
            panels[target]=score_sector_members([m for m in members if m],trade_date=target,
                cutoff_label=cutoff_label_for(AFTERNOON_CUTOFF))
    snapshots=read(files/study['olr_snapshots']);comparisons=[]
    for snapshot in snapshots:
        for candidate in snapshot['candidates']:
            metadata=candidate.get('metadata',{})
            prior_day=metadata.get('prev_session_sector_intraday_date')
            if not prior_day or date.fromisoformat(prior_day) not in panels:continue
            panel=panels[date.fromisoformat(prior_day)]
            feature=panel.feature_for(date.fromisoformat(prior_day),candidate['symbol'],sector=candidate['sector'])
            reference=feature.metadata(prefix='prev_session')
            bad=[k for k,v in reference.items() if metadata.get(k)!=v]
            comparisons.append({'day':snapshot['trade_date'],'symbol':candidate['symbol'],'fields':len(reference),'mismatches':bad})
            assert not bad, 'Actual candidate metadata differs from continuous reference'
    if not comparisons:raise ValueError('No actual baseline metadata overlap for independent history comparison')
    insufficient=[r for r in coverage if r['positive_complete_prior_count']<20]
    if insufficient:raise ValueError('Unresolved rolling-history readiness: '+str(insufficient))
    report={'status':'passed','scoring_windows':read(root/'protocol.json')['windows'],
        'history_policy':'All authenticated IS observations; no scoring-boundary feature reset.',
        'independent_reference':'Actual retained bars, complete-prefix policy, prior 20 positive observations; no return data used.',
        'positive_complete_prior_counts':{w:{'min':min(r['positive_complete_prior_count'] for r in coverage if r['window']==w),
            'max':max(r['positive_complete_prior_count'] for r in coverage if r['window']==w)} for w in ('is','oos')},
        'all_symbols_and_boundary_features':coverage,'actual_candidate_feature_comparisons':comparisons,
        'future_observations_changed_previous_features':False,
        'old_reset_first_oos_symbol_features_changed':sum(r['continuous_relative_volume']!=r['old_reset_relative_volume']
            for r in coverage if r['window']=='oos' and r['feature_available']),
        'complete_prefix_exclusions_by_month':dict(Counter(x[-10:-3] for x in excluded)),
        'performance_established':False,
        'afternoon_path':'Existing frozen build_afternoon_contexts rebuilds a same-day panel in both periods. Full-history rolling panel supplies previous-session metadata. This source repair does not redefine that shared strategy behavior; assess a causal rolling current-panel experiment separately under the targeted review.'}
    (root/'continuous_history_verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('all_symbols_and_boundary_features','actual_candidate_feature_comparisons')}),flush=True)


if __name__=='__main__':
    verify(Path(sys.argv[1]).resolve())

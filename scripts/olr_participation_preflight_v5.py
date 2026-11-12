"""Before any native outcomes: exact incumbent parity and source-bound signal counts."""
from collections import Counter
import copy
from pathlib import Path
from scripts import olr_participation_study_v5 as t
n,s,b=t.n,t.s,t.b


def main():
    assert not (t.ROOT/'cases').exists(),'Preflight must precede outcomes'
    plan=n.read(t.ROOT/'candidate_plan.json'); evidence=[]
    for w in ('train','oos'):
        runtimes={c['label']:n.read(n.prepare_case(t.ROOT,w,c,100_000_000)) for c in plan['candidates']}
        ready=n.read(t.ENGINES/w/'ready.json'); files=Path(ready['files'])
        old={r['trade_date']:r for r in n.read(files/ready['study']['olr_snapshots'])}
        snapshot_path=files/runtimes['bounded_participation_rescue']['olr_snapshots']
        new={r['trade_date']:r for r in n.read(snapshot_path)}
        assert set(old)==set(new)
        additions=[]; incumbents=0
        screen=n.read(t.REPO/f'data/backtests/analysis/olr_repair_failure_20261011/frequency_feasibility_{w}.json')
        screened={(r['date'],r['symbol']):r for r in screen['rows'] if r['current_sector_confirmed']}
        period=ready['study']['configuration']['date_range']
        for day,before in old.items():
            after=new[day];proof=after['metadata']['empty_participation_rescue_v5']
            assert proof['baseline_empty']==(not bool(before['candidates']))
            if before['candidates']:
                # The added default-off config field changes the full research
                # provenance hash. Strip only that identity, never a feature,
                # rank, sizing input or score; final executable config remains
                # separately bound by the snapshot/request native audit.
                def economic_candidates(rows):
                    rows=copy.deepcopy(rows)
                    for c in rows:
                        assert c['metadata'].pop('research_config_hash',None)
                    return rows
                assert economic_candidates(before['candidates'])==economic_candidates(after['candidates']),(w,day,'Incumbent candidate/features/rank changed')
                incumbents+=len(before['candidates'])
            else:
                for c in after['candidates']:
                    m=c['metadata'];f=m['afternoon_features'];sym=c['symbol']
                    assert before['metadata']['rejected_symbols'][sym]==['sector_participation_below_floor']
                    assert m['empty_participation_rescue_v5'] and sym in proof['alternate_top_six']
                    assert f['lagged_flow_5d']>0 and f['sector_intraday_ret']>0 and f['sector_intraday_breadth']>=.5 and f['sector_intraday_effective_count']>=3
                    if period['start']<=day<=period['end']:
                        prior=screened.pop((day,sym))
                        assert c['rank']==prior['rank'] and c['selection_score']==prior['score']
                        additions.append(dict(date=day,symbol=sym,rank=c['rank'],score=c['selection_score'],
                            six_bar_signal=prior['six_bar_signal'],signal_time=prior['signal_time']))
        assert not screened,(w,'Previously authenticated feasible candidates were lost',screened)
        evidence.append(dict(window=w,incumbent_membership_rank_size_inputs_and_features_exact=True,
            incumbent_candidates=incumbents,candidate_snapshot_path=str(snapshot_path),candidate_snapshot_sha256=n.sha(snapshot_path),
            additions=additions,confirmed_candidates=len(additions),six_bar_signals=sum(x['six_bar_signal'] for x in additions),
            signal_days=len({x['date'] for x in additions if x['six_bar_signal']}),
            diagnostic_screen_sha256=n.sha(t.REPO/f'data/backtests/analysis/olr_repair_failure_20261011/frequency_feasibility_{w}.json')))
    n.write(t.OUT/'preflight_receipt.json',dict(passed=True,utc=s.utc(),registration_sha256=n.sha(t.ROOT/'registration.json'),
        sources=n.read(t.ROOT/'preparation_complete.json')['sources'],evidence=evidence,script_sha256=n.sha(__file__),
            outcome_scope='No future return inspected. Candidates and signals are not funded trades.',
            identity_exception='Only candidate metadata.research_config_hash changes for the new config schema; all other candidate fields must be exactly equal.'))
    print({x['window']:{k:x[k] for k in ('confirmed_candidates','six_bar_signals','signal_days')} for x in evidence})


if __name__=='__main__':main()

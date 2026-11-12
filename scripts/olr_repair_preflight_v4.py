"""Source-bound candidate reconstruction and unchanged-incumbent feature checks."""
from pathlib import Path
from scripts import olr_repair_study_v4 as t
n,s,b=t.n,t.s,t.b

def main():
    assert not (t.ROOT/'cases').exists(),'Preflight before native outcomes only'
    plan=n.read(t.ROOT/'candidate_plan.json');evidence=[]
    for w in ('train','oos'):
        # Runtime paths/configurations are prepared without executing trades.
        for c in plan['candidates']:
            p=n.prepare_case(t.ROOT,w,c,100_000_000)
            if c['label']=='empty_afternoon_flow_rescue':rescue=n.read(p)
        ready=n.read(t.ENGINES/w/'ready.json');files=Path(ready['files'])
        old={r['trade_date']:r for r in n.read(files/ready['study']['olr_snapshots'])}
        new={r['trade_date']:r for r in n.read(files/rescue['olr_snapshots'])}
        assert set(old)==set(new)
        additions=[]
        for day,prior in old.items():
            after=new[day];proof=after['metadata']['empty_sector_flow_rescue_v4']
            assert proof['baseline_empty']==(not bool(prior['candidates'])),(w,day,'base membership changed')
            if prior['candidates']:
                assert [x['symbol'] for x in prior['candidates']]==[x['symbol'] for x in after['candidates']],(w,day)
                for a,z in zip(prior['candidates'],after['candidates']):
                    for key in ('symbol','rank','selection_score','daily_atr','sector','prior_day_close','prior_day_high','prior_day_low','expected_5m_volume'):
                        assert a[key]==z[key],(w,day,key,a[key],z[key])
                    assert a['metadata']['afternoon_features']==z['metadata']['afternoon_features'],(w,day,'feature migration')
                    assert not z['metadata'].get('empty_sector_flow_rescue_v4')
            for candidate in after['candidates'] if not prior['candidates'] else []:
                sym=candidate['symbol'];m=candidate['metadata'];f=m['afternoon_features']
                assert m['empty_sector_flow_rescue_v4'] and prior['metadata']['afternoon_rejected_symbols'][sym]==['lagged_sector_flow_below_floor']
                assert f['lagged_flow_5d']>0 and f['sector_intraday_ret']>0 and f['sector_intraday_breadth']>=.5 and f['sector_intraday_effective_count']>=3
                assert after['generated_at'][:10]==day and '14:30:00' in after['generated_at']
                additions.append(dict(date=day,symbol=sym,rank=candidate['rank'],score=candidate['selection_score'],stock_flow=f['lagged_flow_5d'],sector_return=f['sector_intraday_ret'],sector_breadth=f['sector_intraday_breadth']))
        evidence.append(dict(window=w,sessions=len(old),incumbent_membership_rank_size_inputs_and_features_exact=True,
            candidate_snapshot_sha256=n.sha(files/rescue['olr_snapshots']),candidate_snapshot_path=str(files/rescue['olr_snapshots']),
            new_candidates=additions,scope='Candidate feasibility only; no fill, P&L or frequency claim. Includes context-only IS dates before scoring boundary.'))
    n.write(t.OUT/'preflight_receipt.json',dict(passed=True,utc=s.utc(),registration_sha256=n.sha(t.ROOT/'registration.json'),
        sources=n.read(t.ROOT/'preparation_complete.json')['sources'],evidence=evidence,script_sha256=n.sha(__file__)))
    print({'passed':True,'new_candidate_counts':{x['window']:len(x['new_candidates']) for x in evidence}})

if __name__=='__main__':main()

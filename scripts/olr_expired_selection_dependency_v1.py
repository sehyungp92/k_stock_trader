"""Resolve0039 remaining state/funding interactions against accepted0035 full."""
import argparse
from collections import defaultdict,Counter
from datetime import datetime
import math
from pathlib import Path
import sys
from olr_selective_half_continuation_v1 import read,write,sha,now,status
from olr_opportunity_continuity_v1 import ledger


def run(root,out):
    out.mkdir(parents=True,exist_ok=False)
    design=Path(__file__).resolve().parents[1]/'docs/optimisation/olr/experiments/olr-exp-0039.md'
    source=root.parent/'opportunity_continuity_v2/train_rows.json'
    previous=root/'postmortem_v1/assessment.json'
    parent=root.parent/'relative_context_v2/assessment.json';full=read(parent)['summaries']['full'];case=Path(full['case'])
    ready=read(root/'native_engines/train/ready.json');files=Path(ready['files']);snapshot_path=files/ready['study']['olr_snapshots']
    bindings={str(p):sha(p) for p in [Path(__file__),source,previous,parent,case/'completion.json',snapshot_path]}
    write(out/'protocol.json',dict(at=now(),inputs=bindings,design_sha256=sha(design),question='Does accepted contextual retirement alter the remaining native eligibility/funding constraint, including003670 recovery/renewal? No counterfactual trade or new threshold.'))
    (out/'design_at_freeze.md').write_bytes(design.read_bytes());(out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    assert sha(case/'completion.json')==full['completion_sha256']
    rows=read(source);by_day=defaultdict(list)
    for r in rows:by_day[r['day']].append(r)
    outside=[r for r in rows if r['disposition']=='outside_slots'];assert len(outside)==24
    points=sorted({datetime.fromisoformat(r['timeline'][-1]['at']) for r in outside})
    status(out,'reconstructing_full_context_existing_book')
    book=ledger(case,points)
    snapshots={s['trade_date']:s for s in read(snapshot_path)}
    oldextra={(r['symbol'],r['day']) for r in read(previous)['extra_rows']}
    decisions=[]
    for day in sorted({r['day'] for r in outside}):
        group=by_day[day];original=sorted((r for r in group if r['rank']<=6),key=lambda r:r['rank'])
        at=original[0]['timeline'][-1]['at'];state=book['snapshots'][at]
        def held_or_pending(s):return bool(state['quantities'].get(s) or any(o['symbol']==s and o['side']=='BUY' for o in state['pending']))
        expired=[r['symbol'] for r in original if r['timeline'][-1]['source_complete'] and not r['first_confirmation'] and not held_or_pending(r['symbol'])]
        extras=sorted((r for r in group if r['rank']>6),key=lambda r:r['rank'])
        eligible=[r for r in extras if r['timeline'][-1]['source_complete'] and r['timeline'][-1]['passes'] and not held_or_pending(r['symbol'])]
        admitted={r['symbol'] for r in eligible[:len(expired)]}
        candidates={c['symbol']:c for c in snapshots[day]['candidates']}
        for r in extras:
            if r['disposition']!='outside_slots':continue
            z=r['timeline'][-1];c=candidates[r['symbol']]
            reason='admitted_by_unchanged_expiry' if r['symbol'] in admitted else 'held_or_pending' if held_or_pending(r['symbol']) else 'no_final_confirmation' if not z['passes'] else 'no_expired_priority' if not expired else 'higher_rank_extra_uses_priority'
            parent_state=z['book'];parent_held=bool(parent_state['quantities'].get(r['symbol']) or any(o['symbol']==r['symbol'] and o['side']=='BUY' for o in parent_state['pending']))
            current_reason='held_or_pending' if parent_held else 'no_final_confirmation' if not z['passes'] else 'admitted_actual' if (r['symbol'],day) in oldextra else 'original_or_higher_rank_priority'
            price=z['close']*1.015;unit=price*1.0007;risk=max(c['daily_atr'],z['close']*.005,1.);loss=1.25*risk+price*.0034
            weight=1.2*r['rank']**-1.5/sum(o['rank']**-1.5 for o in original)
            limits=dict(weight=math.floor(state['equity']*weight/unit),cash=math.floor(max(0,state['deployable'])/unit),
                gross=math.floor(max(0,state['equity']-state['gross']-state['reserve'])/unit),
                name=math.floor(max(0,.04*state['equity']-state['risk'].get(r['symbol'],0))/loss),
                sector=math.floor(max(0,.06*state['equity']-state['sector_risk'].get(c['sector'],0))/loss),
                account=math.floor(max(0,.08*state['equity']-sum(state['risk'].values()))/loss))
            upper=min(limits.values()) if r['symbol'] in admitted else 0
            decisions.append(dict(symbol=r['symbol'],day=day,at=at,rank=r['rank'],first_confirmation=r['first_confirmation'],final_passes=z['passes'],
                observed_access_only_reason=current_reason,full_context_reason=reason,expired_originals=expired,
                rank_order_eligible=[x['symbol'] for x in eligible],newly_eligible=r['symbol'] in admitted and (r['symbol'],day) not in oldextra,
                full_context_book=state,limits=limits,quantity_upper_bound=upper,notional_upper_bound=upper*price,
                scope='After actual existing full-context native cohort. Upper bound excludes fresh capacity, extra-to-extra funding and resulting future path; not a combined-policy simulation.'))
    assert len(decisions)==24
    write(out/'assessment.json',dict(at=now(),rows=decisions,newly_eligible=[r for r in decisions if r['newly_eligible']],
        final_invalid_earlier_confirmed=[r for r in decisions if not r['final_passes'] and r['first_confirmation']],
        exact_remaining_cases=[r for r in decisions if (r['symbol'],r['day']) in {('003670','2025-10-15'),('055550','2025-10-22'),('053800','2026-01-28')}],
        actual_gross_limited_comparators=[r for r in decisions if (r['symbol'],r['day']) in {('373220','2025-10-22'),('090430','2025-11-12')}],
        full_context_reasons=dict(Counter(r['full_context_reason'] for r in decisions)),new_financial_accounts=0,economic_objective_achieved=False))
    for p,h in bindings.items():assert sha(Path(p))==h
    status(out,'complete_awaiting_disposition')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.root,a.out)

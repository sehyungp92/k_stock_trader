"""0039 predefined complete-population lifecycle/funding review; no policy changes."""
from collections import defaultdict
from datetime import datetime, timedelta
import argparse
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys
import psutil
from olr_selective_half_continuation_v1 import read,write,sha,now,env_for
from olr_relative_context_assessment_v1 import occupancy
from olr_opportunity_continuity_v1 import ledger


def window(root,out,w):
    assessment=read(root/'assessment.json');summaries=assessment['summaries']
    ready=read(root/'native_engines'/w/'ready.json');sys.path.insert(0,ready['files'])
    from strategy_common.execution_session import session_cutoff
    from strategy_olr.execution import DECISION_CUTOFF
    census_path=root.parent/'opportunity_continuity_v2'/(w+'_rows.json')
    census={(r['symbol'],r['day']):r for r in read(census_path)}
    checks=[];trades={};contexts={};occupancies={};decisions={}
    for p in ('control','access'):
        case=Path(summaries[w+'_'+p]['case']);completion=read(case/'completion.json')
        accepted=read(root/'accepted'/(w+'_'+p+'.json'))
        assert sha(case/'completion.json')==accepted['completion_sha256']
        for n,h in completion['artifacts'].items():
            assert sha(case/n)==h,(p,n)
            checks.append(dict(policy=p,artifact=n,sha256=h))
        raw=read(case/'trades.json');trades[p]={(t['symbol'],t['entry_decision_time'][:10]):t for t in raw}
        assert len(trades[p])==len(raw)
        contexts[p]={}
        for r in read(case/'funded_population_receipt.json')['entry_contexts']:
            for d in r['decisions']:
                proposal=r['proposals'][d['intent_id']];key=(proposal['symbol'],r['at'])
                v=dict(at=r['at'],proposal=proposal,decision=d,account=r['account'],cash=r['actual_deployable_cash'])
                if key in contexts[p]:assert contexts[p][key]==v
                contexts[p][key]=v
        decisions[p]=[]
        with gzip.open(case/'native_decisions.jsonl.gz','rt',encoding='utf8') as stream:
            for line in stream:
                r=json.loads(line)
                if r['decision_code'].startswith('ENTRY'):decisions[p].append(r)
        occupancies[p]=occupancy(case)
        if w=='oos':
            study=read(root/'runtime'/w/p/'study.json')
            for key,n in [('capacity_checkpoint','capacity_terminal_state.json'),('capacity_checkpoint_completion','completion.json'),('capacity_checkpoint_result','result.json')]:
                assert sha(Path(ready['files'])/study[key])==sha(root/'cases/train'/p/n),(p,key)
            restart=read(case/'capacity_restart_receipt.json')
            write(out/(p+'_oos_restart.json'),restart)
    extra=[r for r in contexts['access'].values() if r['proposal']['metadata'].get('expired_selection_v1')]
    points=sorted({datetime.fromisoformat(r['at']) for r in extra})
    books={p:ledger(Path(summaries[w+'_'+p]['case']),points) for p in ('control','access')}
    traces=[]
    for r in extra:
        a=r['proposal'];m=a['metadata'];e=m['expired_selection_v1'];at=datetime.fromisoformat(r['at'])
        symbol=a['symbol'];key=(symbol,at.date().isoformat())
        cutoff=session_cutoff(at.date(),DECISION_CUTOFF)
        assert at==datetime.combine(at.date(),cutoff,at.tzinfo)+timedelta(minutes=30)
        assert e['available_at']==r['at'] and datetime.fromisoformat(e['source_label'])+timedelta(minutes=5)==at
        assert symbol in e['candidates'] and m['candidate_rank']>6
        assert len(e['candidates'])<=len(e['expired']) and not set(e['reserved'])&set(e['expired'])
        assert m['allocation_weight']==e['candidates'][symbol]['weight']
        independent=census[key]
        source=next(z for z in independent['timeline'] if z['at']==r['at'])
        assert independent['disposition']=='outside_slots' and source['source_complete'] and source['passes']
        assert source['close']==m['entry_submission_close']
        assert all(not census[(s,key[1])]['first_confirmation'] for s in e['expired'])
        assert not books['access']['snapshots'][r['at']]['quantities'].get(symbol)
        t=trades['access'].get(key);orders=[]
        for o in books['access']['orders'].values():
            if o['symbol']!=symbol:continue
            submitted=datetime.fromisoformat(o['submitted_at'])
            if t and at-timedelta(minutes=5)<=submitted<=datetime.fromisoformat(t['exit_fill_time']):orders.append(o)
            elif not t and o['side']=='BUY' and submitted.date()==at.date():orders.append(o)
        buys=[o for o in orders if o['side']=='BUY'];assert len(buys)<=1
        if t:
            assert m['risk_per_share']==t['route_metadata']['risk_per_share']
            acquired=sum(x['fill_qty'] for o in buys for x in o['events'] if x['kind']=='fill')
            sold=sum(x['fill_qty'] for o in orders if o['side']=='SELL' for x in o['events'] if x['kind']=='fill')
            assert acquired==sold==t['qty'],(key,acquired,sold,t['qty'])
            assert all(datetime.fromisoformat(x['at'])>at for o in buys for x in o['events'] if x['kind']=='fill')
        traces.append(dict(symbol=symbol,day=key[1],decision=r,independent_source=source,
            control_proposals=[x for (s,t),x in contexts['control'].items() if s==symbol and t[:10]==key[1]],
            all_access_entry_decisions=[x for x in decisions['access'] if x['symbol']==symbol and x['timestamp'][:10]==key[1]],
            trade=t,orders=orders,control_trade=trades['control'].get(key),
            book_at_boundary={p:books[p]['snapshots'][r['at']] for p in books}))
    pair=assessment['comparisons'][w];per=pair['accounting']['per_opportunity']
    targeted={(r['symbol'],r['day']) for r in traces}
    meaningful=lambda t:bool(t and t['qty']*t['entry_price']>=1e6)
    keys=trades['control'].keys()|trades['access'].keys()
    changed={k for k in keys if meaningful(trades['control'].get(k))!=meaningful(trades['access'].get(k))}
    groups=defaultdict(float)
    for r in per:groups['extra_lifecycles' if (r['symbol'],r['day']) in targeted else 'other_opportunities']+=r['delta_pnl']
    assert math.isclose(sum(groups.values()),pair['delta']['return_pct']*1e6,abs_tol=.01)
    changed_rows=[r for r in per if (r['symbol'],r['day']) in changed]
    original_differences=[]
    for key in sorted(contexts['control'].keys()|contexts['access'].keys(),key=lambda k:(k[1],k[0])):
        a=contexts['control'].get(key);b=contexts['access'].get(key)
        if b and b['proposal']['metadata'].get('expired_selection_v1'):continue
        if a!=b:
            original_differences.append(dict(symbol=key[0],at=key[1],control=a,access=b))
    # Full raw context includes provenance fields. Identify actual allocation differences separately.
    changes=[r for r in original_differences if not r['control'] or not r['access']
        or r['control']['decision']['approved_qty']!=r['access']['decision']['approved_qty']]
    write(out/(w+'_assessment.json'),dict(at=now(),completion_checks=checks,window=w,extra_traces=traces,census_sha256=sha(census_path),
        extra_contacts=len(extra),extra_dates=len({r['day'] for r in traces}),
        extra_acquisitions=sum(r['trade'] is not None for r in traces),
        extra_meaningful=sum(meaningful(r['trade']) for r in traces),
        extra_profitable=sum(bool(r['trade'] and r['trade']['net_pnl']>0) for r in traces),
        targeted_vs_other=dict(groups),changed_meaningful=changed_rows,
        changed_meaningful_pnl=sum(r['delta_pnl'] for r in changed_rows),
        changed_original_allocations=changes,first_changed_original_allocation=changes[0] if changes else None,
        occupancy=occupancies,scope='Exhaustive extra proposals and changed meaningful identities. Additive accounting does not identify independent causal mediation; OOS has no independent new population.'))


def run(root,out):
    out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(at=now(),runner_sha256=sha(__file__),assessment_sha256=sha(root/'assessment.json'),
        method='All actual extras, all displaced meaningful identities; own-policy history, cash, source-time, order and lifecycle reconciliation. No future-label rule selection.'))
    for w in ('train','oos'):
        cmd=[sys.executable,'-X','utf8','-B','-u',str(Path(__file__).resolve()),str(root),'--out',str(out),'--window',w]
        with (out/(w+'.log')).open('x',encoding='utf8') as f:
            p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env_for(root),creationflags=0x08000000)
            write(out/(w+'_process.json'),dict(at=now(),pid=p.pid,created_at=psutil.Process(p.pid).create_time(),command=cmd))
            assert p.wait()==0,'Preserve failed extraction: '+str(out/(w+'.log'))
    write(out/'assessment.json',dict(at=now(),windows={w:read(out/(w+'_assessment.json')) for w in ('train','oos')},
        economic_objective_achieved=False,next='Substantive diagnosis and principal-solution readiness decision; no automatic variant.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path);p.add_argument('--window',choices=['train','oos']);a=p.parse_args()
    out=a.out or a.root/'causal_review_v1'
    window(a.root,out,a.window) if a.window else run(a.root,out)

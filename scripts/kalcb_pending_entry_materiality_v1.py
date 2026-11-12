"""Finite existing-account entry-remainder materiality; zero native replays."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
BASE = Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
SOURCE = BASE/'v49_staged_support'
HELPER = REPO/'backtests/strategies/kalcb/pending_entry_materiality_v1.py'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def bound(p):
    p=Path(p).resolve()
    with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p),'sha256':sha}
def verify(bs):
    for b in bs:
        if bound(b['path'])['sha256']!=b['sha256']:raise ValueError('Changed source '+b['path'])
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)


def register(root,tests):
    t=read(tests);assert t['passed'] and t['tests']==8
    verify(t['programs']+[t['log']])
    original=read(SOURCE/'protocol.json');done=read(SOURCE/'completion.json')
    assert read(SOURCE/'review_v1/completion.json')['passed']
    verify(original['programs']+original['bindings']+done['outputs']+[done['assessment']])
    paths=[SOURCE/x for x in ['protocol.json','completion.json','review_v1/completion.json',
        'review_v1/review.json','post_completion_health.json','economic_decision.json']]
    paths += [Path(tests)]
    cases=original['cases']
    for c in cases:
        paths.extend([SOURCE/'accounts'/Path(c['key']).with_suffix('.json'),Path(c['folder'])/'fills.json'])
    for name in ['kalcb_continuation_implementation_20261006.txt',
                 'kalcb_calibrated_next_steps_20261007.txt','kalcb_campaign_implementation_20261007.txt']:
        paths.append(REPO/'docs/research'/name)
    root.mkdir(exist_ok=False)
    write(root/'protocol.json',dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),
        kind='existing_parent_pending_entry_materiality',cases=cases,source=str(SOURCE),
        population='All2620 original submitted opening intents in12 unchanged accounts; no outcome/date/issuer subset. Reuse V49 independently reviewed completed7bar evidence and original cash lifetimes.',
        fixed_question='At the already declared09:35 completed clock, how much original entry exposure is still unfilled when the frozen V47/V49 price condition is false? Inspect remaining original BUY fills, original full-lifetime winners/losses and costs. Preserve early acquired shares conceptually; no actual cancellation or new policy.',
        diagnostic_cohort='Supported7bar prefix; original price confirmation false; original requestedqty minus originalBUYfills strictlybefore35 >0; no original sale before35 or parent exit intent through35. Future fills do NOT select this cohort. Remaining requestedqty does not prove pending status.',
        outputs='All original orders; before/future exact integer quantities, actual future purchase notional/fees/price-reference differences; separate future-fill-observed group, monthly/issuer support and once-only whole-lifetime PnL. Future data is diagnostic only.',
        limitations='No cancellation PnL, saved loss, attainable exit, pro-rata PnL, forecast, training, policy or funded counterfactual. Original order journals lack an independent time on some terminal/amend events; exact pending state requires a later separately frozen native-state pilot if economics justify one. Scenarios/priors/partials dependent. Historical full source/account/MTM qualification reused, not rerun.',
        stop_rule='No automatic native expansion. Require material exposure, credible opportunity-cost case and actual native pending-state reconstruction before any separate finite cancellation experiment. Preserve all gates/clocks/costs and unchanged priors. Existing-data only.',
        estimate='12 existing accounts; approximately2-5minutes including independent review, zero native cases/fits.',
        programs=[bound(REPO/x) for x in ['scripts/kalcb_pending_entry_materiality_v1.py',
            'scripts/kalcb_pending_entry_materiality_review_v1.py','scripts/kalcb_pending_entry_materiality_pipeline_v1.py',
            'backtests/strategies/kalcb/pending_entry_materiality_v1.py','tests/test_kalcb_pending_entry_materiality_v1.py']],
        bindings=[bound(x) for x in dict.fromkeys(paths)],study_complete=False))


def run(root):
    import psutil
    p=read(root/'protocol.json');verify(p['programs']+p['bindings'])
    me=psutil.Process();write(root/'supervisor_start.json',dict(pid=me.pid,process_created=me.create_time(),command=me.cmdline()))
    s=importlib.util.spec_from_file_location('frozen_pending_entry_materiality',HELPER)
    h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
    outputs=[];accounts={}
    for c in p['cases']:
        old=read(Path(p['source'])/'accounts'/Path(c['key']).with_suffix('.json'))
        fills=read(Path(c['folder'])/'fills.json');by_order=defaultdict(list)
        for f in fills:
            if f['side']=='BUY':by_order[f['order_id']].append(f)
        rows=[h.measure(r,by_order[r['order_id']]) for r in old['rows']]
        groups={'all':rows,'potential_remainders':[r for r in rows if r['potential_remainder_observation']],
            'potential_with_later_fills':[r for r in rows if r['potential_remainder_observation'] and r['later_qty']>0],
            'confirmed_with_later_fills':[r for r in rows if r['prefix_supported'] and not r['weak_price_at35'] and r['later_qty']>0]}
        result={'account':c['key'],'rows':rows,'groups':{k:h.summary(v) for k,v in groups.items()},
            'months':{m:{k:h.summary([r for r in v if r['day'][:7]==m]) for k,v in groups.items()}
                      for m in sorted({r['day'][:7] for r in rows})},
            'issuers':{symbol:h.summary([r for r in groups['potential_with_later_fills'] if r['symbol']==symbol])
                       for symbol in sorted({r['symbol'] for r in groups['potential_with_later_fills']})},
            'new_policy_or_outcomes':False}
        out=root/'accounts'/Path(c['key']).with_suffix('.json');write(out,result);outputs.append(bound(out))
        accounts[c['key']]=result['groups']
        print(json.dumps({'account':c['key'],'all':len(rows),'potential_with_later_fills':len(groups['potential_with_later_fills'])}),flush=True)
        (root/'progress.json').write_text(json.dumps({'complete_accounts':len(outputs),'expected':12}),encoding='utf8')
        del fills,by_order,rows,old,result
    verify(p['programs']+p['bindings'])
    write(root/'assessment.json',dict(accounts=accounts,native_cases=0,fits=0,study_complete=False))
    write(root/'completion.json',dict(complete=True,outputs=outputs,assessment=bound(root/'assessment.json'),
        at_utc=datetime.now(timezone.utc).isoformat(),study_complete=False))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['register','run']);a.add_argument('--root',type=Path,required=True);a.add_argument('--tests',type=Path);args=a.parse_args()
    try:register(args.root,args.tests) if args.mode=='register' else run(args.root)
    except Exception as e:
        if args.root.exists() and not (args.root/'failure.json').exists():write(args.root/'failure.json',{'error':repr(e),'preserve_evidence':True})
        raise

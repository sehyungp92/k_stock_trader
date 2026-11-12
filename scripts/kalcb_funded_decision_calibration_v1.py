"""Finite target-contract contrast on existing audited opening accounts."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import argparse, gzip, hashlib, importlib.util, json, math, statistics, sys

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
SOURCE=ROOT/'v42_decision_value_reconciliation_v2'
HELPER=REPO/'backtests/strategies/kalcb/funded_decision_calibration_v1.py'
CALENDAR=ROOT.parent/'kalcb_aligned_walk_forward_20261010/v3/fit/calendar.json'
PARENTS=('control','control_prior125');SCENARIOS=('baseline','combined')

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(x) for x in f]
def save(p,d):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,allow_nan=False)
def bound(p):return {'path':str(p),'sha256':sha(p)}
def verify(bs):
    for b in bs:
        if sha(b['path'])!=b['sha256']:raise ValueError('Binding changed '+b['path'])
def load_helper():
    spec=importlib.util.spec_from_file_location('registered_funded_decision_calibration',HELPER)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def register(out,tests):
    done=read(SOURCE/'completion.json');assert done['complete'];verify(done['outputs'])
    assert read(tests)['passed']
    files=[SOURCE/'completion.json',SOURCE/'reconciled_decisions.jsonl.gz',CALENDAR,Path(tests),
           ROOT/'v25_pending_funding_screen/assessment_v1/assessment.json']
    for parent in PARENTS:
      for period in ('is','oos'):
       for scenario in SCENARIOS:
        folder=ROOT/f'v25_pending_funding_screen/cases/{parent}/{period}/{scenario}'
        files.extend(folder/n for n in ('completion.json','result.json','observer_evidence.json','orders.jsonl.gz','fills.json','trades.json','independent_exposure_mtm.json'))
    out.mkdir(exist_ok=False)
    save(out/'protocol.json',{'kind':'finite_execution_aligned_target_contrast','registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'hypothesis':'Current final values overpredict retained opening actions. Test whether calibrating on actual selected native decisions improves economic abstention, separating population alignment from funded-target alignment.',
        'recipes':['isolated','funded'],'parents':PARENTS,'scenarios':SCENARIOS,
        'common_population':'Only originally retained opening actions submitted by each unchanged parent; one issuer-day/order intent, losses and zero fills retained. Common valid/mature paired target cohort; unavailable labels retained but excluded from both fits. No pooling across parents/scenarios/origins/quantities.',
        'target_contrast':'Same final forecast, same selected population, same fixed affine rule. isolated uses authenticated exact-reference selected-management net/planned-notional labels. funded uses actual unique lifetime net including all costs divided by ACTUAL submitted quantity times completed decision close; verified no-fill orders target zero. Return-rate target is NOT scaled counterfactual PnL.',
        'model':'One identity-shrunk affine map per origin/quantity, prior100, slope0..1. Session-balanced weights; prior-only forecast min/max domain. No extra features/penalty/threshold sweep.',
        'chronology':'Original20-session base dates, one-session embargo, min60 mature selected sessions and100issuer-days globally; min20issuer-days/8sessions per map. No preIS backfill, no April reset. Both scenarios must support before abstention.',
        'action':'Additional veto only on currently retained opening action if either supported scenario predicts nonpositive value. Preserve all incumbent ranks/scores/quantities/exits, earlier gates and later routes. No forced entries/restoration/bundles.',
        'readiness':'Before native replay require20changed IS issuer-days/8sessions; report OOS support separately without relaxing. Both scenario IS direct foregone native lifetime net must be nonpositive (removing net losses rather than sacrificing aggregate winners). Report all lost winners, actual notional, meaningful activity, months, concentration and max-DD overlap. This is only a necessary observational screen, NOT counterfactual portfolio improvement. OOS cannot rescue a failed IS screen. No automatic native launch.',
        'limits':'Conditioned on unchanged parents. Changed account/funding descendants need actual future native replay; full stress/funded qualification remains mandatory for a credible candidate. April-July reused development.',
        'estimate':'5-10 minutes including independent review; zero new labels or native cases. A passing candidate requires a separately finite native protocol.',
        'programs':[bound(Path(__file__).resolve()),bound(HELPER),bound(REPO/'tests/test_kalcb_funded_decision_calibration_v1.py')],
        'bindings':[bound(p) for p in files]})

def run(out):
    protocol=read(out/'protocol.json');verify(protocol['programs']+protocol['bindings'])
    if (out/'run_receipt.json').exists():raise ValueError('Already run; preserve and version failures')
    save(out/'run_receipt.json',{'argv':sys.argv,'started_at_utc':datetime.now(timezone.utc).isoformat()})
    m=load_helper();calendar=read(CALENDAR)['sessions'];assert calendar.index('2026-04-01')==217
    reconciliation=rows(SOURCE/'reconciled_decisions.jsonl.gz')
    selected={(r['parent'],r['day'],r['value']['action_id'].split('/')[1]):r for r in reconciliation if r['value']['retained']}
    audit=read(ROOT/'v25_pending_funding_screen/assessment_v1/assessment.json')
    inputs_by_path={str(Path(b['path']).resolve()):b for b in audit['bindings']}
    datasets=defaultdict(list);targets=[];summaries=[]
    for parent in PARENTS:
      for period in ('is','oos'):
       for scenario in SCENARIOS:
        folder=ROOT/f'v25_pending_funding_screen/cases/{parent}/{period}/{scenario}'
        for name in ('observer_evidence.json','fills.json','orders.jsonl.gz','trades.json'):
            b=inputs_by_path[str((folder/name).resolve())];verify([b])
        complete=read(folder/'completion.json');assert complete['complete'];verify([complete['evidence']])
        result=read(folder/'result.json')
        assert result['audit_status']=='passed' and result['execution_validity_pass'] and not result['terminal_inventory'] and not result['terminal_pending']
        evidence=read(folder/'observer_evidence.json');positions=audit['accounts'][f'{parent}/{period}/{scenario}']['positions']
        # Normalize the observed order journal's schema; do not invent statuses.
        events=[{**e,'order_id':e.get('order_id',e.get('order',{}).get('order_id'))} for e in rows(folder/'orders.jsonl.gz')]
        exposures=[e for e in evidence['entry_exposures'] if e['action']['metadata'].get('entry_route')=='first30_open_anchor']
        seen=set();aggregate=0.;nofills=0
        for e in exposures:
            day=e['decision_available_at'][:10];symbol=e['action']['symbol'];key=(parent,day,symbol)
            if key in seen:raise ValueError('Multiple submitted opening intents')
            seen.add(key);r=selected[key];v=r['value']
            if not math.isclose(e['action']['metadata']['frontier_selection_score'],v['selection_score'],abs_tol=1e-12):raise ValueError('Actual selection score mismatch')
            t=m.native_target(e,positions,events,day)
            if not any(x['event']=='submit' and x['order_id']==t['order_id'] for x in events):raise ValueError('Missing submission')
            if not t['filled_quantity'] and not any(x['event'] in ('reject','cancel','expire') and x['order_id']==t['order_id'] for x in events):raise ValueError('No terminal no-fill journal')
            aggregate+=t['net_pnl_krw'];nofills+=not t['filled_quantity']
            labels=r['labels'];valid=all(labels[s]['label_valid'] for s in SCENARIOS)
            maturity=max([m.aware(t['mature_at']),*[m.aware(labels[s]['label_available_at']) for s in SCENARIOS]])
            d={'id':v['action_id'],'parent':parent,'scenario':scenario,'period':period,'day':day,
                'origin':v['origin'],'quantity_scale':v['scale'],'available_at':v['available_at'],
                'forecast_valid_from':max(v['base_valid_from'],v['calibration_valid_from']),
                'prediction':v['final'][scenario]/v['notional_equity'],
                'target_valid':valid,'mature_at':maturity.isoformat(),
                'isolated':labels[scenario]['net_value_equity_bps']/v['notional_equity'] if valid else None,
                'funded':t['value'],'native':t,'decision_equity':e['account_before']['actual_equity']}
            datasets[parent,scenario].append(d);targets.append(d)
        wanted=[k for k,r in selected.items() if k[0]==parent and r['period']==period]
        if set(wanted)!=seen:raise ValueError('Missing selected decision including no-fills')
        expected=sum(p['net_pnl_krw'] for p in positions if p['route']=='first30_open_anchor')
        if not math.isclose(aggregate,expected,abs_tol=.01):raise ValueError('Duplicate or missing native PnL')
        summaries.append({'parent':parent,'period':period,'scenario':scenario,'decisions':len(exposures),
                          'no_fill_decisions':nofills,'unique_lifetime_net_krw':aggregate,'exact_account_bridge':True})
    # Pair scenario maturity without counting the pair as two independent
    # issuer-days. A late completion in either account delays both fits.
    paired=defaultdict(list)
    for r in targets:paired[r['parent'],r['id']].append(r)
    for pair in paired.values():
        if len(pair)!=2 or {r['scenario'] for r in pair}!=set(SCENARIOS):raise ValueError('Unpaired target')
        mature=max(m.aware(r['mature_at']) for r in pair).isoformat()
        valid=all(r['target_valid'] for r in pair)
        for r in pair:r.update(mature_at=mature,target_valid=valid)
    with gzip.open(out/'targets.jsonl.gz','xt',encoding='utf-8') as f:
        for r in targets:f.write(json.dumps(r,allow_nan=False)+'\n')
    # No outcomes from current/future folds enter any fit. The fold schedule
    # is the inherited base schedule, not dates chosen from model performance.
    firsts=sorted({r['value']['base_valid_from'] for r in reconciliation})
    forecasts={};models=[]
    for parent in PARENTS:
      for scenario in SCENARIOS:
       data=sorted(datasets[parent,scenario],key=lambda r:(r['day'],r['id']))
       for target in ('isolated','funded'):
        for i,first in enumerate(firsts):
            through=firsts[i+1] if i+1<len(firsts) else '9999-12-31'
            proof=m.fit(data,first,calendar,target);models.append({'parent':parent,'scenario':scenario,**proof})
            for r in data:
                if first<=r['day']<through:
                    forecasts[parent,scenario,target,r['id']]={'first':first,**m.predict(r,proof)}
    report={};changes=[]
    for parent in PARENTS:
      baseline={r['id']:r for r in datasets[parent,'baseline']};combined={r['id']:r for r in datasets[parent,'combined']}
      if set(baseline)!=set(combined):raise ValueError('Scenarios must share selected decision population')
      for target in ('isolated','funded'):
       for period in ('is','oos'):
        selected_changes=[];errors={s:[] for s in SCENARIOS};supported=0
        for key,b in baseline.items():
            if b['period']!=period:continue
            c=combined[key];fb=forecasts[parent,'baseline',target,key];fc=forecasts[parent,'combined',target,key]
            if fb['supported'] and fc['supported']:supported+=1
            for scenario,r,f in [('baseline',b,fb),('combined',c,fc)]:
                if f['supported']:errors[scenario].append({'day':r['day'],'before':(r['prediction']-r['funded'])**2,'after':(f['value']-r['funded'])**2})
            if m.paired_abstention(fb,fc):
                item={'parent':parent,'target':target,'period':period,'id':key,'day':b['day'],
                      'predictions':{'baseline':fb,'combined':fc},'native':{'baseline':b['native'],'combined':c['native']},
                      'decision_equity':{'baseline':b['decision_equity'],'combined':c['decision_equity']}}
                changes.append(item);selected_changes.append(item)
        stats={'supported_both':supported,'changed_issuer_days':len(selected_changes),'changed_sessions':len({x['day'] for x in selected_changes}),'scenarios':{}}
        for scenario in SCENARIOS:
            es=errors[scenario];daily=defaultdict(list)
            for x in es:daily[x['day']].append(x)
            native=[x['native'][scenario] for x in selected_changes]
            meaningful=[x for x in selected_changes if x['native'][scenario]['filled_quantity']>1 and x['native'][scenario]['planned_notional_krw']>=.01*x['decision_equity'][scenario] and x['native'][scenario]['net_pnl_krw']>=.0001*x['decision_equity'][scenario]]
            stats['scenarios'][scenario]={'supported_errors':len(es),
                'session_balanced_mse_before':statistics.fmean(statistics.fmean(x['before'] for x in xs) for xs in daily.values()) if daily else None,
                'session_balanced_mse_after':statistics.fmean(statistics.fmean(x['after'] for x in xs) for xs in daily.values()) if daily else None,
                'foregone_native_net_krw':sum(x['net_pnl_krw'] for x in native),
                'positive_native_pnl_sacrificed_krw':sum(max(0,x['net_pnl_krw']) for x in native),
                'negative_native_pnl_removed_krw':sum(min(0,x['net_pnl_krw']) for x in native),
                'filled_decisions_removed':sum(x['filled_quantity']>0 for x in native),
                'profitable_decisions_removed':sum(x['net_pnl_krw']>0 for x in native),
                'meaningful_profitable_removed_posthoc_definition':len(meaningful),
                'scope':'Foregone original lifetimes only; NOT candidate return or recovery estimate.'}
        stats['necessary_is_screen_pass']=period=='is' and stats['changed_issuer_days']>=20 and stats['changed_sessions']>=8 and all(v['foregone_native_net_krw']<=0 for v in stats['scenarios'].values())
        report[f'{parent}/{target}/{period}']=stats
    save(out/'models.json',models)
    save(out/'forecasts.json',[{'parent':p,'scenario':s,'target':t,'id':k,**v} for (p,s,t,k),v in forecasts.items()])
    save(out/'assessment.json',{'complete':True,'target_bridges':summaries,'contrasts':report,'changes':changes,
         'native_cases':0,'policy_files':0,'no_improvement_claimed':True,'study_complete':False})
    verify(protocol['programs']+protocol['bindings'])
    save(out/'completion.json',{'complete':True,'study_complete':False,'native_cases':0,'policy_files':0,
        'outputs':[bound(out/n) for n in ('targets.jsonl.gz','models.json','forecasts.json','assessment.json')],
        'completed_at_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'complete':True,'decisions':len(targets),'contrast_count':len(report),'passed_is_screens':[k for k,v in report.items() if v['necessary_is_screen_pass']]}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','run']);p.add_argument('--root',type=Path,required=True);p.add_argument('--tests',type=Path)
    a=p.parse_args();register(a.root,a.tests) if a.command=='register' else run(a.root)

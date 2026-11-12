"""One preregistered decision-loss contrast against completed V44 models."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import argparse,gc,gzip,hashlib,importlib.abc,importlib.machinery,importlib.util,json,sys

REPO=Path(__file__).resolve().parents[1];ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
SOURCE=ROOT/'v44_paired_quantity_value';HELPER=REPO/'backtests/strategies/kalcb/paired_decision_regret_v1.py'
PAIR=REPO/'backtests/strategies/kalcb/paired_quantity_value_v1.py'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p)}
def verify(bs):
    for b in bs:
        if sha(b['path'])!=b['sha256']:raise ValueError('Changed '+b['path'])
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def lines(p):
    op=gzip.open if str(p).endswith('.gz') else open
    with op(p,'rt',encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def register(out,tests):
    assert read(tests)['passed'] and read(SOURCE/'review_v1/completion.json')['passed']
    p=read(SOURCE/'protocol.json');done=read(SOURCE/'completion.json')
    bindings=p['bindings']+p['programs']+done['outputs']+[bind(SOURCE/n) for n in ('protocol.json','completion.json','review_v1/completion.json','review_v1/review.json')]+[bind(tests)]
    verify(bindings);out.mkdir(exist_ok=False)
    save(out/'protocol.json',{'kind':'single_economic_decision_loss_contrast','registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'mechanism':'V44 OOS reductions help12/17 but sacrificed5winners outweigh gains. Test economic mistake-weighted decision learning on the SAME paired contrasts and SAME frozen potential basis, not a new feature or quantity sweep.',
        'objective':'Fixed logistic surrogate of paired economic regret: target=sign(high-minus-low full-cost value), sample weight=prior-only absolute value / train weighted mean absolute value times the unchanged issuer/session weight. Fixed coefficient penalty120, no free intercept. A rare valuable wrong decision costs more than a tiny loss. Conditional equivalent decision target, different finite-sample loss; improvement is unproven.',
        'score_contract':'economic-weighted log odds, NOT expected return, unweighted probability of profit or physical risk. Never inject into original value/risk gates. Zero sign is the fixed high-versus-low decision boundary.',
        'controls':'Authenticate/reuse all24 V44 paired_state fits and pair_mean components; exactly24 new regret fits. No refitting controls.',
        'invariants':'Identical complete existing paired labels/preIS/selected management/integer quantities/StateMap/12base windows/20session clock/60mature sessions/100issuerdays/one-session embargo/prior-only physical domain. Both-scenario support and lower original risk<.001/capacity remain. No more than one adjacent reduction, no entries/increases/reranking/exit changes.',
        'evaluation':'Compare equal-session value-weighted decision regret against V44 paired_state and pair_mean on common support, broad and actual parent subsets. Evaluate actual two-scenario proposed action contrasts, harmed winning tails, concentration and conditional5/10session uncertainty. No quantity/PnL scaling.',
        'native_readiness':'Require lower IS economic decision regret versus V44 paired_state in BOTH scenarios on selected parent actions, at least20distinct changed ISissuerdays/8sessions perparent, nonnegative isolated IS action contrasts bothscenarios; no materiality inferred from accuracy alone. Selected genuine forward action calibration and exact native-state support remain required before ANY policy. No automatic dispatch. OOS and tail evidence can reject, not rescue-tune.',
        'estimate':'24model fits, about3-8minutes plus independent review. No new labels/native cases. Failure closes this fixed paired-size learning comparison; no automatic penalty/feature/learner/threshold sweep.',
        'bindings':bindings,'programs':[bind(Path(__file__).resolve()),bind(HELPER),bind(PAIR),bind(REPO/'tests/test_kalcb_paired_decision_regret_v1.py')],
        'limits':'All economic effects here are isolated contrasts, not funded portfolio returns/DD. April-July reused development, no independent validation or promotion.'})
def run(out):
    p=read(out/'protocol.json');verify(p['bindings']+p['programs'])
    if (out/'supervisor_start.json').exists():raise ValueError('Already started')
    import psutil
    proc=psutil.Process();save(out/'supervisor_start.json',{'pid':proc.pid,'create_time':proc.create_time(),'cmdline':proc.cmdline(),'at_utc':datetime.now(timezone.utc).isoformat()})
    src=read(SOURCE/'protocol.json');profile=read(ROOT/'v25_pending_funding_screen/cases/control/oos/input_profile.json');manifest=read(profile['manifest']['path'])
    frozen=Path(profile['manifest']['path']).parent/'files';roots={Path(k).parts[0].removesuffix('.py') for k in manifest['files'] if k.endswith('.py')};origins={}
    class Gate(importlib.abc.MetaPathFinder):
        def find_spec(self,name,path=None,target=None):
            if name.split('.')[0] not in roots:return None
            spec=importlib.machinery.PathFinder.find_spec(name,path)
            if spec is None:raise ImportError(name)
            if spec.origin and spec.origin not in ('built-in','frozen'):
                file=Path(spec.origin).resolve();rel=file.relative_to(frozen.resolve()).as_posix();h=manifest['files'].get(rel)
                if isinstance(h,dict):h=h['sha256']
                if not h or sha(file)!=h:raise ImportError('Unbound '+name)
                origins[name]={'path':str(file),'sha256':h}
            return spec
    sys.path.insert(0,str(frozen));sys.meta_path.insert(0,Gate())
    m=load('registered_paired_regret',HELPER);pair=load('registered_pair_basis',PAIR)
    driver=load('registered_headroom_only',REPO/'scripts/kalcb_paired_quantity_study_v1.py')
    import numpy as np,joblib
    from threadpoolctl import threadpool_limits
    from backtests.strategies.kalcb import continuation_value as cv
    cal=read(driver.CAL);labels={s:{} for s in ('baseline','combined')};actions=[]
    for c in src['cells']:
        prepared=Path(c['prepared']);meta=read(prepared/'prepared.json')
        for scenario in labels:
            folder=driver.ALIGN/'labels'/f"{c['period']}_opening_{scenario}"
            for name in read(folder/'grid_manifest.json')['files']:
                if name.endswith('.jsonl'):
                    for r in lines(folder/name):labels[scenario][r['opportunity_id']]={k:r[k] for k in ('label_valid','label_available_at','net_value_equity_bps')}
        wanted=set(c['sessions'])
        for r in lines(prepared/'opportunities.jsonl'):
            if r['trade_date'] not in wanted or not r['action']['ready'] or not r['guard_at_rank1']:continue
            full=int(r['action']['quantity']);price=r['opening_bars'][-1]['close']
            for q in (.25,.5,1.):
                qty=int(full*q);features=dict(r['features'])
                for key in pair.PHYSICAL:features[key]*=qty/full
                features['action_quantity_scale']=q;identity=r['opportunity_id']+f'/q{q:g}'
                actions.append({'id':identity,'base_id':r['opportunity_id'],'day':r['trade_date'],'available_at':r['feature_available_at'],
                    'source':meta['opportunities_sha256'],'q':q,'qty':qty,'notional':qty*price/1e8,'features':features,'labels':{s:labels[s][identity] for s in labels}})
    actions.sort(key=lambda r:(r['day'],r['base_id'],r['q']));index={r['id']:i for i,r in enumerate(actions)};bases=sorted({r['base_id'] for r in actions})
    old={(r['first'],r['scenario'],r['id']):r for r in lines(SOURCE/'forecasts.jsonl.gz')};folds={(f['valid_from'],f['scenario']):f for f in src['folds']};models=[];forecasts=[]
    for first,last in cal['windows']:
      for scenario in labels:
        driver.wait_headroom(psutil);f=folds[first,scenario];native=joblib.load(f['model_path'])
        x=cv.matrix(actions,f['feature_columns']);z=native.features.transform(x);phi=pair.potential_basis([r['q'] for r in actions],z);pairs=[]
        for base in bases:
            for lo,hi in pair.PAIRS:
                i=index[base+f'/q{lo:g}'];j=index[base+f'/q{hi:g}'];r=pair.make_contrast(actions[i],actions[j],phi[i],phi[j],scenario)
                if r:pairs.append(r)
        with threadpool_limits(limits=1):proof=m.fit(pairs,first,cal['sessions'])
        proof.update(scenario=scenario,through=last,frozen_map_model_path=f['model_path'],frozen_map_model_sha256=f['model_sha256']);models.append(proof)
        for r in pairs:
            if not first<=r['day']<=last:continue
            control=old[first,scenario,r['id']];pred=m.predict(r,proof)
            assert pred['supported']==control['predictions']['paired_state']['supported']
            forecasts.append({**control,'regret_prediction':pred})
        print(json.dumps({'first':first,'scenario':scenario,'optimization':proof.get('optimization'),'supported':proof['supported']}),flush=True)
        del x,z,phi,pairs,native;gc.collect()
    save(out/'models.json',models)
    with gzip.open(out/'forecasts.jsonl.gz','xt',encoding='utf-8') as stream:
        for r in forecasts:stream.write(json.dumps(r,allow_nan=False)+'\n')
    lookup={(r['id'],r['scenario']):r for r in forecasts};changes=[];summary={};errors={}
    for parent in ('control','control_prior125'):
      for period in ('is','oos'):
        selected=[]
        for r in forecasts:
            if r['scenario']!='baseline' or parent not in r['high_parent_selections'] or ('is' if r['day']<='2026-03-31' else 'oos')!=period:continue
            c=lookup[r['id'],'combined'];ps=[v['regret_prediction'] for v in (r,c)]
            if all(v['lower_physical_capacity_pass'] for v in (r,c)) and all(v['supported'] and v['score']<0 for v in ps):
                row={'parent':parent,'period':period,'id':r['id'],'base_id':r['base_id'],'day':r['day'],'low':r['low'],'high':r['high'],
                    'scores':{s:lookup[r['id'],s]['regret_prediction']['score'] for s in labels},
                    'delta_isolated_equity_bps':{s:-lookup[r['id'],s]['value'] if lookup[r['id'],s]['target_valid'] else None for s in labels}}
                selected.append(row);changes.append(row)
        summary[parent+'/'+period]={'changed_issuer_days':len({r['base_id'] for r in selected}),'changed_sessions':len({r['day'] for r in selected}),
            'isolated_contrast_sum_not_portfolio_pnl':{s:sum(r['delta_isolated_equity_bps'][s] or 0 for r in selected) for s in labels},'invalid_current_targets':sum(any(v is None for v in r['delta_isolated_equity_bps'].values()) for r in selected)}
    for period in ('is','oos'):
      for scenario in labels:
       for cohort in ('all','control','control_prior125'):
        rs=[r for r in forecasts if r['scenario']==scenario and ('is' if r['day']<='2026-03-31' else 'oos')==period and r['target_valid'] and r['regret_prediction']['supported'] and (cohort=='all' or cohort in r['high_parent_selections'])]
        daily=defaultdict(list)
        for r in rs:daily[r['day']].append(r)
        losses={}
        for recipe in ('pair_mean','paired_state','decision_regret'):
            losses[recipe]=float(np.mean([np.mean([abs(r['value']) if ((r['regret_prediction']['score'] if recipe=='decision_regret' else r['predictions'][recipe]['value'])>0)!=(r['value']>0) else 0. for r in day]) for day in daily.values()]))
        errors[period+'/'+scenario+'/'+cohort]={'observations':len(rs),'sessions':len(daily),'session_balanced_economic_mistake_equity_bps':losses}
    save(out/'assessment.json',{'complete':True,'changes':changes,'summary':summary,'economic_decision_errors':errors,'scope':'Isolated pairwise decision-regret comparison. No portfolio return/DD/activity result.'})
    verify(p['bindings']+p['programs']);verify(list(origins.values()));save(out/'origins.json',{'passed':True,'origins':origins})
    save(out/'completion.json',{'complete':True,'study_complete':False,'native_cases':0,'policy_files':0,'outputs':[bind(out/n) for n in ('models.json','forecasts.jsonl.gz','assessment.json','origins.json')],'completed_at_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'complete':True,'summary':summary,'errors':errors}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','run']);p.add_argument('--root',type=Path,required=True);p.add_argument('--tests',type=Path);a=p.parse_args();register(a.root,a.tests) if a.command=='register' else run(a.root)

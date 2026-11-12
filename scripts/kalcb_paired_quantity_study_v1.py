"""Finite paired-quantity learning experiment, existing labels only."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import argparse,ctypes,gc,gzip,hashlib,importlib.abc,importlib.machinery,importlib.util,json,math,sys,time

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
ALIGN=ROOT.parent/'kalcb_selected_target_alignment_20261010/v2'
CAL=ROOT.parent/'kalcb_aligned_walk_forward_20261010/v3/fit/calendar.json'
HELPER=REPO/'backtests/strategies/kalcb/paired_quantity_value_v1.py'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,d):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,allow_nan=False)
def bound(p):return {'path':str(p),'sha256':sha(p)}
def verify(bs):
    for b in bs:
        if sha(b['path'])!=b['sha256']:raise ValueError('Changed binding '+b['path'])
def lines(p):
    op=gzip.open if str(p).endswith('.gz') else open
    with op(p,'rt',encoding='utf-8') as f:
        for x in f:
            if x.strip():yield json.loads(x)

def wait_headroom(psutil):
    class Perf(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in
            ('CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache',
             'KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[(k,ctypes.c_ulong) for k in ('HandleCount','ProcessCount','ThreadCount')]
    while True:
        state=Perf();state.cb=ctypes.sizeof(state)
        if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(state),state.cb):raise OSError('Cannot measure Windows commit headroom')
        commit=(state.CommitLimit-state.CommitTotal)*state.PageSize;physical=psutil.virtual_memory().available
        if physical>=384*1024**2 and commit>=1280*1024**2:return {'physical_available':physical,'commit_available':commit}
        print(json.dumps({'waiting_for_headroom':True,'physical_available':physical,'commit_available':commit}),flush=True);time.sleep(10)

def register(out,tests):
    assert read(tests)['passed']
    done=read(ALIGN/'completion.json');assert done['complete'];verify([done['protocol']]+done['grids'])
    files=[ALIGN/'protocol.json',ALIGN/'completion.json',ALIGN/'selected_management.json',CAL,Path(tests),
           ROOT/'v26_opening_decision_audit_v2/completion.json',ROOT/'v26_opening_decision_audit_v2/decisions.jsonl.gz',
           ROOT/'v42_decision_value_reconciliation_v2/completion.json',ROOT/'v43_funded_decision_calibration/review_v1/completion.json']
    cells=[c for c in read(ALIGN/'protocol.json')['cells'] if c['kind']=='opening']
    for c in cells:
        prepared=Path(c['prepared']);files.extend([prepared/'prepared.json',prepared/'opportunities.jsonl'])
        for s in ('baseline','combined'):
            folder=ALIGN/'labels'/f"{c['period']}_opening_{s}"
            files.extend(folder/n for n in ('grid_manifest.json','request.json','completion.json'))
            manifest=read(folder/'grid_manifest.json')
            if sha(prepared/'prepared.json')!=manifest['prepared_sha256']:raise ValueError('Prepared label mismatch')
            for n,h in manifest['files'].items():
                if n.endswith('.jsonl'):
                    if sha(folder/n)!=h:raise ValueError('Consumed label mismatch')
                    files.append(folder/n)
    folds={}
    for period in ('is','oos'):
        profile=ROOT/f'v25_pending_funding_screen/cases/control/{period}/input_profile.json';p=read(profile);files.append(profile)
        files.extend(Path(p[k]['path']) for k in ('manifest','policy'))
        folder=Path(p['policy']['path']).parent;policy=read(folder/'policies_manifest.json')
        files.append(folder/'chronology.json');assert sha(folder/'chronology.json')==policy['files']['chronology.json']
        for f in read(folder/'chronology.json')['folds']:
            if f.get('kind')=='opening' and f.get('scenario') in ('baseline','combined'):
                assert sha(f['model_path'])==f['model_sha256'];files.append(Path(f['model_path']));folds[f['valid_from'],f['scenario']]=f
    files=list(dict.fromkeys(files));out.mkdir(exist_ok=False)
    save(out/'protocol.json',{'kind':'paired_action_value_loss_contrast','registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'mechanism':'Current model selects size using absolute action value. Test learning the extra full-cost payoff of additional shares directly on paired same-issuer/same-clock quantity actions. Avoid algebraically identical differences of separately fitted linear models.',
        'recipes':['pair_mean','paired_state'],'adjacent_pairs':[[.25,.5],[.5,1.]],
        'design':'Keep the exact inherited frozen q*[1, StateMap(x)] potential basis. Fit ONE shared coefficient vector to high-minus-low basis and high-minus-low selected-management native net equity bps. Ridge120; pair_mean zero-shrunk prior100 component. No learner, feature, penalty or gate sweep.',
        'population':'Complete original ready+rank1-guard causal opening population, including preIS. Both quantities and both scenarios require valid mature labels. Invalid labels/failures and integer aliases remain in source and counts. Scenarios, adjacent contrasts and parents are dependent.',
        'weights':'One issuer-day weight across adjacent pairs; equal sessions, normalized to distinct issuer-days.',
        'chronology':'Original12 base windows/20session clock,60 mature sessions/100issuer-days,one-session embargo,20issuer-days/8sessions each pair; no reset/backfill.',
        'domain':'Prior-only min/max of four incremental physical-exposure features (notional/equity, last-bar/opening-volume quantity ratios, low-volume exit slices). Frozen StateMap handles all original economic features; domain is conditional model support, not exchangeability proof.',
        'evaluation':'Measure same-pair error versus pair_mean and original implied raw differences. Separately inspect actual retained prior100/prior125 parent high-quantity actions. Candidate is at most ONE adjacent size reduction if BOTH scenario marginal estimates are supported and negative and lower action passes original risk<.001/capacity. Never increase size/add entries/rerank/change exits.',
        'native_readiness':'No native dispatch from this producer. Before separate protocol require paired_state to improve IS paired error versus pair_mean in BOTH scenarios, at least20 distinct IS changed issuer-days/8sessions per parent, nonnegative sum of isolated IS low-minus-high action contrasts in BOTH scenarios, and complete winning-tail/concentration/uncertainty review. Quantity contrast sums are NOT portfolio returns. All OOS outcomes retained, no rescue tuning.',
        'estimate':'5-15 minutes model-only including frozen source checks;24 conditional fits plus mean component. No new labels or native portfolio cases.',
        'limits':'Reference-state paired labels do not prove the changed funded account; exact native implementation, parent bridges, new quantity/funding/source/account/MTM audits and four-stress qualification still required. April-July reused development.',
        'cells':cells,'folds':list(folds.values()),
        'bindings':[bound(p) for p in files],
        'programs':[bound(Path(__file__).resolve()),bound(HELPER),bound(REPO/'tests/test_kalcb_paired_quantity_value_v1.py')]})

def run(out):
    p=read(out/'protocol.json');verify(p['bindings']+p['programs'])
    if (out/'supervisor_start.json').exists():raise ValueError('Already run')
    import psutil
    proc=psutil.Process();save(out/'supervisor_start.json',{'pid':proc.pid,'create_time':proc.create_time(),'cmdline':proc.cmdline(),'at_utc':datetime.now(timezone.utc).isoformat()})
    profile=read(ROOT/'v25_pending_funding_screen/cases/control/oos/input_profile.json');manifest=read(profile['manifest']['path'])
    frozen=Path(profile['manifest']['path']).parent/'files';roots={Path(k).parts[0].removesuffix('.py') for k in manifest['files'] if k.endswith('.py')};origins={}
    class Gate(importlib.abc.MetaPathFinder):
        def find_spec(self,name,path=None,target=None):
            if name.split('.')[0] not in roots:return None
            spec=importlib.machinery.PathFinder.find_spec(name,path)
            if spec is None:raise ImportError('Missing frozen '+name)
            if spec.origin and spec.origin not in ('built-in','frozen'):
                f=Path(spec.origin).resolve();rel=f.relative_to(frozen.resolve()).as_posix();h=manifest['files'].get(rel)
                if isinstance(h,dict):h=h['sha256']
                if not h or sha(f)!=h:raise ImportError('Unbound frozen '+name)
                origins[name]={'path':str(f),'sha256':h}
            return spec
    sys.path.insert(0,str(frozen));sys.meta_path.insert(0,Gate())
    spec=importlib.util.spec_from_file_location('registered_paired_quantity_value',HELPER);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    import numpy as np,joblib
    from threadpoolctl import threadpool_limits
    from backtests.strategies.kalcb import continuation_value as cv
    calendar=read(CAL);sessions=calendar['sessions'];windows=calendar['windows'];assert len(sessions)==286 and sessions.index('2026-04-01')==217
    labels={s:{} for s in ('baseline','combined')};sources={};actions=[];invalid=defaultdict(int)
    for c in p['cells']:
        period=c['period'];prepared=Path(c['prepared']);meta=read(prepared/'prepared.json')
        if sha(prepared/'opportunities.jsonl')!=meta['opportunities_sha256']:raise ValueError('Prepared bytes')
        for scenario in labels:
            folder=ALIGN/'labels'/f'{period}_opening_{scenario}';g=read(folder/'grid_manifest.json')
            for n in sorted(g['files']):
                if not n.endswith('.jsonl'):continue
                for r in lines(folder/n):
                    key=r['opportunity_id']
                    if key in labels[scenario]:raise ValueError('Overlapping quantity targets')
                    labels[scenario][key]={k:r[k] for k in ('label_valid','label_available_at','net_value_equity_bps')}
                    if not r['label_valid']:invalid[scenario]+=1
        wanted=set(c['sessions'])
        for row in lines(prepared/'opportunities.jsonl'):
            if row['trade_date'] not in wanted or not row['action']['ready'] or not row['guard_at_rank1']:continue
            base=row['opportunity_id'];full=int(row['action']['quantity']);close=float(row['opening_bars'][-1]['close'])
            if base in sources:raise ValueError('Duplicate causal population')
            sources[base]=meta['opportunities_sha256']
            for q in (.25,.5,1.):
                qty=int(full*q);ratio=qty/full;features=dict(row['features'])
                for k in m.PHYSICAL:features[k]*=ratio
                features['action_quantity_scale']=q;key=base+f'/q{q:g}'
                if not math.isclose(features['action_notional_equity'],qty*close/100000000,abs_tol=1e-12):raise ValueError('Integer units')
                actions.append({'id':key,'base_id':base,'day':row['trade_date'],'available_at':row['feature_available_at'],
                    'source':meta['opportunities_sha256'],'q':q,'qty':qty,'notional':qty*close/100000000,
                    'features':features,'labels':{s:labels[s][key] for s in labels}})
    actions.sort(key=lambda r:(r['day'],r['base_id'],r['q']));index={r['id']:i for i,r in enumerate(actions)}
    if any(set(labels[s])!=set(index) for s in labels):raise ValueError('Incomplete action label universe')
    baseids=sorted(sources);folds={(f['valid_from'],f['scenario']):f for f in p['folds']};forecast=[];proofs=[];alias_counts={}
    lookup={r['action_id']:r for r in lines(ROOT/'v26_opening_decision_audit_v2/decisions.jsonl.gz')}
    for first,last in windows:
      for scenario in labels:
        headroom=wait_headroom(psutil)
        f=folds[first,scenario];model=joblib.load(f['model_path'])
        x=cv.matrix(actions,f['feature_columns']);z=model.features.transform(x)
        basis=m.potential_basis([r['q'] for r in actions],z);pairs=[];aliases=0
        for base in baseids:
            for low,high in m.PAIRS:
                i=index[base+f'/q{low:g}'];j=index[base+f'/q{high:g}']
                r=m.make_contrast(actions[i],actions[j],basis[i],basis[j],scenario)
                if r is None:aliases+=1;continue
                pairs.append(r)
        with threadpool_limits(limits=1):proof=m.fit(pairs,first,sessions)
        proof.update(scenario=scenario,through=last,frozen_map_model_path=f['model_path'],frozen_map_model_sha256=f['model_sha256'])
        proofs.append(proof);alias_counts[scenario]=aliases
        for r in pairs:
            if not first<=r['day']<=last:continue
            low=r['base_id']+f"/q{r['low']:g}";high=r['base_id']+f"/q{r['high']:g}"
            if low not in lookup or high not in lookup:raise ValueError('Missing scored V26 action')
            current=lookup[high]['raw'][scenario]-lookup[low]['raw'][scenario]
            item={k:r[k] for k in ('id','base_id','day','available_at','low','high','target_valid','mature_at','value','physical')}
            item.update(scenario=scenario,first=first,original_raw_difference=current,
                predictions={recipe:m.predict(r,proof,recipe) for recipe in ('pair_mean','paired_state')},
                high_parent_selections=list(lookup[high]['selected']),
                lower_physical_capacity_pass=lookup[low]['physical_risk_pass'] and lookup[low]['capacity_pass'])
            forecast.append(item)
        print(json.dumps({'through':last,'scenario':scenario,'training_sessions':proof['mature_sessions'],'training_issuer_days':proof['issuer_days'],'supported':proof['supported'],'headroom':headroom}),flush=True)
        del x,z,basis,pairs,model;gc.collect()
    save(out/'models.json',proofs)
    with gzip.open(out/'forecasts.jsonl.gz','xt',encoding='utf-8') as stream:
        for r in forecast:stream.write(json.dumps(r,allow_nan=False)+'\n')
    # Assessment is a deterministic fixed descriptive screen, not a policy.
    joined={(r['id'],r['scenario']):r for r in forecast};summary={};changes=[]
    for parent in ('control','control_prior125'):
      for recipe in ('pair_mean','paired_state'):
       for period in ('is','oos'):
        selected=[]
        for r in forecast:
            if r['scenario']!='baseline' or parent not in r['high_parent_selections']:continue
            if ('is' if r['day']<='2026-03-31' else 'oos')!=period:continue
            c=joined[r['id'],'combined'];fb=r['predictions'][recipe];fc=c['predictions'][recipe]
            if r['lower_physical_capacity_pass'] and c['lower_physical_capacity_pass'] and fb['supported'] and fc['supported'] and fb['value']<0 and fc['value']<0:
                item={'parent':parent,'recipe':recipe,'period':period,'id':r['id'],'base_id':r['base_id'],'day':r['day'],
                      'low':r['low'],'high':r['high'],'forecasts':{'baseline':fb['value'],'combined':fc['value']},
                      'delta_isolated_equity_bps':{s:-v['value'] if v['target_valid'] else None for s,v in [('baseline',r),('combined',c)]}}
                selected.append(item);changes.append(item)
        summary[parent+'/'+recipe+'/'+period]={'changed_issuer_days':len({r['base_id'] for r in selected}),'changed_sessions':len({r['day'] for r in selected}),
                'isolated_contrast_sum_not_portfolio_pnl':{s:sum(r['delta_isolated_equity_bps'][s] or 0 for r in selected) for s in labels},
                'invalid_current_targets':sum(any(v is None for v in r['delta_isolated_equity_bps'].values()) for r in selected)}
    errors={}
    for period in ('is','oos'):
      for scenario in labels:
        rs=[r for r in forecast if r['scenario']==scenario and ('is' if r['day']<='2026-03-31' else 'oos')==period and r['target_valid'] and all(x['supported'] for x in r['predictions'].values())]
        daily=defaultdict(list)
        for r in rs:daily[r['day']].append(r)
        errors[period+'/'+scenario]={recipe:float(np.mean([np.mean([(r['predictions'][recipe]['value']-r['value'])**2 for r in xs]) for xs in daily.values()])) for recipe in ('pair_mean','paired_state')}
    save(out/'assessment.json',{'complete':True,'changes':changes,'summary':summary,'session_balanced_pair_errors':errors,
        'invalid_source_actions_retained':dict(invalid),'integer_aliases_per_scenario':alias_counts,
        'scope':'Reference-state paired contrasts; not funded portfolio returns or attained return/DD gains. No policy/native case created.'})
    verify(p['bindings']+p['programs']);verify(list(origins.values()))
    save(out/'origins.json',{'passed':True,'origins':origins})
    save(out/'completion.json',{'complete':True,'study_complete':False,'native_cases':0,'policy_files':0,
        'outputs':[bound(out/n) for n in ('models.json','forecasts.jsonl.gz','assessment.json','origins.json')],
        'completed_at_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'complete':True,'summary':summary,'errors':errors}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','run']);p.add_argument('--root',type=Path,required=True);p.add_argument('--tests',type=Path);a=p.parse_args()
    register(a.root,a.tests) if a.command=='register' else run(a.root)

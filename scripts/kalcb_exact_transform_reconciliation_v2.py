"""Authenticate saved-model inference using the exact integer action features."""
from pathlib import Path
import argparse, gzip, hashlib, importlib.abc, importlib.machinery, json, sys
from collections import defaultdict

ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(x) for x in f]
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)

def main(out):
    out.mkdir(exist_ok=False)
    source=ROOT/'v42_decision_value_reconciliation_v2'
    done=read(source/'completion.json');assert done['complete']
    bindings=[]
    def bind(p,expected=None):
        actual=sha(p)
        if expected is not None and actual!=expected:raise ValueError('Source drift '+str(p))
        bindings.append({'path':str(p),'sha256':actual});return p
    for b in done['outputs']:bind(b['path'],b['sha256'])
    profile=read(ROOT/'v25_pending_funding_screen/cases/control/oos/input_profile.json')
    manifest=read(bind(profile['manifest']['path'],profile['manifest']['sha256']))
    frozen=Path(profile['manifest']['path']).parent/'files'
    origins={}
    application_roots={Path(k).parts[0].removesuffix('.py') for k in manifest['files'] if k.endswith('.py')}
    class FrozenOnly(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] not in application_roots:return None
            spec=importlib.machinery.PathFinder.find_spec(fullname,path)
            if spec is None:raise ImportError('Missing frozen module '+fullname)
            if spec.origin and spec.origin not in ('built-in','frozen'):
                p=Path(spec.origin).resolve();rel=p.relative_to(frozen.resolve()).as_posix()
                expected=manifest['files'].get(rel)
                if isinstance(expected,dict):expected=expected['sha256']
                if not expected or sha(p)!=expected:raise ImportError('Unbound frozen import '+fullname)
                origins[fullname]={'path':str(p),'sha256':expected}
            return spec
    sys.path.insert(0,str(frozen));sys.meta_path.insert(0,FrozenOnly())
    # Registration precedes importing the model's application modules.
    folds={}
    for period in ('is','oos'):
        p=read(ROOT/f'v25_pending_funding_screen/cases/control/{period}/input_profile.json')
        policy=read(bind(p['policy']['path'],p['policy']['sha256']));folder=Path(p['policy']['path']).parent
        chron=read(bind(folder/'chronology.json',policy['files']['chronology.json']))
        for f in chron['folds']:
            if f.get('kind')=='opening' and f.get('scenario') in ('baseline','combined'):
                bind(f['model_path'],f['model_sha256']);folds[(f['valid_from'],f['scenario'])]=f
    save(out/'protocol.json',{'scope':'Supplemental complete application import closure, derived from all manifest Python roots. Saved inference only; zero fits/policy/native variants. All17445 exact action vectors, both scenarios. V1 narrower47-module origin receipt preserved.',
         'program':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},'bindings':bindings,
         'runtime':'1-5 minutes','study_complete':False})
    import joblib,numpy as np
    from threadpoolctl import threadpool_limits
    from backtests.strategies.kalcb import continuation_value as cv
    feature={r['action_id']:r['features'] for r in lines(source/'exact_model_features.jsonl.gz')}
    records=lines(ROOT/'v26_opening_decision_audit_v2/decisions.jsonl.gz')
    groups=defaultdict(list)
    for r in records:groups[r['base_valid_from']].append(r)
    output=[];total=0;max_error=0.
    for first,items in sorted(groups.items()):
      for scenario in ('baseline','combined'):
        f=folds[first,scenario];model=joblib.load(f['model_path'])
        x=cv.matrix([{'features':feature[r['action_id']]} for r in items],f['feature_columns'])
        with threadpool_limits(limits=1):prediction=model.predict(x)
        error=np.abs(prediction-np.asarray([r['raw'][scenario] for r in items]))
        if error.max()>1e-8:raise ValueError('Saved raw inference mismatch '+str(error.max()))
        total+=len(items);max_error=max(max_error,float(error.max()))
        transformed=model.features.transform(x)
        filled=model.features.impute.transform(x);z=model.features.scale.transform(filled)
        for population in ('all','control','control_prior125'):
          for period in ('is','oos'):
            idx=[i for i,r in enumerate(items) if r['period']==period and (population=='all' or population in r['selected'])]
            if not idx:continue
            output.append({'first':first,'scenario':scenario,'population':population,'period':period,'rows':len(idx),
                'raw_columns':len(f['feature_columns']),'transformed_columns':transformed.shape[1],
                'missing_values':int(np.isnan(x[idx]).sum()),'rows_any_clipped_z':int((np.abs(z[idx])>5).any(axis=1).sum()),
                'clipped_features':{k:int((np.abs(z[idx,j])>5).sum()) for j,k in enumerate(f['feature_columns']) if (np.abs(z[idx,j])>5).any()}})
        print(json.dumps({'fold':first,'scenario':scenario,'rows':len(items),'maximum_raw_error':float(error.max())}),flush=True)
    for b in bindings:
        if sha(b['path'])!=b['sha256']:raise ValueError('Binding changed during inference')
    for b in origins.values():
        if sha(b['path'])!=b['sha256']:raise ValueError('Frozen import changed')
    save(out/'review.json',{'passed':True,'forecast_values_reconstructed':total,'maximum_raw_error':max_error,
         'origins':origins,'feature_use':output,'scope':'Exact parent transformation/inference; clipping counts are descriptive, not an outcome-derived policy gate.'})
    save(out/'completion.json',{'complete':True,'study_complete':False,'review_sha256':sha(out/'review.json'),'protocol_sha256':sha(out/'protocol.json')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)

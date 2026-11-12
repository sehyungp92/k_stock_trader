"""Executable C16 qualification. Research controls and invalid evidence differ."""
from copy import deepcopy
from math import isfinite
from statistics import NormalDist
import hashlib
import json
from pathlib import Path
import numpy as np

SCENARIOS=('baseline','participation','cost','combined')
CAPITALS=(95000000,100000000,105000000)


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_acceptance(receipt):
    if receipt.get('passed') is not True:
        raise ValueError('Failed or missing native acceptance')
    for name,binding in receipt['files'].items():
        if digest(binding['path'])!=binding['sha256']:
            raise ValueError('Stale accepted artifact '+name)
    if not receipt['files']:
        raise ValueError('Acceptance has no bound evidence')
    return True


def scalar(cells):
    expected={f'{p}/{s}' for p in ('is','oos') for s in SCENARIOS}
    if set(cells)!=expected:raise ValueError('Incomplete ensemble')
    for key,cell in cells.items():
        if cell.get('native_valid') is not True:
            raise ValueError('Invalid evidence cannot enter scoring: '+key)
        if not all(isfinite(cell[k]) for k in ('return_pp','mtm_dd_pp')) or cell['mtm_dd_pp']<0:
            raise ValueError('Nonfinite/invalid account metric')
    R=sum(c['return_pp'] for c in cells.values())/8
    D=sum(max(0,c['mtm_dd_pp']-(8 if k.endswith('/baseline') else 10)) for k,c in cells.items())/8
    return dict(R=R,D=D,S=R-.25*D)


def paired_statistics(daily_differences, *, family_size=8):
    """Paired circular date blocks; scenario means preserve shared episodes."""
    if family_size!=8:raise ValueError('Frozen family changed')
    vector={}
    for p in ('is','oos'):
        rows=np.asarray(daily_differences[p],dtype=float)
        if rows.shape != (4,156 if p=='is' else 69) or not np.isfinite(rows).all():
            raise ValueError('Wrong complete paired calendar/scenario shape')
        vector[p]=rows.mean(axis=0)
    value=.5*sum(v.sum() for v in vector.values()); summaries={}
    for length in (5,10,20):
        draws=np.zeros(4000)
        for p,values in vector.items():
            n=len(values);rng=np.random.default_rng(41017+length+n)
            starts=rng.integers(0,n,size=(4000,int(np.ceil(n/length))))
            indices=((starts[:,:,None]+np.arange(length))%n).reshape(4000,-1)[:,:n]
            draws+=.5*values[indices].sum(axis=1)
        se=float(np.std(draws,ddof=1));q=.05/(2*family_size)
        summaries[str(length)]=dict(gain_pp=float(value),paired_SE_pp=se,
            unadjusted_CI95_pp=np.quantile(draws,[.025,.975]).tolist(),
            simultaneous_Bonferroni_CI95_pp=np.quantile(draws,[q,1-q]).tolist(),
            planning_MDE_pp=(NormalDist().inv_cdf(1-q)+NormalDist().inv_cdf(.8))*se,
            block_equivalents=dict(is_156=156/length,oos_69=69/length),
            effective_sample_size=None,credible_population_power=False,
            limitation='Conditional reused-history planning; n/L is not independent-market ESS. Adaptive earlier search remains.')
    return summaries


def qualify(control,candidate,all_capital_cells,statistics,checks,*,diagnostic=False):
    score=scalar(candidate);base=scalar(control)
    expected={f'{p}/{s}/{c}' for p in ('is','oos') for s in SCENARIOS for c in CAPITALS}
    risk_complete=set(all_capital_cells)==expected
    failures=[]
    if not risk_complete:failures.append('missing_capital_qualification')
    for key,cell in all_capital_cells.items():
        if not cell.get('native_valid'):failures.append('invalid_account:'+key)
        limit=12 if '/baseline/' in key else 15
        marks=[cell.get('mtm_dd_pp'),cell.get('finer_dd_pp')]
        if any(v is not None and (not isfinite(v) or v>limit) for v in marks):
            failures.append('hard_MTM_limit:'+key)
    gain=score['R']-base['R']; losses={k:candidate[k]['return_pp']-control[k]['return_pp'] for k in control}
    primary=statistics['5']
    if primary['simultaneous_Bonferroni_CI95_pp'][0]<=0:failures.append('paired_simultaneous_lower_bound')
    if gain<max(1.,primary['planning_MDE_pp']):failures.append('incremental_materiality_resolution')
    if sum(v for k,v in losses.items() if k.startswith('is/'))/4 < -2:failures.append('IS_sacrifice')
    if min(losses.values()) < -1:failures.append('individual_scenario_sacrifice')
    for key in ('mechanism','independent_accounting','source','flat','ablation','neighbourhood','delay','cost','retained_components'):
        if checks.get(key) is not True:failures.append('qualification:'+key)
    if diagnostic:failures.append('diagnostic_only_not_selectable')
    return dict(promote=not failures,failures=failures,gain_pp=gain,score_gain=score['S']-base['S'],
        score=score,frequency_credit=0,
        frequency_credit_reason='No independently qualified positive-expectancy addition group supplied; raw counts descriptive')


def selectable_score(evaluation):
    """No failed qualification may be disguised as an ordinary low score."""
    if evaluation.get('native_valid') is not True:raise ValueError('Invalid evaluation')
    if evaluation.get('accepted_control') is True:return float(evaluation['score']['S'])
    if evaluation.get('qualification',{}).get('promote') is not True:
        raise ValueError('Unqualified candidate cannot be selected')
    return float(evaluation['score']['S'])

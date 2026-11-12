"""C16 R2: register exact predeclared capital/capacity tuples; no economic change."""
import copy
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import kalcb_round2_noise_v1 as base

OLD=base.OUT
OUT=base.ROOT/'noise_r2'


def freeze_existing(period,initial):
    old=base.read(OLD/'runtime'/f'{period}_{initial}'/'profile.json')
    mp=Path(old['manifest']['path']);payload=base.read(mp);signature=payload.pop('manifest_hash')
    assert base.stable_signature(payload)==signature
    src=mp.parent/'files';folder=OUT/'runtime'/f'{period}_{initial}';code=folder/'inputs/files'
    qold=base.read(src/payload['evidence']['qualification_path']);qualification=copy.deepcopy(qold)
    declaration=qualification['capacity']
    assert declaration['kind']=='declared_bar_participation_model' and declaration['phase_liquidity_verified'] is False
    new=[]
    for sc in base.SCENARIOS:
        a=base.read(base.SOURCE/'cases'/f'combined_{period}_{sc}'/'execution_audit.json')
        policy=a['audit_receipt']['qualification']['capacity']['policy']
        matches=[r for r in declaration['registered_execution_policies'] if r['initial_equity']==100000000 and r['buying_power_leverage']==2 and r['policy']==policy]
        assert len(matches)==1
        row=copy.deepcopy(matches[0]);row['initial_equity']=initial
        assert row not in declaration['registered_execution_policies']
        declaration['registered_execution_policies'].append(row);new.append(row)
    check=copy.deepcopy(qualification);check['capacity']['registered_execution_policies']=check['capacity']['registered_execution_policies'][:-4]
    assert check==qold
    assert not any(r['initial_equity']==97000000 for r in declaration['registered_execution_policies'])
    for relative,sha in payload['files'].items():
        source=src/relative;dest=code/relative
        assert base.digest(source)==sha,relative
        dest.parent.mkdir(parents=True,exist_ok=True);os.link(source,dest)
    qrel='research/c16/r2_qualification.json';base.write(code/qrel,qualification)
    payload['files'][qrel]=base.digest(code/qrel);payload['evidence']['qualification_path']=qrel
    payload['evidence']['c16_r2']=dict(registration=base.bound(OUT/'registration.json'),original_profile=base.bound(OLD/'runtime'/f'{period}_{initial}'/'profile.json'),added_exact_tuples=new)
    base.write(folder/'inputs/manifest.json',{**payload,'manifest_hash':base.stable_signature(payload)})
    item={**old,'folder':str(folder),'manifest':base.bound(folder/'inputs/manifest.json'),
          'policy':base.bound(code/'research/kalcb_executed_value/policies/policies_manifest.json')}
    assert item['policy']['sha256']==old['policy']['sha256']
    # Every Python input is retained byte-for-byte. Only a JSON audit declaration is added.
    assert all(payload['files'][r]==v for r,v in base.read(mp)['files'].items())
    base.write(folder/'profile.json',item)
    base.write(folder/'preflight.json',dict(passed=True,added_capacity_tuples=4,original_fields_preserved=True,
        original_all_files_preserved=True,undeclared_97m_rejected=True,source_generation_unchanged=True))
    return item


def main():
    OUT.mkdir(exist_ok=False)
    base.write(OUT/'registration.json',dict(program=base.bound(__file__),executor=base.bound(base.__file__),
        repair=base.bound(base.REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0041-repair-r2.md'),
        prior_registration=base.bound(OLD/'registration.json'),failure=base.bound(OLD/'failure.json'),max_total_executions=27))
    old=base.read(base.SOURCE/'registration.json')['profiles'];items={};receipts=[]
    for p in ('is','oos'):
        for capital in (95000000,105000000):items[p,capital]=freeze_existing(p,capital)
    base.OUT=OUT
    for p in ('is','oos'):
        for sc in base.SCENARIOS:
            if (p,sc)!=('is','baseline'):
                receipts.append(base.execute(old[p+'_cap2'],p,sc,f'exact_{p}_{sc}',100000000,base.SOURCE/'cases'/f'combined_{p}_{sc}'))
            for capital in (95000000,105000000):
                prior=OLD/'cases'/'capital_is_baseline_95000000' if (p,sc,capital)==('is','baseline',95000000) else None
                receipts.append(base.execute(items[p,capital],p,sc,f'capital_{p}_{sc}_{capital}',capital,prior))
    base.write(OUT/'completion.json',dict(complete=True,cases=receipts,new_executions=len(receipts),
        reused_100m_bridges=[base.bound(OLD/f'bridge_{p}_verdict.json') for p in ('is','oos')],
        preserved_unverified_capital_account=str(OLD/'cases/capital_is_baseline_95000000'),
        total_executions_including_original_pilot_and_preserved_failure=len(receipts)+4))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():base.write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise

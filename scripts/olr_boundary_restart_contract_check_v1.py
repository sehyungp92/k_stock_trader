"""Exercise the exact frozen replay checkpoint branch, including bad inputs.

No replay/source changes. Bound input objects are copied in memory; SQLite is
read-only. Hashes for synthetic bad checkpoints are consistently recomputed so
the semantic validation, rather than merely a stale digest, must reject them.
"""
from __future__ import annotations
import copy
from dataclasses import asdict,replace
from datetime import date,datetime,time,timedelta
import hashlib,json
from pathlib import Path
import sqlite3,sys,textwrap
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010/boundary_restart_v1')
OUT=REPO/'data/backtests/analysis/olr_round1_optimization_20261010'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    study=read(ROOT/'runtime/c480e063db1f3b716/oos/baseline/study.json')
    files=Path(study['code']);sys.path.insert(0,str(files))
    from strategy_olr.config import OLRConfig
    from strategy_common.clock import KST
    from backtests.auto.shared.cache_keys import stable_signature
    source=files/'scripts/portfolio_synergy_study.py';value=source.read_text(encoding='utf-8')
    branch=textwrap.dedent(value[value.index("    if study.get('capacity_checkpoint'):"):value.index('    capacity_history_receipt = None')])
    code=compile(branch,str(source)+'::checkpoint-branch','exec')
    payload={k:read(files/study[k]) for k in ('capacity_checkpoint','capacity_checkpoint_completion','capacity_checkpoint_result','olr_request')}
    request=payload['olr_request'];recipe=study['candidates']['candidate']
    cfg=replace(OLRConfig.from_mapping(request['config'],request['mutations']),slippage_bps=5,tax_bps_on_sell=20)
    cfg=replace(cfg,**recipe.get('olr_capacity',{}))
    assert stable_signature(asdict(cfg))==payload['capacity_checkpoint']['effective_config_hash']
    import yaml
    u=Path(study['configuration']['universe']);u=next(p for p in (files/u,files/'config'/u) if p.exists())
    universe={str(s).zfill(6) for s in yaml.safe_load(u.read_text())['symbols']}
    outcomes={}
    with sqlite3.connect('file:'+(files/study['database']).as_posix()+'?mode=ro',uri=True) as connection:
        for case in ('valid','wrong_scenario','wrong_configuration','wrong_compounding_mode','wrong_initial_equity','wrong_source_generation','unqualified_parent','open_parent_position',
                     'future_observation','fabricated_volume','undeclared_symbol','duplicate_observation','wrong_phase'):
            data=copy.deepcopy(payload);checkpoint=data['capacity_checkpoint'];hist=checkpoint['capacity_history_v1']
            symbol=next(k for k,v in hist.items() if v['auction']);sample=hist[symbol]['auction'][0]
            scenario='baseline'
            if case=='wrong_scenario':scenario='combined'
            elif case=='wrong_configuration':checkpoint['effective_config_hash']='invalid'
            elif case=='wrong_compounding_mode':checkpoint['capped']=True
            elif case=='wrong_initial_equity':checkpoint['initial_equity']+=1
            elif case=='wrong_source_generation':checkpoint['source_generation_hash']='invalid'
            elif case=='unqualified_parent':data['capacity_checkpoint_result']['audit_status']='unverified'
            elif case=='open_parent_position':data['capacity_checkpoint_result']['terminal_inventory']={'OLR':1}
            elif case=='future_observation':hist[symbol]['auction']=[['2026-04-01T15:30:00+09:00',sample[1]]]
            elif case=='fabricated_volume':sample[1]+=1
            elif case=='undeclared_symbol':hist['000000']=hist.pop(symbol)
            elif case=='duplicate_observation':hist[symbol]['auction']=[sample,sample]
            elif case=='wrong_phase':hist[symbol]['continuous']=[sample]
            synthetic={str(files/study[k]):stable_signature(v) for k,v in data.items()}
            completion=data['capacity_checkpoint_completion']
            completion['artifacts']['capacity_terminal_state.json']=synthetic[str(files/study['capacity_checkpoint'])]
            completion['artifacts']['result.json']=synthetic[str(files/study['capacity_checkpoint_result'])]
            adapter=SimpleNamespace(strategy_id='OLR',state=SimpleNamespace(meta={'preserved_sentinel':'unchanged'}))
            written={}
            scope=dict(study=study,bound=lambda k:data[k],generation=SimpleNamespace(root=files),
                digest=lambda p:synthetic.get(str(p),'synthetic-receipt'),stable_signature=stable_signature,asdict=asdict,
                ocfg=cfg,args=SimpleNamespace(scenario=scenario,no_upward_compounding=False,output=Path('unused')),
                ks={date(2026,4,1):None},datetime=datetime,time=time,timedelta=timedelta,KST=KST,
                capacity_observation_symbols=universe,connection=connection,adapters=[adapter],json=json,
                write_json=lambda p,v:written.update({p.name:v}))
            try:exec(code,scope)
            except ValueError as exc:
                assert case!='valid',(case,str(exc));outcomes[case]={'rejected':True,'reason':str(exc)}
            else:
                assert case=='valid',case
                receipt=written['capacity_restart_receipt.json']
                assert receipt['verified_observations']==sum(len(a) for h in hist.values() for a in h.values())
                assert adapter.state.meta['capacity_history_v1']==hist
                assert adapter.state.meta['preserved_sentinel']=='unchanged'
                outcomes[case]={'accepted':True,'verified_observations':receipt['verified_observations']}
    result={'status':'passed','tests':outcomes,'replay_source_sha256':sha(source),'checker_sha256':sha(__file__),
        'method':'Exact frozen native checkpoint branch; real retained observations; bad contracts altered in memory with consistent synthetic hashes.'}
    (OUT/'literal_restart_contract_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

"""Reauthenticate accepted cap/VETO controls; do not execute or rewrite them."""
from pathlib import Path
import sys,json,re
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.round3_native_io_v1 import ROOT,SOURCE,read,write,bound
from backtests.strategies.kalcb.round3_contract_v1 import verify_acceptance
from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_case
from backtests.strategies.kalcb.round3_analysis_v1 import account
from backtests.strategies.kalcb.round3_integrity_v1 import rows

OUT=ROOT/'reused_controls_v1';OUT.mkdir(exist_ok=False)
protocol=read(SOURCE/'protocol.json');reviews=read(SOURCE/'predecessor_reviews.json');arms=read(ROOT/'native_v5/arms.json');profiles=read(ROOT/'native_v5/preparation.json')['profiles']
registry={}
for old,name in [('veto','remove_cap'),('cap','remove_veto')]:
    registry[name]={}
    for key,directory in protocol['controls'][old].items():
        folder=Path(directory);p,s=key.split('/');prior=reviews[old+'/'+key];profile=prior['profile'];request=read(folder/'request.json')
        for binding in prior['funding']['bindings']:
            assert bound(binding['path'])['sha256']==binding['sha256'],'Historical accepted evidence changed'
        assert prior['funding']['verified'] is True
        actual=read(request['plan']['config']);expected=arms[name]['config'];manifest_path=Path(request['arguments']['audit_manifest'])
        if 'risk_max_gross_equity_pct' not in actual:
            config_source=(manifest_path.parent/'files/strategy_kalcb/config.py').read_text()
            assert re.search(r'risk_max_gross_equity_pct:\s*float\s*=\s*0(?:\.0)?\b',config_source)
            actual['risk_max_gross_equity_pct']=0
        assert actual==expected,'Prior effective settings differ'
        policy=Path(request['arguments']['policies'])/'policies_manifest.json'
        assert bound(policy)['sha256']==profiles[p][name+'/100000000']['policy']['sha256']
        assert request['plan']['input_sha256'][request['plan']['database']]==read(profiles[p][name+'/100000000']['plan']['path'])['input_sha256'][request['plan']['database']]
        if old=='veto':
            assert request['arguments']['mode']=='veto'
        else:assert 'mode' not in request['arguments'] # Accepted cap-only worker predates this controller.
        checked=validate_case(folder,profile,s)
        files={n:bound(folder/n) for n in ('request.json','completion.json','result.json','execution_audit.json','fills.json','trades.json','daily.json','equity.json','decisions.jsonl.gz','orders.jsonl.gz')}
        files.update(predecessor_reviews=bound(SOURCE/'predecessor_reviews.json'),original_protocol=bound(SOURCE/'protocol.json'),configuration=bound(request['plan']['config']),policy=bound(policy),source_manifest=bound(manifest_path))
        receipt=dict(passed=True,eligible_for_scoring=True,diagnostic_only=True,folder=str(folder),profile=profile,
            item=dict(arm=name,period=p,initial=100000000,source_generation_hash=request['source_generation_hash']),
            process=dict(scenario=s,reused=True,seconds=0),files=files,validation=checked['validation'],
            prior_funding=prior['funding'],equivalence='Exact effective config/data/policy; accepted C12 native cap/VETO bridges plus C17 eight-cell default identity. Diagnostic reuse only; original source/version preserved.')
        verify_acceptance(receipt);account(receipt)
        dest=OUT/(name+'_'+p+'_'+s+'.json');write(dest,receipt);registry[name][key]=bound(dest)
write(OUT/'completion.json',dict(passed=True,controls=registry,program=bound(__file__),new_financial_runs=0,
    effective_configuration_equivalence=True,original_generations_preserved=True))
print('Reauthenticated sixteen accepted component-control accounts')

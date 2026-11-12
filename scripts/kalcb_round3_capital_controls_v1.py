"""Authenticate the sixteen frozen accepted capital sensitivity controls."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.round3_native_io_v1 import ROOT,read,write,bound,profile_for
from backtests.strategies.kalcb.round3_contract_v1 import verify_acceptance
from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_case
from backtests.strategies.kalcb.round3_analysis_v1 import account
OUT=ROOT/'capital_controls_v1';OUT.mkdir(exist_ok=False);freeze=read(ROOT/'freeze/manifest.json');registry={}
for a in freeze['accounts']:
    capital=a['initial_equity']
    if capital==100000000:continue
    folder=Path(a['files']['result.json']['original']['path']).parent;s=a['scenario'];p=a['period'];item=a['profile']
    for x in a['files'].values():
        for b in x.values():assert bound(b['path'])['sha256']==b['sha256']
    for key in ('manifest','plan','policy'):assert bound(item[key]['path'])['sha256']==item[key]['sha256']
    profile=profile_for(item,folder,s);checked=validate_case(folder,profile,s)
    receipt=dict(passed=True,eligible_for_scoring=True,folder=str(folder),item=item,profile=profile,
        process=dict(scenario=s,reused=True,seconds=0),validation=checked['validation'],
        files={n:x['original'] for n,x in a['files'].items()}|{n:bound(folder/n) for n in ('request.json','completion.json')},
        original_C16_generation_preserved=True)
    verify_acceptance(receipt);account(receipt)
    path=OUT/f'{p}_{s}_{capital}.json';write(path,receipt)
    registry.setdefault(str(capital),{})[p+'/'+s]=bound(path)
assert all(len(v)==8 for v in registry.values()) and len(registry)==2
write(OUT/'completion.json',dict(passed=True,capitals=registry,program=bound(__file__),freeze=bound(ROOT/'freeze/manifest.json'),new_financial_runs=0))
print('Sixteen capital controls reauthenticated')

"""Preserve v3; bind the four actual scenarios, not every historical capacity tuple."""
from pathlib import Path
source=Path(__file__).with_name('kalcb_round3_runtime_v3.py').read_text(encoding='utf-8')
source=source.replace("OUT=ROOT/'native_v3'","OUT=ROOT/'native_v4'")
old="""    originals=[deepcopy(r) for r in declaration if r['initial_equity']==100000000 and r['buying_power_leverage']==2]
    if len(originals)!=4:raise ValueError('Unexpected reference capacity tuples')"""
new="""    originals=[]
    archive=Path('E:/CodexResearch/k_stock_trader/kalcb_opportunity_lifecycle_20261016/funded_interaction_v1')
    for scenario in SCENARIOS:
        policy=read(archive/'cases'/f'combined_{period}_{scenario}'/'execution_audit.json')['audit_receipt']['qualification']['capacity']['policy']
        matches=[r for r in declaration if r['initial_equity']==100000000 and r['buying_power_leverage']==2 and r['policy']==policy]
        if len(matches)!=1:raise ValueError('Ambiguous exact registered capacity tuple')
        originals.append(deepcopy(matches[0]))
    if len({stable_signature(x) for x in originals})!=4:raise ValueError('Scenario alias')"""
assert source.count(old)==1
exec(compile(source.replace(old,new),__file__,'exec'),{'__name__':'__main__','__file__':__file__})

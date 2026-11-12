"""Declare paired management and risk-control configurations before replay."""
import argparse,json,sys
from copy import deepcopy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev
from scripts.kalcb_executed_value_study import load_plan
from strategy_kalcb.config import KALCBConfig

CHANGES={
    'persistent_failure':{'exit_failed_followthrough_persistent':True},
    'acceptance_extension':{'exit_timed_acceptance_extension_bars':3},
    'hard_stop':{'exit_hard_stop_enabled':True},
    'constant65':{'entry_quantity_multiplier':.65},
    'concentration':{'risk_max_single_equity_pct':.35,'risk_max_gross_equity_pct':1.2},
    'governed':{'risk_max_single_equity_pct':.35,'risk_max_gross_equity_pct':1.2,
        'risk_drawdown_start_pct':.03,'risk_drawdown_full_pct':.07,'risk_drawdown_min_mult':.5},
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();parent=load_plan(a.plan);original=json.loads(Path(parent['config']).read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    for name,changes in CHANGES.items():
        config={**original,**changes};KALCBConfig.from_mapping(config)
        path=(a.output/(name+'.config.json')).resolve();ev.write_json(path,config)
        plan=deepcopy(parent);plan['input_sha256'].pop(plan['config']);plan['config']=str(path)
        plan['input_sha256'][str(path)]=ev.digest(path);plan['campaign_experiment']=name
        ev.write_json(a.output/(name+'.plan.json'),plan)
    ev.write_json(a.output/'protocol.json',{'parent_plan_sha256':ev.digest(a.plan),'changes':CHANGES,
        'management':'Paired current cv_interaction admissions/quantities; no promotion without new policy labels if management changes',
        'allocation':'Select after feature-only result; compare constant reduction, concentration caps and causal drawdown ramp',
        'user_drawdown_targets_pct':{'baseline':8,'combined':10},
        'scope':'Previously used chronological development data; no claim of fresh OOS validation'})

if __name__=='__main__':main()

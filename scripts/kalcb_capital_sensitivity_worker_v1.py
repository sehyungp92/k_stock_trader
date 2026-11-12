"""Frozen matched-capital diagnostic; no policy fitting or continuous-state claim."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('code','plan','policies','audit-manifest','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();code=args.code.resolve();sys.path.insert(0,str(code))
    if Path(__file__).resolve()!=code/'scripts/kalcb_capital_sensitivity_worker_v1.py':
        raise ValueError('Execute the frozen worker')
    from scripts.kalcb_executed_value_replay import run
    from scripts.kalcb_executed_value_study import load_plan,config_for
    from backtests.strategies.kalcb import executed_value as ev
    from backtests.strategies.kalcb.fixed_allocation_reference_v1 import FixedAllocationReference,verify_identity_bridge
    from backtests.auto.shared.cache_keys import source_generation_hash
    plan=load_plan(args.plan);contract=plan['capital_sensitivity'];initial=float(contract['initial_equity'])
    if initial<=0 or contract['mode'] not in ('compound','fixed') or contract['reference']!=100_000_000:
        raise ValueError('Invalid capital contract')
    cfg=config_for(plan,contract['scenario'])
    for key in ('risk_dynamic_notional_enabled','risk_max_single_equity_pct','risk_max_gross_equity_pct',
                'risk_daily_vol_target','risk_drawdown_full_pct','frontier_shadow_enabled','frontier_rotation_enabled'):
        if getattr(cfg,key):raise ValueError('Unsupported persistent/capital dependency: '+key)
    context=json.loads(Path(plan['audit_configuration']).read_text(encoding='utf-8'))
    if context['initial_equity']!=initial:raise ValueError('Audit/ledger capital mismatch')
    if float(ev.REFERENCE_EQUITY)!=100_000_000:raise ValueError('Unexpected original equity constant')
    source=source_generation_hash();args.output.mkdir(parents=True,exist_ok=False)
    arguments={k:str(getattr(args,k)) for k in ('plan','policies','output','audit_manifest')}
    arguments.update(policy='interaction_cost_margin',scenario=contract['scenario'])
    ev.write_json(args.output/'request.json',dict(arguments=arguments,plan=plan,
                  source_generation_hash=source,capital_sensitivity=contract))
    # The frozen run uses this constant only for broker/adapter initialization and
    # account metric denominators. No label generation or model fitting is called.
    ev.REFERENCE_EQUITY=initial
    allocation=FixedAllocationReference(contract['reference'],enabled=contract['mode']=='fixed')
    allocation.install()
    try:run(plan,args.policies,'interaction_cost_margin',contract['scenario'],args.output,args.audit_manifest)
    finally:allocation.restore();ev.REFERENCE_EQUITY=100_000_000.
    result=json.loads((args.output/'result.json').read_text())
    if result['audit_status']!='passed':raise ValueError('Native source/account audit failed')
    ev.write_json(args.output/'allocation_reference.json',allocation.evidence())
    if contract.get('bridge'):
        ev.write_json(args.output/'identity_bridge.json',verify_identity_bridge(args.output,contract['bridge']))
    if source_generation_hash()!=source:raise ValueError('Executing source changed')
    load_plan(args.plan)
    ev.write_json(args.output/'completion.json',dict(complete=True,source_generation_hash=source,
        initial_equity=initial,mode=contract['mode'],full_continuous_state_claim=False,
        independent_account_and_funding_review_required=True))
    print(json.dumps(dict(complete=True,case=contract['id'],output=str(args.output))),flush=True)


if __name__=='__main__':main()

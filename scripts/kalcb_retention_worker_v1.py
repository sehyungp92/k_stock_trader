"""One frozen native replay with observational exit-backout evidence."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('code','plan','policies','audit-manifest','observer','backout','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--scenario',required=True)
    parser.add_argument('--bridge',type=Path)
    args = parser.parse_args()
    sys.path.insert(0,str(args.code.resolve()))
    from scripts.kalcb_executed_value_replay import run
    from scripts.kalcb_executed_value_study import load_plan
    from backtests.auto.shared.cache_keys import source_generation_hash
    from backtests.strategies.kalcb import executed_value as ev
    module_spec = importlib.util.spec_from_file_location('retention_observer_frozen',args.observer)
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    plan = load_plan(args.plan); source = source_generation_hash()
    args.output.mkdir(parents=True,exist_ok=False)
    arguments = {'plan':str(args.plan),'policies':str(args.policies),'policy':'interaction_cost_margin',
                 'scenario':args.scenario,'audit_manifest':str(args.audit_manifest),'output':str(args.output)}
    ev.write_json(args.output/'request.json',{'arguments':arguments,'plan':plan,'source_generation_hash':source,
                  'observational_programs':{str(p):ev.digest(p) for p in (Path(__file__),args.observer,args.backout)}})
    observer = module.RetentionObserver(read(args.backout)); observer.install()
    try:
        run(plan,args.policies,'interaction_cost_margin',args.scenario,args.output,args.audit_manifest)
    finally:
        observer.restore()
    result = read(args.output/'result.json')
    support = observer.summarize(read(args.output/'fills.json'))
    ev.write_json(args.output/'intervention_support.json',{'complete':True,'summary':support,'probes':observer.probes})
    # Scored outcomes include the new management's full native cash costs;
    # forecasts intentionally remain fixed to isolate a management ablation.
    labels = []
    for trade in read(args.output/'trades.json'):
        labels.append({'symbol':trade['symbol'],'entry_decision_time':trade['entry_decision_time'],
            'entry_fill_time':trade['entry_fill_time'],'label_available_at':trade['exit_fill_time'],
            'qty':trade['qty'],'net_pnl_krw':trade['net_pnl'],'gross_pnl_krw':trade['gross_pnl'],
            'recorded_costs_krw':trade['commission'],'exit_reason':trade['exit_reason']})
    ev.write_json(args.output/'native_management_outcomes.json',{'labels':labels,
        'scope':'New fully funded native trade outcomes, dependent on account state. Not isolated action targets, not fed into forecasts or calibration. A future refit requires freshly replayed full causal action targets and label maturity checks.'})
    if args.bridge:
        for name in ('daily.json','equity.json'):
            if read(args.output/name) != read(args.bridge/name):
                raise ValueError('Observer account bridge drift '+name)
        def physical_fill(row):
            return {**{k:row[k] for k in ('order_id','strategy_id','symbol','side','qty','price','timestamp','reason')},
                    'execution':{k:row['metadata'].get(k) for k in ('execution_phase','base_price','fee','decision_available_at','submitted_at','execution_bar')}}
        if list(map(physical_fill,read(args.output/'fills.json'))) != list(map(physical_fill,read(args.bridge/'fills.json'))):
            raise ValueError('Observer execution bridge drift')
        if support['candidate_signal_differences'] or support['direct_changed_exit_orders']:
            raise ValueError('Identity backout changed native signal')
        ev.write_json(args.output/'observer_bridge.json',{'passed':True,'reference':str(args.bridge),
            'scope':'Every timestamped equity mark, daily account balance and physical fill reproduced.'})
    if source_generation_hash() != source:raise ValueError('Source changed during native replay')
    load_plan(args.plan)
    ev.write_json(args.output/'completion.json',{'complete':True,'source_generation_hash':source})
    print(json.dumps({'support':{k:v for k,v in support.items() if k!='execution_checks'},
                      'audited':result['audit_status']}),flush=True)


if __name__ == '__main__':main()

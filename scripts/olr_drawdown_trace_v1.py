"""Observe a frozen native replay; require exact saved economic-path parity.

No strategy parameters, fills, source files or market inputs are changed. The
in-memory observers export evidence already present during the native replay.
"""
from __future__ import annotations
import argparse,gzip,hashlib,importlib.util,json,sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    assert not a.output.exists(),'Preserve earlier traces; use a new output directory.'
    reference=a.reference.resolve();req=read(reference/'request.json');study=Path(req['arguments']['study']);frozen=Path(read(study)['code'])
    comp=read(reference/'completion.json')
    for name,h in comp['artifacts'].items():assert sha(reference/name)==h,name
    assert read(reference/'execution_audit.json')['audit_pass']
    sys.path.insert(0,str(frozen))
    spec=importlib.util.spec_from_file_location('observed_native_olr_replay',frozen/'scripts/portfolio_synergy_study.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from strategy_olr.core import logic
    from backtests.engine import integrity
    original_logic=logic._maybe_submit_managed_exit;original_audit=integrity.audit_broker
    stream=None;observed=0
    def observe_exit(state,bar,config):
        nonlocal stream,observed
        ss=state.symbol_state(bar.symbol);position=ss.position
        if position is not None and position.qty_open>0:
            if stream is None:stream=gzip.open(a.output/'managed_state.jsonl.gz','wt',encoding='utf-8')
            risk=logic._position_risk(position);held=tuple(v for v in ss.session_bars if v.timestamp>position.entry_time)
            plan=logic._exit_plan(config);eligible=bool(held and bar.timestamp>position.entry_time and risk>0 and plan.mode=='managed')
            hypothetical=logic._completed_bar_exit_reason(held,len(held)-1,position.entry_price,risk,position.max_favorable_price,plan,int(position.metadata.get('vwap_fail_streak',0))) if eligible else {}
            row={'timestamp':bar.timestamp.isoformat(),'symbol':str(bar.symbol),'entry_order_id':position.entry_order_id,
                'entry_time':position.entry_time.isoformat(),'entry_price':position.entry_price,'qty':position.qty_open,
                'bar':bar.to_json_dict(),'risk':risk,'mfe_price':position.max_favorable_price,'mae_price':position.max_adverse_price,
                'close_r':(float(bar.close)-position.entry_price)/risk if risk else None,'mfe_r':(position.max_favorable_price-position.entry_price)/risk if risk else None,
                'held_session_bars':len(held),'pending_exit_order_id':ss.pending_exit_order_id,
                'pending_role':state.order_roles.get(ss.pending_exit_order_id,{}),'pending_metadata':ss.pending_exit_metadata,
                'eligible_without_pending_gate':eligible,'reason_without_pending_gate':hypothetical.get('reason','')}
            stream.write(json.dumps(row,default=str)+'\n');observed+=1
        return original_logic(state,bar,config)
    def observe_audit(broker,*args,**kwargs):
        nonlocal stream
        if stream is not None:stream.close();stream=None
        with gzip.open(a.output/'native_journal.jsonl.gz','wt',encoding='utf-8') as f:
            for row in broker.journal:f.write(json.dumps(row,default=str)+'\n')
        write(a.output/'all_fills.json',[asdict(f) for f in broker.fills])
        write(a.output/'uncollapsed_exit_legs.json',[t.to_json_dict() for t in broker.trades])
        write(a.output/'observer_receipt.json',{'method':'Read-only wrappers around managed-exit evaluation and final broker audit; no action or execution modifications.',
            'reference':str(reference),'reference_completion_sha256':sha(reference/'completion.json'),
            'observer_sha256':sha(__file__),'frozen_source':str(frozen),'source_generation_hash':comp['source_generation_hash'],'position_bar_observations':observed})
        return original_audit(broker,*args,**kwargs)
    logic._maybe_submit_managed_exit=observe_exit;integrity.audit_broker=observe_audit
    args=dict(req['arguments']);args.update(study=study,output=a.output.resolve(),full_audit=True)
    module.replay(SimpleNamespace(**args))
    checks={}
    for name in ('trades.json','daily.json','equity.json','capacity_terminal_state.json'):
        checks[name]=read(reference/name)==read(a.output/name);assert checks[name],name
    assert read(a.output/'execution_audit.json')['audit_pass']
    write(a.output.parent/(a.output.name+'_exact_parity.json'),{'status':'passed','economic_parity':checks,'native_audit_pass':True,
        'reference':str(reference),'trace':str(a.output),'trace_completion_sha256':sha(a.output/'completion.json')})
    print({'observational_replay':'passed','reference':str(reference),'output':str(a.output),'observed_bars':observed},flush=True)


if __name__=='__main__':main()

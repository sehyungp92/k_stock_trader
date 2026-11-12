"""Standard KALCB diagnostic engine with C17 evidence/qualification context."""
import gc
import json
from pathlib import Path
import shutil
from .round3_native_io_v1 import bundle,read,bound,write
from .round3_contract_v1 import verify_acceptance
from .qualified_artifacts import diagnostic_result,close


def report(receipt,scenario,destination,config,mutations,*,round_num=3,round_name='C16 frozen round3',round_context=None):
    from .full_diagnostics import (analyze_kalcb_result,render_kalcb_diagnostics_report,
        _atomic_write_text,_atomic_write_json,_atomic_write_jsonl,_artifact_paths,_index_payload)
    verify_acceptance(receipt);data=bundle(receipt,scenario);result=diagnostic_result(data,scenario)
    result.metrics['invalid_order_rejection_count']=data['cases'][scenario]['audit']['invalid_order_rejection_count']
    cfg={**config,'date_range':data['profile']['date_range'],'scenario':scenario,'period':receipt['item']['period']}
    analysis=analyze_kalcb_result(result,config=cfg,mutations=mutations)
    native=data['cases'][scenario]
    for key,expected in [('net_return_pct',native['result']['return_pct']/100),
        ('max_drawdown_pct',native['result']['drawdown_pct']/100),('total_trades',native['result']['trades']),
        ('profit_factor',native['validation']['net_profit_factor'])]:
        close(analysis['final_metrics'][key],expected,'C17 full report drift '+key)
    context=dict(round_context or {},native_profile=data['profile'],acceptance=receipt,
        metric_units='Canonical return/DD fractions; C16 score and source result fields use percentage points; KRW currency.',
        development='All225 sessions adaptively consumed; IS156 and reused OOS69 are separate flat starts.',
        holdout='Unavailable: access audit found no genuinely untouched retained holdout.',
        oracle='Global funded S/E/SE, all-rejected labels and matched-entry placebo remain uncomputed; this report does not supply their estimates.',
        production_enabled=False,financing='Native model includes no financing/borrow charge or separate collateral-settlement ledger.')
    analysis.update(round=round_num,round_name=round_name,promotion_status=cfg.get('promotion_status','preflight'))
    analysis['gate_result']=dict(status=cfg.get('promotion_status','preflight'),native_integrity_pass=True,
        production_ready=False,qualification=(round_context or {}).get('qualification'))
    analysis['live_parity_audit'].update(status='offline_native_integrity_only',
        paper_promotion_blockers=['No independent untouched validation.','Live controller, queue/latency and process restart remain separately unqualified.'])
    analysis['diagnostics_summary']['round3_context']=context
    analysis['diagnostics_summary']['live_parity_audit']=analysis['live_parity_audit']
    text=render_kalcb_diagnostics_report(analysis).replace('Full-universe alpha','Non-selected outcome evidence').replace('Intraday Alpha Curve','Intraday realized PnL attribution')
    text+='\n\nC17 FROZEN CONTRACT / QUALIFICATION / COVERAGE\n'+json.dumps(context,indent=2,default=str)+'\n'
    out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    _atomic_write_text(out/'round_final_diagnostics.txt',text)
    _atomic_write_text(out/'round_final_full_diagnostics.txt',text)
    for name,value in [('diagnostics_summary.json',analysis['diagnostics_summary']),
        ('candidate_frontier.json',analysis['candidate_frontier']),('live_parity_audit.json',analysis['live_parity_audit']),
        ('round_final_full_diagnostics.json',{k:v for k,v in analysis.items() if k not in ('trades','decisions')}),
        ('qualified_replay_diagnostics.json',context),('selected_research_profile.json',data['profile'])]:
        _atomic_write_json(value,out/name)
    _atomic_write_jsonl(out/'trade_events.jsonl',analysis['trade_events'])
    _atomic_write_jsonl(out/'decision_events.jsonl',[d.to_json_dict() for d in result.decisions])
    shutil.copyfile(data['profile']['configuration'],out/'strategy_config.json')
    _atomic_write_json(_index_payload(analysis,_artifact_paths(out),{'native_input_manifest':data['manifest']}),out/'full_diagnostics_index.json')
    verify_acceptance(receipt)
    write(out/'diagnostic_integrity.json',dict(passed=True,source_and_account_bytes_unchanged=True,
        report=bound(out/'round_final_diagnostics.txt'),full_analysis=bound(out/'round_final_full_diagnostics.json')))
    summary=analysis['diagnostics_summary'];del data,result,analysis;gc.collect();return summary

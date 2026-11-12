"""Authenticate completed V25 full reports before the user-facing judgment."""
import argparse
from pathlib import Path
from datetime import datetime,timezone
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify
from backtests.strategies.kalcb.pending_funding_assessment_v1 import completion_gate,require,exact_process,close
from backtests.strategies.kalcb.pending_funding_full_diagnostics_v1 import authenticated_protocol,STATUS


def verify_reports(root,output):
    rows=completion_gate(root);reports=root/'full_reports_v1';p=authenticated_protocol(reports)
    require(not exact_process(read(reports/'supervisor_start.json')),'Report supervisor still running')
    require(not exact_process(read(root/'analysis_launch_v1/supervisor_start.json')),'Completion launcher still running')
    done=read(reports/'completion.json');require(done['complete'] and len(done['reports'])==16,'Incomplete reports')
    a=read(root/'assessment_v1/assessment.json');verify(a['bindings'])
    checks=[];artifacts=[];seen=set()
    for row in done['reports']:
        name,period,scenario=(row[k] for k in ('name','period','scenario'));key='/'.join((name,period,scenario))
        require(key not in seen,'Duplicate report');seen.add(key);verify([row['completion']]);receipt=read(row['completion']['path'])
        refs=[receipt[k] for k in ('report','full_diagnostics','integrity','framework_status')];verify(refs);artifacts += refs+[row['completion']]
        full=read(receipt['full_diagnostics']['path']);integrity=read(receipt['integrity']['path']);status=read(receipt['framework_status']['path'])
        require(integrity['passed'] and integrity['native_evidence_unchanged'],'Integrity failure')
        require(full['promotion_status']==full['diagnostics_summary']['promotion_status']==status['payload']['promotion_status']==STATUS,'Misleading inherited promotion status')
        ctx=full['diagnostics_summary']['pending_funding_context'];profile=ctx['profile']
        require(profile['primary_profile']==name and profile['period']==period and ctx['scenario']==scenario,'Wrong report case context')
        require(profile['recipe']==p['recipes'][name] and not ctx['production_enabled'] and not ctx['canonical_replaced'],'Wrong recipe or promotion context')
        m=a['accounts'][key]['metrics']
        close(full['final_metrics']['net_return_pct']*100,m['return_pct'],'Full report/native return mismatch')
        close(full['final_metrics']['max_drawdown_pct']*100,m['drawdown_pct'],'Full report/native DD mismatch')
        require(ctx['profitable_activity']==a['accounts'][key]['activity'] and ctx['actual_intervention_support']==a['accounts'][key]['support'],'Activity/support context drift')
        text=Path(receipt['report']['path']).read_text(encoding='utf-8')
        require('AUDITED OPENING PENDING-FUNDING SCREENING' in text and STATUS in text,'Missing truthful full-text context')
        checks.append({'case':key,'native_return_pct':m['return_pct'],'native_mtm_drawdown_pct':m['drawdown_pct'],
            'profitable_positions':ctx['profitable_activity']['profitable_positions'],'passed':True})
    require(seen=={'/'.join(r[k] for k in ('name','period','scenario')) for r in rows},'Report/native population mismatch')
    verify(p['programs']+p['bindings']+artifacts)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'full_report_verification.json',{'passed':True,'verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'native_cases':16,'exact_parent_bridges':8,'existing_engine_full_reports':16,'registered_programs':len(p['programs']),
        'registered_bindings':len(p['bindings']),'report_artifacts':len(artifacts),'checks':checks,'artifacts':artifacts,
        'verifier':bound(__file__),'study_complete':False,'next_experiments_authorized':False,
        'scope':'All existing-engine full reports and exact metrics/context authenticated. Economic recommendation remains required; no new experiment or promotion.'})
    print('Verified all16 full reports, metrics, recipe contexts and promotion statuses.',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('root','output'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();verify_reports(a.root,a.output)

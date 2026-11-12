"""Authenticate all sixteen report paths without rendering or replaying anything."""
import argparse, sys
from pathlib import Path
from datetime import datetime, timezone
REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound, verify
from backtests.strategies.kalcb.pending_funding_qualification_assessment_v1 import completion_gate, require
from backtests.strategies.kalcb.research_process_v1 import exact_process
from backtests.strategies.kalcb.pending_funding_qualification_full_diagnostics_v1 import authenticated_protocol
from backtests.strategies.kalcb.pending_funding_qualification_report_checks_v1 import check_report


def verify_reports(root, output):
    rows=completion_gate(root); reports=root/'full_reports_v1'; p=authenticated_protocol(reports)
    require(not exact_process(read(reports/'supervisor_start.json')), 'Report supervisor still running')
    status=read(reports/'status.json')
    require(not status.get('child') or not exact_process(status['child']), 'Report child still running')
    done=read(reports/'completion.json')
    require(done['complete'] and len(done['reports'])==16, 'Incomplete report inventory')
    a=read(p['assessment']['path']); verify(a['bindings'])
    checks=[]; artifacts=[]; seen=set()
    for row in done['reports']:
        name,period,scenario=(row[k] for k in ('name','period','scenario')); key='/'.join((name,period,scenario))
        require(key not in seen, 'Duplicate report'); seen.add(key)
        require(row['reused']==(scenario in ('baseline','combined')), 'Wrong reuse designation')
        verify([row['completion']])
        check=check_report(row['completion']['path'],key,a['accounts'][key],p['recipes'][name],
            p['profiles'][name][period][scenario]['folder'],row['reused'])
        artifacts+=check['artifacts']; checks.append(check)
    require(seen=={'/'.join(r[k] for k in ('name','period','scenario')) for r in rows}, 'Report/native scope mismatch')
    verify(p['programs']+p['bindings']+artifacts)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'full_report_verification.json',{'passed':True,'verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'accounted_cases':16,'fresh_native_cases':8,'authenticated_native_reuse':8,'exact_parent_bridges':8,
        'existing_engine_full_reports':16,'fresh_reports':8,'unchanged_v25_reports':8,
        'registered_programs':len(p['programs']),'registered_bindings':len(p['bindings']),
        'report_artifacts':len(artifacts),'checks':checks,'artifacts':artifacts,'verifier':bound(__file__),
        'study_complete':False,'economic_judgment_pending':True,
        'scope':'Exact metrics/context/statuses for eight new stress reports and eight unchanged V25 baseline/combined reports. No funded, independent or live qualification.'})
    print('Verified eight fresh and eight authentically reused full reports.',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('root','output'): ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();verify_reports(a.root,a.output)

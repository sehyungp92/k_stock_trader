"""Read-only, authenticated progress snapshot for the background OLR study."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.olr.economic_evidence import completed_case,compare


def snapshot(root):
    paths=[root/'cases/train/baseline_baseline',root/'cases/train/combined_baseline_attempt1']
    generation=root/'acquisition_contract_v3'
    paths += [generation/'cases/train'/f'{m}_baseline'
              for m in ('lifecycle','allocation','management','combined')]
    evidence={};records=[]
    for path in paths:
        row=dict(path=str(path),status='pending')
        if (path/'completion.json').exists():
            try:
                value=completed_case(path);evidence[str(path)]=value
                row.update({k:v for k,v in value.items() if k!='acquired'})
                row['status']='complete' if value['valid'] else 'invalid_execution'
            except (ValueError,KeyError,OSError) as exc:
                row.update(status='invalid_receipt',error=str(exc))
        records.append(row)
    control=evidence.get(str(paths[0]))
    if control:
        for row in records[2:]:
            if row['path'] in evidence:
                try:row['comparison']=compare(control,evidence[row['path']])
                except ValueError as exc:row['comparison_error']=str(exc)
    return dict(utc=datetime.now(timezone.utc).isoformat(),cases=records,
        completed=sum(r['status']=='complete' for r in records),expected_initial_completed=6,
        executions_finished=sum(r['status']!='pending' for r in records),
        invalid_finished=sum(r['status'].startswith('invalid') for r in records),
        funded_successor_qualified=False,
        note='Initial nominal diagnostics only. Full original, guard, selected, stress, capped, meaningful-frequency and OOS gates remain required. Invalid or provisional outputs cannot qualify.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();result=snapshot(args.root)
    folder=args.root/'progress_observations';folder.mkdir(exist_ok=True)
    path=folder/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    with path.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

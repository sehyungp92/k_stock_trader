"""Bind reconciled source choices before any fine-clock performance outcomes."""
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'data/backtests/analysis/olr_participation_20261011'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()


def main():
    target=OUT/'minute_exit_source_plan.json'
    if target.exists():raise FileExistsError('Do not replace a bound source plan')
    sources={str(Path(__file__).resolve()):sha(__file__)}
    def read(path):
        sources[str(path)]=sha(path);return json.loads(path.read_text(encoding='utf-8'))
    def bind(records):
        for name,digest in records.items():
            if name in sources:assert sources[name]==digest
            else:assert sha(name)==digest;sources[name]=digest
    inventory=read(OUT/'minute_exit_source/complete.json');bind(inventory['sources'])
    windows={}
    for window in ('train','oos'):
        path=OUT/'minute_exit_source'/f'{window}.json'
        assert sha(path)==inventory['outputs'][window+'.json']
        raw=read(path);overrides={};discrepant=set()
        for stage in (window,window+'_completion'):
            receipt=read(OUT/'minute_exit_source_resolution'/stage/'resolution.json')
            bind(receipt['sources'])
            for row in receipt['sessions']:
                key=(row['symbol'],row['date']);discrepant.add(key)
                if row['status']!='resolved_exact_aggregation':continue
                assert key not in overrides
                bind({row['output']:row['output_sha256']})
                overrides[key]=dict(symbol=key[0],date=key[1],path=row['output'],sha256=row['output_sha256'],
                    minute_path_hash=row['minute_path_hash'],receipt_stage=stage)
        assert discrepant==set(overrides),'Unresolved or ambiguous sessions cannot be accepted'
        primary={r['symbol']:dict(path=r['source_path'],sha256=r['source_sha256']) for r in raw['symbols'] if r['native_rows']}
        assert len(primary)==raw['declared_symbols']
        windows[window]=dict(period=raw['period'],primary=primary,
            whole_session_overrides=[overrides[k] for k in sorted(overrides)],
            resolved_discrepant_sessions=len(overrides),initial_mismatch_bins=raw['counts']['mismatch_bins'])
    result=dict(version='olr-minute-source-plan-v5',sources=sources,windows=windows,
        no_pnl_consumed=True,source_selection='Unique complete-session paths exactly reconstructing frozen native 5m OHLCV using the native session parser.',
        not_yet_fine_execution_qualified=True,
        obligations=['Recompute fine-input provenance, basis, calendar, raw parsing and event chronology in the new native engine.',
            'Retain all native coarse source/feature audits; verify every actual minute consumed separately.',
            'Use completed minutes, actual auction match times and causal preceding-volume capacity; never stale volume across gaps.',
            'Run a matched fine-clock control before attributing changes to faster exits.',
            'Do not modify any raw archive, previous frozen source generation or running entry experiment.'])
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(path=str(target),sha256=sha(target),windows={w:r['resolved_discrepant_sessions'] for w,r in windows.items()})))


if __name__=='__main__':main()

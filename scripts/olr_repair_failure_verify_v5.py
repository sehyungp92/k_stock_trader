"""Verify new diagnostic evidence and immutable parent bindings, no replay."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys

from scripts.olr_repair_failure_diagnosis_v5 import REPO, ROOT, OUT, sha


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    sources = {}
    for name in ['one_minute_feasibility.json','frequency_feasibility_train.json','frequency_feasibility_oos.json','support_checks.json','receipt.json']:
        result = read(OUT / name)
        for path, expected in result['source_sha256'].items():
            if path in sources:
                assert sources[path] == expected, path
            sources[path] = expected
    for path, expected in sources.items():
        assert sha(path) == expected, ('Changed diagnostic input',path)
    cache_proofs = {}
    for w in ['train','oos']:
        study = read(ROOT / 'repair_v4/runtime/c4f49be01ed09b8dc' / w / 'baseline/study.json')
        code = Path(study['code'])
        desc = read(ROOT / f'continuous_panel_v1/cache_sources/{w}/source.json')
        index_path = Path(desc['cache']) / 'index.json'
        assert sha(index_path) == desc['index_sha256']
        assert sha(desc['source_receipt']) == desc['source_receipt_sha256']
        index = read(index_path)
        signed = index.pop('index_signature')
        sig = hashlib.sha256(json.dumps(index,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
        assert signed == sig
        assert index['identity']['base']['database_sha256'] == sha(code / study['database']) == study['database_hash']
        for path,expected in index['inputs'].items():
            assert sha(code/path) == expected, ('Cache underlying data mismatch',w,path)
        cache_proofs[w] = {'index_signature_pass':True,'database_pass':True,'underlying_source_files_verified':len(index['inputs'])}
    parent = REPO / 'data/backtests/analysis/olr_repair_20261011'
    assert sha(parent/'results.json') == '85d1a4cd53a1b7b3466093bc9eb4f235fd0b838a9b5811ba8d9be79cefd9cc83'
    assert sha(parent/'selected_research_policy.json') == '49a88f0b5c015cc1e5c262a38641c80402977d8f2bf7f1957125754a7de778eb'
    micro = read(OUT/'one_minute_feasibility.json')
    for w,window in micro['windows'].items():
        for r in window['rows']:
            assert sum(x['qty'] for x in r['fills']) == r['early_qty'] <= r['qty']
            for fill in r['fills']:
                assert datetime.fromisoformat(r['trigger']['decision_available']) <= datetime.fromisoformat(fill['time']) < datetime.fromisoformat(r['native_first_sale'])
                assert fill['qty'] <= int(fill['lag_volume']*.01)
            delta = sum(x['qty']*(x['price']-r['native_exit_vwap']) for x in r['fills'])
            assert abs(delta-r['isolated_gross_exit_value_change']) < 1e-7
    assert micro['windows']['oos']['mismatch_count'] == 0
    receipt = read(OUT/'receipt.json')
    for path,expected in receipt['artifacts'].items():
        assert sha(OUT/path) == expected
    reviewed = '--chart-reviewed' in sys.argv
    result = {'passed':True,'chart_reviewed':reviewed,'source_files_verified':len(sources),'cache_proofs':cache_proofs,
              'frozen_parent_bindings_unchanged':True,'early_sale_timing_and_lagged_capacity_arithmetic_pass':True,
              'scope':'Integrity and arithmetic verification of diagnostic screens, NOT native strategy qualification.',
              'script_sha256':sha(__file__),'report_receipt_sha256':sha(OUT/'receipt.json')}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

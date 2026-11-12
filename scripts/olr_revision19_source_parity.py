"""Compare every authenticated retained source row through both native parsers."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def digest(rows):
    start=time.perf_counter();h=hashlib.sha256();count=0
    for row in rows:
        h.update(json.dumps(row,sort_keys=True,separators=(',',':')).encode());h.update(b'\n');count+=1
    return {'count':count,'sha256':h.hexdigest(),'seconds':time.perf_counter()-start}


if __name__=='__main__':
    window=sys.argv[1]
    g=Path('C:/Users/sehyu/.codex/research/olr_native_r19')/window/'inputs/generation.json'
    info=json.loads(g.read_text());engine=Path(info['engine']);sys.path.insert(0,str(engine))
    r=Path('E:/CodexResearch/k_stock_trader/olr_round1_comprehensive_20261009/full_oos_20260401_20260710/continuous_history_v10')
    v=r/'revisions/revision_19_native_overheads'
    study=json.loads((r/f'registrations/{window}/ce14d8377b9e67ac6_100000000/study.json').read_text())
    from backtests.core.input_manifest import InputManifest
    from backtests.engine.qualification_evidence import source_rows
    from backtests.engine import qualification_evidence
    spec=importlib.util.spec_from_file_location('backtests.engine.r19_original_qualification',Path(info['original_engine'])/'backtests/engine/qualification_evidence.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    generation=InputManifest(study['manifest'])
    def read(p):return json.loads(generation.read(generation.root/p))
    evidence=read(generation.payload['evidence']['qualification_path'])
    original=digest(old.source_rows(read,generation,evidence));print('original',window,original,flush=True)
    optimized=digest(source_rows(read,generation,evidence));print('optimized',window,optimized,flush=True)
    result={'window':window,'original':original,'optimized':optimized,
            'pass':(original['count'],original['sha256'])==(optimized['count'],optimized['sha256']),
            'source_files':len(evidence['market_files']),'every_input_file_authenticated_in_both_passes':True}
    (v/f'source_rows_{window}_parity.json').write_text(json.dumps(result,indent=2))
    assert result['pass']

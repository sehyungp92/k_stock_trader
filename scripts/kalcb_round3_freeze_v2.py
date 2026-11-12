"""Preserve v1; quarantine non-authoritative Python cache drift explicitly."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kalcb_round3_freeze_v1 as original

original.ROOT = original.ROOT.parent / 'attempt_2'
source = Path(original.__file__).read_text(encoding='utf-8')
source = source.replace("ROOT = BASE/'kalcb_round3_20261017/attempt_1'", "ROOT = BASE/'kalcb_round3_20261017/attempt_2'")
old = '''        if digest(path)!=sha:raise ValueError('Changed baseline input '+path)
        return dict(path=path,sha256=sha)
    with ThreadPoolExecutor(max_workers=2) as pool: verified=list(pool.map(check,inputs.items()))'''
new = '''        actual = digest(path)
        return dict(path=path,sha256=sha,actual_sha256=actual,
                    python_cache=Path(path).suffix == '.pyc' and '__pycache__' in Path(path).parts)
    with ThreadPoolExecutor(max_workers=2) as pool: inspected=list(pool.map(check,inputs.items()))
    drift = [x for x in inspected if x['sha256'] != x['actual_sha256']]
    write(ROOT/'freeze/input_drift.json', dict(all_differences=drift,
        policy='Generated .pyc is quarantined; workers import authenticated .py in a separate cache-free source capsule.',
        original_failed_generation=str(ROOT.parent/'attempt_1'), inspected=len(inspected)))
    fatal = [x for x in drift if not x['python_cache']]
    if fatal: raise ValueError('Non-cache baseline input drift: ' + repr(fatal[:5]))
    verified = [dict(path=x['path'],sha256=x['sha256']) for x in inspected if not x['python_cache']]
    write(ROOT/'freeze/excluded_python_caches.json', [x for x in inspected if x['python_cache']])'''
if old not in source:
    raise RuntimeError('Freeze repair anchor drifted')
source = source.replace(old, new)
exec(compile(source, __file__, 'exec'), {'__name__': '__main__', '__file__': __file__})

"""Revision19: isolated replay generations; never alter reconstruction generations.

Every source observation remains authenticated by the native InputManifest and
every replay runs the complete native audit. No audit verdict is cached here.
"""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def available_memory_mb():
    """Read host headroom without spawning a process or changing OS settings."""
    if sys.platform != 'win32':
        return None
    import ctypes
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong),
                    *[(name,ctypes.c_ulonglong) for name in
                      ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]]
    status=MemoryStatus();status.length=ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('Cannot read physical memory headroom')
    return status.available/(1024*1024)


def additional_job_allowed(policy, active, available_mb):
    # Make forward progress with one job. Extra slots require measured headroom;
    # active evaluations are never suspended or terminated by this policy.
    return (not policy or active == 0 or (available_mb is not None and
            available_mb >= policy['minimum_available_mb_for_additional_job']))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temp.replace(path)


def copy_verified(source, target, digest):
    source, target = Path(source), Path(target)
    if target.exists():
        if sha(target) != digest:
            raise ValueError('Changed replay generation input: ' + str(target))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha(target) != digest:
        raise ValueError('Changed original input while copying: ' + str(source))


def replace_exact(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Patch context differs from reviewed source: ' + before[:100])
    return text.replace(before, after, 1)


def optimized_source(name, text):
    if name == 'strategy_common/session_coverage.py':
        text = replace_exact(text, 'from datetime import datetime, time, timedelta',
                             'from datetime import datetime, time, timedelta\nfrom functools import lru_cache')
        start = text.index('    start = datetime.combine(day, opening, tzinfo=KST)')
        end = text.index('    contradictory = sorted(observed & halted)', start)
        original = text[start:end]
        helper = original.replace('calendar.closing_auction_minutes','closing_auction_minutes')
        helper = helper.replace('    match = calendar.closing_match_time(symbol, day)\n','')
        helper = helper.replace('    periodic_sources = calendar.periodic_auction_sources(symbol, day)\n','')
        helper = helper.replace('for first, last, _ in calendar.intraday_halts_for(symbol, day):','for first, last in halts:')
        text = text[:start] + (
            '    # Cache only the sourced calendar grid, never observations or audit\n'
            '    # outcomes. Every fact affecting the grid is part of the key.\n'
            '    match = calendar.closing_match_time(symbol, day)\n'
            '    periodic_sources = calendar.periodic_auction_sources(symbol, day)\n'
            '    halts = tuple((first, last) for first, last, _ in calendar.intraday_halts_for(symbol, day))\n'
            '    expected, halted = _calendar_grid(day, minutes, opening, closing,\n'
            '        calendar.closing_auction_minutes, match, bool(periodic_sources), halts)\n'
        ) + text[end:]
        text += ('\n\n@lru_cache(maxsize=4096)\n'
                 'def _calendar_grid(day, minutes, opening, closing, closing_auction_minutes, match, periodic_sources, halts):\n'
                 '    """Exact original calendar construction, shared only by equal facts."""\n'
                 + helper + '    return frozenset(expected), frozenset(halted)\n')
    if name == 'backtests/core/input_manifest.py':
        text = replace_exact(text,
            '        for name in payload["files"]:\n            if not (self.root / name).resolve().is_relative_to(self.root.resolve()):\n                raise ValueError("Input manifest path escapes its generation")\n        self.files = {str((self.root / name).resolve()): digest for name, digest in payload["files"].items()}',
            '        resolved_root = self.root.resolve()\n        self.files = {}\n        for name, digest in payload["files"].items():\n            resolved = (self.root / name).resolve()\n            if not resolved.is_relative_to(resolved_root):\n                raise ValueError("Input manifest path escapes its generation")\n            self.files[str(resolved)] = digest')
    if name in ('backtests/core/input_manifest.py', 'backtests/auto/shared/cache_keys.py'):
        text = replace_exact(text,
            '    for chunk in encoder.iterencode(value):\n        digest.update(chunk.encode("utf-8"))',
            '    # Bound temporary JSON to 32 records; retain streaming fallback for\n    # arbitrary objects and small/non-sequence inputs. All canonical bytes match.\n    if type(value) in (list, tuple) and len(value) >= 128:\n        digest.update(b"[")\n        for start in range(0, len(value), 32):\n            if start:\n                digest.update(b",")\n            block = encoder.encode(value[start:start + 32])\n            digest.update(block[1:-1].encode("utf-8"))\n        digest.update(b"]")\n    else:\n        for chunk in encoder.iterencode(value):\n            digest.update(chunk.encode("utf-8"))')
    if name == 'backtests/engine/qualification_evidence.py':
        text = replace_exact(text,
            '        indexed, grouped = {}, defaultdict(list)\n        for row in source_rows(read, generation, evidence):\n            if key(row) in indexed:\n                raise ValueError("duplicate_source_event")\n            indexed[key(row)] = values(row)',
            '        indexed, grouped = {}, defaultdict(list)\n        required_event_keys = {key(row) for row in expected}\n        seen_source_keys = set()\n        for row in source_rows(read, generation, evidence):\n            source_key = key(row)\n            # Duplicate detection and timestamp coverage still consume EVERY\n            # retained row. Only matching execution rows need retained OHLCV.\n            if source_key in seen_source_keys:\n                raise ValueError("duplicate_source_event")\n            seen_source_keys.add(source_key)\n            if source_key in required_event_keys:\n                indexed[source_key] = values(row)')
        text = replace_exact(text,
            '        for row in frame.itertuples(index=False):\n            yield MarketBar(symbol=item["symbol"], timestamp=ensure_kst(row.timestamp), timeframe=item["timeframe"],',
            '        for row in frame.itertuples(index=False):\n            stamp = row.timestamp\n            # Pandas timezone conversion is costly per row. Preserve fractional\n            # nanoseconds through the original parser; never truncate them.\n            if isinstance(stamp, pd.Timestamp) and stamp.nanosecond == 0:\n                stamp = stamp.to_pydatetime()\n            yield MarketBar(symbol=item["symbol"], timestamp=ensure_kst(stamp), timeframe=item["timeframe"],')
    if name == 'backtests/engine/integrity.py':
        text = replace_exact(text,
            '                if not bar_execution_issue(MarketBar(**source)):',
            '                native_bar = MarketBar(**source)\n                if not bar_execution_issue(native_bar):')
        text = replace_exact(text,
            'bar_reopens_after_label(MarketBar(**source)) or bar_has_periodic_auction(MarketBar(**source)) or bar_is_delayed_closing_auction(MarketBar(**source))',
            'bar_reopens_after_label(native_bar) or bar_has_periodic_auction(native_bar) or bar_is_delayed_closing_auction(native_bar)')
    if name == 'scripts/kalcb_executed_value_study.py':
        text = replace_exact(text, 'def load_day(conn, day, source, symbols=None):',
                             'def load_day(conn, day, source, symbols=None, *, stream=False):')
        text = replace_exact(text,
            '    return [MarketBar(s, datetime.fromisoformat(t), "5m", *values, source="kis_krx_parquet",\n                      source_fingerprint=source) for s, t, *values in conn.execute(query, parameters)]',
            '    bars = (MarketBar(s, datetime.fromisoformat(t), "5m", *values, source="kis_krx_parquet",\n                      source_fingerprint=source) for s, t, *values in conn.execute(query, parameters))\n    return bars if stream else list(bars)')
    if name == 'scripts/portfolio_synergy_study.py':
        text = replace_exact(text,
            'bars = load_day(connection, str(day), study["database_hash"], symbols | set(pm["observation_symbols"]))',
            'bars = load_day(connection, str(day), study["database_hash"], symbols | set(pm["observation_symbols"]), stream=True)')
    return text


PATCHED = ('strategy_common/session_coverage.py', 'backtests/core/input_manifest.py', 'backtests/auto/shared/cache_keys.py',
           'backtests/engine/qualification_evidence.py', 'backtests/engine/integrity.py',
           'scripts/kalcb_executed_value_study.py', 'scripts/portfolio_synergy_study.py')


def build_generation(study_path, destination):
    """Copy bytes (never hardlink) before patching an isolated native generation."""
    study = read(study_path)
    original = Path(study['code'])
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Preserve the existing engine; choose a new revision destination')
    engine = destination/'files'
    manifest = read(study['manifest'])
    files = dict(manifest['files'])
    for directory in ('backtests', 'strategy_common', 'strategy_kalcb', 'strategy_olr', 'strategy_pcim',
                      'deployment', 'instrumentation', 'oms', 'oms_client', 'kis_core', 'scripts'):
        for p in (original/directory).rglob('*.py'):
            files[p.relative_to(original).as_posix()] = sha(p)
    for p in (original/'kis_core/data').glob('*.yaml'):
        files[p.relative_to(original).as_posix()] = sha(p)
    for name, digest in files.items():
        copy_verified(original/name, engine/name, digest)
    changes = {}
    for name in PATCHED:
        p = engine/name
        before = files[name]
        value = optimized_source(name, p.read_text(encoding='utf-8'))
        p.write_text(value, encoding='utf-8', newline='\n')
        changes[name] = {'before':before, 'after':sha(p)}
        assert sha(original/name) == before
    command = 'from backtests.auto.shared.cache_keys import source_generation_hash; print(source_generation_hash())'
    actual = subprocess.check_output([sys.executable, '-B', '-c', command], cwd=engine, text=True).strip()
    info = {'revision':19, 'engine':str(engine), 'original_engine':str(original),
            'original_source_hash':study['source_generation_hash'], 'source_generation_hash':actual,
            'changed_files':changes, 'files':{name:sha(engine/name) for name in files},
            'builder_sha256':sha(__file__), 'source_observations_changed':False}
    write(destination/'generation.json', info)
    return destination/'generation.json'


def prepare_replay(study_path, generation_path, destination):
    """Bind the same inputs/configurations to the newly reviewed replay code."""
    info = read(generation_path)
    study, original = read(study_path), read(study_path)
    if study['source_generation_hash'] != info['original_source_hash']:
        raise ValueError('Replay acceleration source generation mismatch')
    engine, source = Path(info['engine']), Path(study['code'])
    manifest = read(study['manifest'])
    manifest.pop('manifest_hash')
    for name, digest in list(manifest['files'].items()):
        if name in info['changed_files']:
            changed = info['changed_files'][name]
            if digest != changed['before'] or sha(engine/name) != changed['after']:
                raise ValueError('Changed native implementation binding: ' + name)
            manifest['files'][name] = changed['after']
        elif info['files'].get(name) != digest:
            # Static inputs were authenticated when the isolated generation was
            # built. Native open_verified still rehashes every consumed file on
            # every audit. Only new per-trial inputs need copying here.
            copy_verified(source/name, engine/name, digest)
    # Bind every executable source file, including otherwise unused scripts.
    for name, digest in info['files'].items():
        if name.endswith('.py') or name.startswith('kis_core/data/'):
            manifest['files'][name] = digest
    study.update(code=str(engine), source_generation_hash=info['source_generation_hash'])
    manifest['source_generation_hash'] = info['source_generation_hash']
    token = signature({'original':original,'original_manifest':sha(original['manifest']),
                       'generation':sha(generation_path)})[:24]
    mp = engine.parent/(token+'.manifest.json')
    study['manifest'] = str(mp)
    study['configuration']['input_manifest_path'] = str(mp)
    study['spec_path'] = 'research/olr_comprehensive/revision19_specs/'+token+'.json'
    spec = {k:v for k,v in study.items() if k not in ('manifest','code','spec_path','source_generation_hash')}
    write(engine/study['spec_path'], spec)
    manifest['files'][study['spec_path']] = sha(engine/study['spec_path'])
    write(mp, {**manifest, 'manifest_hash':signature(manifest)})
    write(destination, study)
    write(Path(destination).with_name('native_acceleration_receipt.json'), {
        'original_study':str(study_path), 'original_study_sha256':sha(study_path),
        'replay_study_sha256':sha(destination), 'generation':str(generation_path),
        'generation_sha256':sha(generation_path), 'helper_sha256':sha(__file__),
        'original_source_hash':original['source_generation_hash'],
        'replay_source_hash':study['source_generation_hash'], 'full_native_audit_required':True})
    return Path(destination)

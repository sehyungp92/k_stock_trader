"""Read-only native source checks for the 0036 OOS calendar assembly repair."""
from datetime import date
import json
from pathlib import Path
import sqlite3
import sys


def main():
    root = Path(sys.argv[1])
    ready = json.loads((root/'native_engines/oos/ready.json').read_text(encoding='utf8'))
    files = Path(ready['files'])
    # Called with -c and cwd=files: imports must resolve to this frozen engine.
    import kis_core.trading_calendar as calendars
    assert Path(calendars.__file__).resolve().is_relative_to(files)
    from backtests.core.input_manifest import InputManifest
    from backtests.engine.olr_minute_source_v5 import origin_verifier, verify_closing_clock_sources
    from backtests.auto.shared.cache_keys import source_generation_hash
    generation = InputManifest(root/'native_engines/oos/calendar_preflight.manifest.json')
    with generation.activate():
        assert source_generation_hash() == ready['study']['source_generation_hash']
        definition = json.loads(generation.read(files/ready['study']['minute_source_definition']))
        calendar = calendars.get_trading_calendar()
        features = json.loads(generation.read(files/ready['study']['relative_context_features']))
        days = [date.fromisoformat(d) for d in features['evaluation_days']]
        assert len(days) == 69
        assert calendar.session_coverage[0] <= min(days) <= max(days) <= calendar.session_coverage[1]
        assert all(calendar.is_trading_day(d) for d in days)
        assert str(calendar.closing_match_time('042700', date(2026,5,12))) == '15:32:00'
        session, clear, sources = origin_verifier(generation, definition)
        corrections = verify_closing_clock_sources(generation, definition, session, clear)
        assert corrections['corrected_sessions'] == 99 and corrections['clock_corrections_verified']
        # The coarse and fine point themselves are already retained; no clock guard is relaxed.
        with sqlite3.connect((files/definition['canonical_database']).as_uri()+'?mode=ro',uri=True) as db:
            point = db.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND ts=?',
                ('042700','2026-05-12T15:32:00+09:00')).fetchone()
        with sqlite3.connect((files/definition['minute_database']).as_uri()+'?mode=ro',uri=True) as db:
            fine = db.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND ts=?',
                ('042700','2026-05-12T15:32:00+09:00')).fetchone()
        assert point == fine and point is not None
        result = dict(pass_=True,source=ready['study']['source_generation_hash'],calendar_sessions=69,
            native_closing_clock_verification=corrections,authenticated_origin_files=len(sources),
            reproduced_failed_boundary=dict(symbol='042700',day='2026-05-12',corrected_point=point),
            scope='Existing OOS source calendar restored; native independent source verifier, no financial outcome and no policy change.')
        with (root/'calendar_preflight.json').open('x',encoding='utf8') as stream:
            json.dump(result,stream,indent=2)
        print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()

"""Regression protection for the IS/OOS intraday warm-up boundary."""
from datetime import date
import sqlite3

from scripts.olr_round1_comprehensive_study import reconstruction_days


def test_oos_scoring_keeps_older_and_is_history_without_future_sessions(tmp_path):
    database=tmp_path/'retained.sqlite'
    retained=('2025-05-12','2025-08-07','2026-03-30','2026-03-31',
              '2026-04-01','2026-04-02','2026-04-03')
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE bars (day TEXT)')
        connection.executemany('INSERT INTO bars VALUES (?)',[(d,) for d in retained])
    scored=[date(2026,4,1),date(2026,4,2)]
    actual=reconstruction_days(database,scored)
    assert actual==[date.fromisoformat(d) for d in retained[:-1]]
    assert actual[-len(scored):]==scored
    assert date(2025,5,12) in actual and date(2026,3,31) in actual
    assert date(2026,4,3) not in actual

"""Durable ordered outbox for the OMS's existing fill and order-event producer."""
import json
from pathlib import Path
import sqlite3
from contextlib import contextmanager


class ExecutionFeedback:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS feedback (sequence INTEGER PRIMARY KEY AUTOINCREMENT, identity TEXT UNIQUE NOT NULL, strategy_id TEXT NOT NULL, payload TEXT NOT NULL)")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def append(self, identity, strategy_id, payload):
        encoded = json.dumps(payload, sort_keys=True, default=str)
        with self._connect() as db:
            previous = db.execute("SELECT payload FROM feedback WHERE identity=?", (identity,)).fetchone()
            if previous:
                if previous[0] != encoded:
                    raise ValueError("Feedback identity collision")
                return
            db.execute("INSERT INTO feedback(identity,strategy_id,payload) VALUES(?,?,?)", (identity, strategy_id, encoded))

    def read(self, strategy_id, after=0, limit=500):
        with self._connect() as db:
            rows = db.execute("SELECT sequence,payload FROM feedback WHERE strategy_id=? AND sequence>? ORDER BY sequence LIMIT ?", (strategy_id.upper(), int(after), min(max(int(limit), 1), 1000))).fetchall()
        return [{"sequence": seq, **json.loads(payload)} for seq, payload in rows]

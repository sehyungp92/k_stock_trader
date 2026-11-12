"""Durable local projection for standalone file-based instrumentation."""
from contextlib import contextmanager
import hashlib
import json
import os
import math
from pathlib import Path
import sqlite3


class TradeProjection:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "trade_projection.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,trade_id TEXT,execution_id TEXT,day TEXT,payload TEXT);
                CREATE TABLE IF NOT EXISTS open_trades(trade_id TEXT PRIMARY KEY,payload TEXT);
                CREATE TABLE IF NOT EXISTS cursors(consumer TEXT PRIMARY KEY,sequence INTEGER);
                CREATE TABLE IF NOT EXISTS imported_files(path TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS dirty_days(day TEXT PRIMARY KEY);
            """)
            # Preserve pre-migration JSONL records before making exports rebuildable.
            for path in sorted((self.root / "trades").glob("trades_*.jsonl")):
                if db.execute("SELECT 1 FROM imported_files WHERE path=?", (path.name,)).fetchone():
                    continue
                for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                    item = json.loads(line)
                    identity = f"legacy:{path.name}:{index}:" + hashlib.sha256(line.encode()).hexdigest()
                    db.execute("INSERT OR IGNORE INTO events VALUES(?,?,?,?,?)", (identity, item.get("trade_id", ""),
                        item.get("execution_id", ""), path.stem.removeprefix("trades_"), line))
                db.execute("INSERT INTO imported_files VALUES(?)", (path.name,))
                db.execute("INSERT OR IGNORE INTO dirty_days VALUES(?)", (path.stem.removeprefix("trades_"),))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def open_trades(self):
        with self.connect() as db:
            return {key: json.loads(value) for key, value in db.execute("SELECT trade_id,payload FROM open_trades")}

    def execution(self, trade_id, execution_id):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM events WHERE trade_id=? AND execution_id=? ORDER BY rowid LIMIT 1", (trade_id, execution_id)).fetchone()
        return json.loads(row[0]) if row else None

    def reconcile_costs(self, trade_id, execution_id, fees_paid, *, source_ref):
        """Append an idempotent cost revision without modifying execution/state rows."""
        fees = float(fees_paid)
        if not source_ref or not math.isfinite(fees) or fees < 0:
            raise ValueError("Verified source and finite nonnegative costs required")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            record = db.execute("SELECT day,payload FROM events WHERE trade_id=? AND execution_id=? ORDER BY rowid DESC LIMIT 1",
                                (trade_id, execution_id)).fetchone()
            if record is None:
                raise ValueError("Unknown execution for cost correction")
            day, payload = record
            event = json.loads(payload)
            if event.get("fees_paid") == fees:
                return False
            if event.get("gross_pnl") is None:
                raise ValueError("Execution gross economics missing")
            revision = int(event.get("cost_revision", 0)) + 1
            identity = f"{trade_id}:exit:{execution_id}:cost:{revision}"
            event.update(cost_revision=revision, cost_source_ref=source_ref,
                         fees_paid=fees, net_pnl=round(event["gross_pnl"] - fees, 4),
                         pnl=round(event["gross_pnl"] - fees, 4),
                         realized_pnl_krw=round(event["gross_pnl"] - fees, 4),
                         pnl_basis="net_with_supplied_costs", schema_version="trade_execution_v3",
                         event_type="trade_execution")
            metadata = dict(event.get("event_metadata") or {})
            metadata.update(event_type="trade_execution", schema_version="trade_execution_v3",
                            event_id=hashlib.sha256(identity.encode()).hexdigest(), payload_key=identity)
            event["event_metadata"] = metadata
            db.execute("INSERT INTO events VALUES(?,?,?,?,?)", (identity, trade_id, execution_id, day, json.dumps(event)))
            db.execute("INSERT OR IGNORE INTO dirty_days VALUES(?)", (day,))
        self.export()
        return True

    def cursor(self, consumer):
        with self.connect() as db:
            row = db.execute("SELECT sequence FROM cursors WHERE consumer=?", (consumer,)).fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _advance(db, consumer, sequence):
        if consumer and sequence is not None:
            db.execute("INSERT INTO cursors VALUES(?,?) ON CONFLICT(consumer) DO UPDATE SET sequence=MAX(sequence,excluded.sequence)",
                       (consumer, int(sequence)))

    def advance(self, consumer, sequence):
        with self.connect() as db:
            self._advance(db, consumer, sequence)

    def commit(self, event, remaining, *, consumer=None, sequence=None):
        identity = f"{event['trade_id']}:{event.get('stage')}:{event.get('execution_id') or 'entry'}"
        day = str(event.get("exit_time") or event.get("entry_time") or "")[:10]
        if not day:
            from datetime import datetime, timezone
            day = datetime.now(timezone.utc).date().isoformat()
        with self.connect() as db:
            old = db.execute("SELECT payload FROM events WHERE id=?", (identity,)).fetchone()
            if old:
                previous = json.loads(old[0])
                for field in ("trade_id", "stage", "execution_id", "exit_price", "position_size", "fees_paid"):
                    if previous.get(field) != event.get(field):
                        raise ValueError("Trade projection identity collision")
                self._advance(db, consumer, sequence)
                return False
            db.execute("INSERT INTO events VALUES(?,?,?,?,?)", (identity, event["trade_id"], event.get("execution_id", ""), day,
                                                                  json.dumps(event, default=str)))
            db.execute("INSERT OR IGNORE INTO dirty_days VALUES(?)", (day,))
            if remaining is None:
                db.execute("DELETE FROM open_trades WHERE trade_id=?", (event["trade_id"],))
            else:
                db.execute("INSERT INTO open_trades VALUES(?,?) ON CONFLICT(trade_id) DO UPDATE SET payload=excluded.payload",
                           (event["trade_id"], json.dumps(remaining, default=str)))
            self._advance(db, consumer, sequence)
        return True

    def export(self):
        # Record ownership before publishing a file. A crash after rename must
        # never cause our own export to be imported again as legacy events.
        with self.connect() as db:
            days = [row[0] for row in db.execute("SELECT day FROM dirty_days")]
            db.executemany("INSERT OR IGNORE INTO imported_files VALUES(?)",
                           [(f"trades_{day}.jsonl",) for day in days])
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            days = [row[0] for row in db.execute("SELECT day FROM dirty_days")]
            for day in days:
                path = self.root / "trades" / f"trades_{day}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                with temporary.open("w", encoding="utf-8") as stream:
                    for (payload,) in db.execute("SELECT payload FROM events WHERE day=? ORDER BY rowid", (day,)):
                        stream.write(payload + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                db.execute("INSERT OR IGNORE INTO imported_files VALUES(?)", (path.name,))
                db.execute("DELETE FROM dirty_days WHERE day=?", (day,))

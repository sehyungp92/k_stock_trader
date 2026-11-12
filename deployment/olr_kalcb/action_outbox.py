"""Persist exact routed intents before submission; replay their stable identities."""
from contextlib import contextmanager
from dataclasses import asdict
from enum import Enum
import json
import sqlite3

from oms.intent import Intent, IntentConstraints, RiskPayload, IntentType, Urgency, TimeHorizon


def encode(value):
    return json.dumps(value, sort_keys=True, default=lambda x: x.name if isinstance(x, Enum) else str(x))


def restore_intent(payload):
    data = dict(payload)
    data["intent_type"] = IntentType[data["intent_type"]]
    data["urgency"] = Urgency[data["urgency"]]
    data["time_horizon"] = TimeHorizon[data["time_horizon"]]
    data["constraints"] = IntentConstraints(**data["constraints"])
    data["risk_payload"] = RiskPayload(**data["risk_payload"])
    return Intent(**data)


class ActionOutbox:
    def __init__(self, path):
        self.path = path
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS actions(identity TEXT PRIMARY KEY,intent TEXT NOT NULL,template TEXT NOT NULL,result TEXT)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def get(self, identity):
        with self.connect() as db:
            row = db.execute("SELECT intent,template,result FROM actions WHERE identity=?", (identity,)).fetchone()
        return tuple(json.loads(value) if value else None for value in row) if row else None

    def prepare(self, identity, intent, template):
        with self.connect() as db:
            db.execute("INSERT INTO actions(identity,intent,template) VALUES(?,?,?)",
                       (identity, encode(asdict(intent)), encode(template)))

    def finish(self, identity, result):
        with self.connect() as db:
            db.execute("UPDATE actions SET result=? WHERE identity=?", (encode(result), identity))

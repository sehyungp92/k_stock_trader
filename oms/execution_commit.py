"""Atomic execution accounting using the OMS's existing Postgres connection pool."""
from datetime import datetime
from dataclasses import asdict
import hashlib
import json
import math
import uuid


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def decode(value):
    return json.loads(value) if isinstance(value, str) else dict(value)


def execution_signature(event):
    meta = event["metadata"]
    return hashlib.sha256(canonical({
        "order": event["order_id"], "symbol": event["symbol"], "side": event["side"],
        "qty": event["qty"], "price": event["price"],
        "before": meta["filled_qty_before"], "after": meta["filled_qty_after"],
        "order_date": meta["order_date"], "account": meta["account_id"],
        "commission": meta.get("commission"), "tax": meta.get("tax"),
        "price_provenance": meta["price_provenance"],
    }).encode()).hexdigest()


def intent_signature(intent):
    payload = asdict(intent)
    for key in ("intent_id", "timestamp", "metadata", "idempotency_key"):
        payload.pop(key, None)
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


class ExecutionPersistence:
    async def verify_action_identity(self, intent):
        if not str(intent.idempotency_key).startswith("ACTION:"):
            return
        if not self._is_connected():
            raise RuntimeError("Action identity persistence unavailable")
        signature = intent_signature(intent)
        stored = await self.pool.fetchval("""INSERT INTO intent_effects VALUES($1,$2,$3)
            ON CONFLICT(oms_id,identity) DO UPDATE SET signature=intent_effects.signature RETURNING signature""",
            self.oms_id, intent.idempotency_key, signature)
        if stored != signature:
            raise ValueError("Action identity reused with different execution effects")

    async def _execution_order(self, db, order):
        if order.oms_order_id:
            row = await db.fetchrow("SELECT * FROM orders WHERE oms_id=$1 AND oms_order_id=$2::uuid FOR UPDATE",
                                    self.oms_id, order.oms_order_id)
        else:
            day = order.created_at.strftime("%Y%m%d")
            rows = await db.fetch("""SELECT * FROM orders WHERE oms_id=$1 AND kis_order_id=$2
                AND COALESCE(kis_order_date, to_char(created_at AT TIME ZONE 'Asia/Seoul','YYYYMMDD'))=$3
                FOR UPDATE""", self.oms_id, order.order_id, day)
            if len(rows) != 1:
                raise ValueError("Execution requires one canonical scoped order")
            row = rows[0]
        if not row or row["strategy_id"] != order.strategy_id or row["symbol"] != order.symbol or row["side"] != order.side:
            raise ValueError("Execution order identity mismatch")
        return row

    async def commit_execution(self, order, event, *, stop_price=None, stop_mode="OMS_WATCHER"):
        if not self._is_connected():
            raise RuntimeError("Execution persistence unavailable")
        meta = event["metadata"]
        identity = meta["execution_id"]
        signature = execution_signature(event)
        try:
            async with self.pool.acquire() as db:
                async with db.transaction():
                    # Serializes sequence allocation with commit order within an OMS.
                    await db.execute("SELECT pg_advisory_xact_lock(hashtextextended($1,0))", f"execution:{self.oms_id}")
                    previous = await db.fetchrow("SELECT signature,payload FROM execution_commits WHERE oms_id=$1 AND identity=$2",
                                                 self.oms_id, identity)
                    if previous:
                        if previous["signature"] != signature:
                            raise ValueError("Execution identity collision")
                        return {"duplicate": True, "event": decode(previous["payload"])}
                    stored_order = await self._execution_order(db, order)
                    before, after = int(meta["filled_qty_before"]), int(meta["filled_qty_after"])
                    qty, price = int(event["qty"]), float(event["price"])
                    if qty <= 0 or after - before != qty or after > stored_order["qty"] or before != stored_order["filled_qty"]:
                        raise ValueError("Execution cumulative quantity gap or conflict")
                    if not math.isfinite(price) or price <= 0:
                        raise ValueError("Execution price must be positive and finite")
                    alloc = await db.fetchrow("SELECT * FROM allocations WHERE oms_id=$1 AND symbol=$2 AND strategy_id=$3 FOR UPDATE",
                                             self.oms_id, order.symbol, order.strategy_id)
                    old_qty = int(alloc["qty"]) if alloc else 0
                    old_price = float(alloc["cost_basis"] or 0) if alloc else 0.0
                    entry_ts = alloc["entry_ts"] if alloc else None
                    fill_ts = datetime.fromisoformat(event["timestamp"])
                    new_qty = old_qty + (qty if order.side == "BUY" else -qty)
                    if new_qty < 0:
                        raise ValueError("Execution exceeds canonical allocation; reconcile broker exposure")
                    cost = (old_qty * old_price + qty * price) / new_qty if order.side == "BUY" else old_price
                    entry_ts = (entry_ts or fill_ts) if new_qty else None
                    realized = (price - old_price) * qty if order.side == "SELL" else 0.0
                    soft_stop = stop_price if stop_price is not None else (float(alloc["soft_stop_px"]) if alloc and alloc["soft_stop_px"] else None)
                    event = decode(canonical(event))
                    event["strategy_id"] = order.strategy_id
                    event["metadata"].update({
                        "previous_allocation": {"symbol": order.symbol, "strategy_id": order.strategy_id, "qty": old_qty, "cost_basis": old_price,
                            "entry_ts": alloc["entry_ts"].isoformat() if alloc and alloc["entry_ts"] else None},
                        "new_allocation": {"symbol": order.symbol, "strategy_id": order.strategy_id, "qty": new_qty,
                            "cost_basis": cost, "entry_ts": entry_ts.isoformat() if entry_ts else None, "soft_stop_px": soft_stop},
                        "realized_pnl": realized, "realized_pnl_basis": "gross_costs_unknown" if meta.get("commission") is None or meta.get("tax") is None else "gross",
                    })
                    # Scope the legacy globally unique column without discarding the broker ID in metadata.
                    await db.execute("""INSERT INTO fills(kis_exec_id,oms_order_id,strategy_id,symbol,side,qty,price,commission,tax,fill_ts,oms_id)
                        VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)""",
                        identity, stored_order["oms_order_id"], order.strategy_id, order.symbol, order.side, qty, price,
                        meta.get("commission"), meta.get("tax"), fill_ts, self.oms_id)
                    await db.execute("""INSERT INTO allocations(oms_id,symbol,strategy_id,qty,cost_basis,entry_ts,soft_stop_px)
                        VALUES($1,$2,$3,$4,$5,$6,$7) ON CONFLICT(oms_id,symbol,strategy_id) DO UPDATE SET
                        qty=EXCLUDED.qty,cost_basis=EXCLUDED.cost_basis,entry_ts=EXCLUDED.entry_ts,
                        soft_stop_px=EXCLUDED.soft_stop_px,last_update_at=NOW()""",
                        self.oms_id, order.symbol, order.strategy_id, new_qty, cost, entry_ts, soft_stop)
                    avg = (float(stored_order["avg_fill_price"] or 0) * before + qty * price) / after
                    await db.execute("""UPDATE orders SET filled_qty=$2,avg_fill_price=$3,
                        status=CASE WHEN filled_qty+$4>=qty THEN 'FILLED' ELSE 'PARTIAL' END,last_update_at=NOW()
                        WHERE oms_order_id=$1""", stored_order["oms_order_id"], after, avg, qty)
                    await self._execution_protection(db, order, new_qty, stop_price, stop_mode, identity)
                    await self._execution_trade(db, order, qty, price, fill_ts, realized)
                    encoded = canonical(event)
                    await db.execute("INSERT INTO execution_commits VALUES($1,$2,$3,$4::jsonb)", self.oms_id, identity, signature, encoded)
                    await db.execute("INSERT INTO execution_feedback(oms_id,identity,strategy_id,payload) VALUES($1,$2,$3,$4::jsonb)",
                                     self.oms_id, identity, order.strategy_id, encoded)
            self._record_success()
            return {"duplicate": False, "event": event}
        except Exception:
            self._record_failure()
            raise

    async def _execution_protection(self, db, order, qty, stop_price, mode, identity):
        stop = await db.fetchrow("""SELECT * FROM protective_stops WHERE oms_id=$1 AND strategy_id=$2 AND symbol=$3
            AND status IN ('PENDING','ACTIVE','TRIGGERED','TRIGGERED_PENDING_EXECUTION','EXIT_SUBMITTED') FOR UPDATE""",
            self.oms_id, order.strategy_id, order.symbol)
        if stop:
            await db.execute("""UPDATE protective_stops SET qty=$2,
                status=CASE WHEN $2=0 THEN 'FILLED' ELSE status END,
                stop_price=CASE WHEN status IN ('PENDING','ACTIVE') THEN GREATEST(stop_price,COALESCE($3,stop_price)) ELSE stop_price END,
                source_metadata=source_metadata || $4::jsonb,updated_at=NOW() WHERE stop_id=$1""",
                stop["stop_id"], qty, stop_price, canonical({"last_execution_id": identity, "required_qty": qty}))
        elif order.side == "BUY" and stop_price is not None and qty:
            if mode == "SYNTHETIC_ONLY":
                # An observed execution is an accounting fact even when its
                # protection policy cannot be installed. Recovery halts admission.
                return
            await db.execute("""INSERT INTO protective_stops(stop_id,oms_id,strategy_id,symbol,qty,stop_price,
                protection_mode,status,entry_order_id,entry_intent_id,activated_at,source_metadata)
                VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::uuid,NOW(),$11::jsonb)""",
                uuid.uuid5(uuid.NAMESPACE_URL, f"{self.oms_id}:{identity}:protection"), self.oms_id,
                order.strategy_id, order.symbol, qty, stop_price, mode, "ACTIVE" if mode == "OMS_WATCHER" else "PENDING",
                order.order_id, self._normalize_uuid(order.intent_id), canonical({"execution_id": identity}))

    async def _execution_trade(self, db, order, qty, price, timestamp, realized):
        if order.side == "BUY":
            trade_id = uuid.uuid5(uuid.NAMESPACE_URL, canonical([self.oms_id, order.strategy_id,
                str(order.created_at.date()), order.order_id]))
            # Adopt an existing projection only with a proven entry-intent link.
            # Legacy UUIDs cannot safely be inferred from symbol alone.
            intent_id = self._normalize_uuid(order.intent_id)
            rows = await db.fetch("""SELECT trade_id,entry_intent_id FROM trades
                WHERE oms_id=$1 AND strategy_id=$2 AND symbol=$3
                AND (status='OPEN' OR entry_intent_id=$4::uuid) FOR UPDATE""",
                self.oms_id, order.strategy_id, order.symbol, intent_id)
            if rows:
                if len(rows) != 1 or (rows[0]["trade_id"] != trade_id and
                    (intent_id is None or str(rows[0]["entry_intent_id"]) != str(intent_id))):
                    raise ValueError("Entry execution requires a reconciled trade identity")
                trade_id = rows[0]["trade_id"]
            await db.execute("""INSERT INTO trades(trade_id,oms_id,strategy_id,symbol,direction,entry_qty,entry_price,entry_ts,entry_intent_id,status)
                VALUES($1,$2,$3,$4,'LONG',$5,$6,$7,$8::uuid,'OPEN') ON CONFLICT(trade_id) DO UPDATE SET
                entry_price=(trades.entry_price*trades.entry_qty+EXCLUDED.entry_price*EXCLUDED.entry_qty)/(trades.entry_qty+EXCLUDED.entry_qty),
                entry_qty=trades.entry_qty+EXCLUDED.entry_qty,status='OPEN',closed_at=NULL""",
                trade_id, self.oms_id, order.strategy_id, order.symbol, qty, price, timestamp, self._normalize_uuid(order.intent_id))
        else:
            rows = await db.fetch("SELECT trade_id FROM trades WHERE oms_id=$1 AND strategy_id=$2 AND symbol=$3 AND status='OPEN' FOR UPDATE",
                                  self.oms_id, order.strategy_id, order.symbol)
            if len(rows) != 1:
                raise ValueError("Execution requires one reconciled open trade")
            await db.execute("""UPDATE trades SET exit_price=(COALESCE(exit_price*exit_qty,0)+$2::integer*$3::numeric)/(COALESCE(exit_qty,0)+$2),
                exit_qty=COALESCE(exit_qty,0)+$2,exit_ts=$4,exit_intent_id=$5::uuid,
                realized_pnl_krw=COALESCE(realized_pnl_krw,0)+$6,
                status=CASE WHEN COALESCE(exit_qty,0)+$2>=entry_qty THEN 'CLOSED' ELSE 'OPEN' END,
                closed_at=CASE WHEN COALESCE(exit_qty,0)+$2>=entry_qty THEN NOW() ELSE NULL END
                WHERE trade_id=$1""", rows[0]["trade_id"], qty, price, timestamp, self._normalize_uuid(order.intent_id), realized)

    async def commit_order_transition(self, order, status, reason, payload=None):
        if not self._is_connected():
            raise RuntimeError("Order transition persistence unavailable")
        identity = f"order:{order.created_at.date()}:{order.order_id}:{status.name}:{order.filled_qty}"
        async with self.pool.acquire() as db:
            async with db.transaction():
                await db.execute("SELECT pg_advisory_xact_lock(hashtextextended($1,0))", f"execution:{self.oms_id}")
                row = await self._execution_order(db, order)
                if int(row["filled_qty"]) != order.filled_qty:
                    raise ValueError("Terminal order progress disagrees with canonical executions")
                event = {"kind": "order", "order_id": order.order_id, "symbol": order.symbol, "side": order.side,
                    "qty": order.qty, "status": status.name, "order_type": order.order_type, "reason": reason,
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "metadata": {**dict(payload or {}), "account_id": self.oms_id, "order_date": str(order.created_at.date()),
                        "filled_qty": order.filled_qty, "order_qty": order.qty, "provisional_order_ref": order.submit_ref or ""}}
                await db.execute("UPDATE orders SET status=$2,last_update_at=NOW() WHERE oms_order_id=$1", row["oms_order_id"], status.name)
                await db.execute("""INSERT INTO execution_feedback(oms_id,identity,strategy_id,payload) VALUES($1,$2,$3,$4::jsonb)
                    ON CONFLICT(oms_id,identity) DO NOTHING""", self.oms_id, identity, order.strategy_id, canonical(event))

    async def read_execution_feedback(self, strategy_id, after=0, limit=500):
        if not self._is_connected():
            raise RuntimeError("Execution feedback unavailable")
        rows = await self.pool.fetch("""SELECT sequence,payload FROM execution_feedback
            WHERE oms_id=$1 AND strategy_id=$2 AND sequence>$3 ORDER BY sequence LIMIT $4""",
            self.oms_id, strategy_id.upper(), int(after), min(max(int(limit), 1), 1000))
        return [{"sequence": row["sequence"], **decode(row["payload"])} for row in rows]

    async def execution_snapshot(self, strategy_id, after=0, limit=500):
        if not self._is_connected():
            raise RuntimeError("Execution snapshot unavailable")
        strategy_id = strategy_id.upper()
        await self.pool.execute("""INSERT INTO execution_generations(oms_id,generation) VALUES($1,$2)
            ON CONFLICT(oms_id) DO NOTHING""", self.oms_id, uuid.uuid4())
        async with self.pool.acquire() as db:
            async with db.transaction(isolation="repeatable_read", readonly=True):
                generation = await db.fetchval("SELECT generation FROM execution_generations WHERE oms_id=$1", self.oms_id)
                watermark = await db.fetchval("SELECT COALESCE(MAX(sequence),0) FROM execution_feedback WHERE oms_id=$1 AND strategy_id=$2",
                                              self.oms_id, strategy_id)
                rows = await db.fetch("""SELECT sequence,payload FROM execution_feedback
                    WHERE oms_id=$1 AND strategy_id=$2 AND sequence>$3 AND sequence<=$4 ORDER BY sequence LIMIT $5""",
                    self.oms_id, strategy_id, int(after), watermark, min(max(int(limit), 1), 1000))
                allocations = await db.fetch("SELECT symbol,qty FROM allocations WHERE oms_id=$1 AND strategy_id=$2", self.oms_id, strategy_id)
        events = [{"sequence": row["sequence"], **decode(row["payload"])} for row in rows]
        return {"events": events, "watermark": watermark, "has_more": bool(events and events[-1]["sequence"] < watermark),
                "allocations": {row["symbol"]: row["qty"] for row in allocations if row["qty"]}, "durable": True,
                "stream_id": f"postgres-execution-v2:{self.oms_id}:{strategy_id}:{generation}"}

    async def latest_execution_sequence(self, strategy_id, symbol):
        return await self.pool.fetchval("""SELECT COALESCE(MAX(sequence),0) FROM execution_feedback
            WHERE oms_id=$1 AND strategy_id=$2 AND payload->>'symbol'=$3 AND payload->>'kind'='fill'""",
            self.oms_id, strategy_id, symbol)

    async def execution_schema_ready(self):
        if not self._is_connected():
            return False
        return bool(await self.pool.fetchval("SELECT to_regclass('execution_commits') IS NOT NULL AND to_regclass('execution_feedback') IS NOT NULL AND to_regclass('intent_effects') IS NOT NULL AND to_regclass('execution_generations') IS NOT NULL"))

"""Run the registered recovery queue without duplicating existing research jobs.

This supervisor only dispatches predeclared experiments. A thread heartbeat
reviews their evidence before registering any adaptive follow-up experiments.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def verify(queue):
    root = Path(queue["root"])
    for proof in queue["bound_inputs"]:
        if digest(proof["path"]) != proof["sha256"]:
            raise ValueError("Changed registered input: " + proof["path"])
    for filename in ("preparation.json", "boundary_continuity.json"):
        if read(root / filename).get("passed") is not True:
            raise ValueError("Failed gate: " + filename)
    sessions = read(root / "preparation.json")["sessions"]
    if len(sessions) != 69 or sessions[0] != "2026-04-01" or sessions[-1] != "2026-07-10":
        raise ValueError("Wrong full OOS window")
    for scenario in ("baseline", "participation", "cost", "combined"):
        r = read(root / "control" / scenario / "result.json")
        if (r["audit_status"] != "passed" or not r["execution_validity_pass"]
                or r["residual_sessions"] or r["terminal_inventory"] or r["terminal_pending"]):
            raise ValueError("Unqualified control: " + scenario)
    if "8 passed" not in (root / "tests.log").read_text(encoding="utf-8"):
        raise ValueError("Timing test evidence missing")


def diagnostics(root, name):
    return list((root / name / "framework").glob("*/round_1/round_final_full_diagnostics.json"))


def receipt_complete(path):
    try:
        return read(path).get("complete") is True
    except (FileNotFoundError, json.JSONDecodeError):
        # The existing producer writes a small completion receipt in place.
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    queue = read(a.queue)
    verify(queue)
    if a.check_only:
        print(json.dumps({"gates_passed": True, "queued_rounds": [t["name"] for t in queue["tasks"]]}))
        return
    import msvcrt
    root = Path(queue["root"])
    lock = (root / "background.lock").open("a+b")
    lock.seek(0)
    if not lock.read(1):
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        raise RuntimeError("Another recovery supervisor holds the lock") from exc
    state = {"queue": str(a.queue.resolve()), "pid": os.getpid(),
             "process_created": psutil.Process().create_time(), "completed": [], "child": None}
    last_event = None

    def status(stage, **details):
        nonlocal last_event
        state.update(stage=stage, updated_at_utc=datetime.now(timezone.utc).isoformat(), **details)
        write(root / queue.get("status_file", "background_status.json"), state)
        event = json.dumps({"stage": stage, **details}, sort_keys=True)
        if event != last_event:
            print(event, flush=True)
            last_event = event

    def stopped():
        if (root / "background_hold.json").exists():
            raise RuntimeError("Explicit background hold receipt present")

    def wait_for(description, predicate):
        start = time.monotonic()
        while not predicate():
            stopped()
            status("waiting", waiting_for=description)
            if time.monotonic() - start > 48 * 3600:
                raise TimeoutError("Dependency did not finish in 48 hours: " + description)
            time.sleep(30)

    def command(args, logfile):
        stopped()
        verify(queue)
        # An interrupted supervisor must never launch a duplicate phase runner.
        token = args[args.index("--round-name") + 1] if "--round-name" in args else None
        if token:
            for process in psutil.process_iter(["pid", "cmdline"]):
                cmd = process.info["cmdline"] or []
                if process.pid != os.getpid() and token in cmd and any("kalcb_recovery_experiment_round.py" in c for c in cmd):
                    raise RuntimeError("Round already running in PID " + str(process.pid))
        with logfile.open("x", encoding="utf-8") as output:
            child = subprocess.Popen([sys.executable, "-B", "-u", *args], cwd=REPO,
                                     stdout=output, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            status("running", child={"pid": child.pid, "process_created": psutil.Process(child.pid).create_time(),
                                     "argv": args, "log": str(logfile)})
            started = time.monotonic()
            while child.poll() is None:
                status("running")
                if time.monotonic() - started > 48 * 3600:
                    raise TimeoutError("Child remains active after 48 hours; preserved for investigation")
                time.sleep(30)
            code = child.returncode
        status("child_finished", exit_code=code, child=None)
        if code:
            raise RuntimeError("Native research command failed; inspect " + str(logfile))

    try:
        status("gates_passed")
        # Finish the existing ablations/admission work before adding replay workers.
        for name in queue["prerequisite_rounds"]:
            wait_for("completed diagnostics for " + name, lambda n=name: bool(diagnostics(root, n)))
        for task in queue["tasks"]:
            name = task["name"]
            if diagnostics(root, name):
                state["completed"].append(name)
                continue
            for receipt in task["receipts"]:
                wait_for(receipt, lambda r=receipt: receipt_complete(r))
            wait_for("registered minimum available RAM", lambda: psutil.virtual_memory().available >= queue.get("min_available_gib",1.5) * 1024**3)
            study = root / name / "study.json"
            if not study.exists():
                command(["scripts/kalcb_recovery_experiment_round.py", "prepare", "--root", str(root),
                         "--round-name", name, "--recipes", task["definitions"], "--workers", str(task.get("workers",2))],
                        root / (name + "_prepare.log"))
            else:
                # Existing registration must be exactly the queue's definitions.
                if read(study)["experiment_definitions"] != read(task["definitions"]):
                    raise ValueError("Existing study differs from registered queue")
            command(["scripts/kalcb_recovery_experiment_round.py", "run", "--root", str(root),
                     "--round-name", name], root / (name + "_run.log"))
            if not diagnostics(root, name):
                raise RuntimeError("Phase runner exited without final diagnostics: " + name)
            state["completed"].append(name)
        status(queue.get("completed_stage", "ready_for_targeted_review"), waiting_for=queue.get("completion_message", "Thread heartbeat to assess all ablations and register targeted experiments"))
    except Exception as exc:
        status("attention_required", error=repr(exc))
        raise
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()


if __name__ == "__main__":
    main()

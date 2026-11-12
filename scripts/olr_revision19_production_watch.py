"""Read-only, bounded observation of the released study; no scheduling changes."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import time
import psutil

r = Path('E:/CodexResearch/k_stock_trader/olr_round1_comprehensive_20261009/full_oos_20260401_20260710/continuous_history_v10')
v = r/'revisions/revision_19_native_overheads'
launch = json.loads((r/'revision19_launch_verification.json').read_text(encoding='utf-8-sig'))
supervisor = psutil.Process(launch['supervisor']['ProcessId'])
created = supervisor.create_time()
assert abs(created-datetime.fromisoformat(launch['supervisor']['CreationDate']).timestamp()) < .01
hashes = json.loads((r/'revision19_validation.json').read_text(encoding='utf-8'))['generations']
seen, completions = set(), []
deadline = time.monotonic()+3600
with (v/'production_resource_samples.jsonl').open('a', encoding='utf-8') as stream:
    while time.monotonic() < deadline:
        try:
            if supervisor.create_time() != created or not supervisor.is_running():
                break
            children = []
            for p in supervisor.children(recursive=True):
                try:
                    cpu, mem = p.cpu_times(), p.memory_info()
                    children.append(dict(pid=p.pid, created=p.create_time(), args=p.cmdline(),
                        cpu_seconds=cpu.user+cpu.system, private=mem.private, rss=mem.rss,
                        peak_working_set=mem.peak_wset))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            sample = dict(utc=datetime.now(timezone.utc).isoformat(),
                host_cpu_percent=psutil.cpu_percent(), available_mib=psutil.virtual_memory().available/2**20,
                supervisor_pid=supervisor.pid, children=children)
            stream.write(json.dumps(sample)+'\n'); stream.flush()
            starts = {}
            for line in (r/'mechanism_supervisor_r19.log').read_text(encoding='utf-8-sig').splitlines():
                if not line.startswith('{'):
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get('status') == 'running':
                    starts[(row['candidate'], row['window'])] = row['utc']
            for (cid, window), start in starts.items():
                if (cid, window) in seen:
                    continue
                path = r/'cases'/cid/window/'100000000_baseline/assessment.json'
                if not path.exists():
                    continue
                result = json.loads(path.read_text(encoding='utf-8'))
                if result.get('source_generation_hash') != hashes[window]['source_generation_hash']:
                    continue
                done = json.loads(path.with_name('completion.json').read_text(encoding='utf-8'))
                for name, field in [('result.json','result_sha256'), ('execution_audit.json','audit_sha256')]:
                    with path.with_name(name).open('rb') as f:
                        assert hashlib.file_digest(f, 'sha256').hexdigest() == done[field]
                finished = path.stat().st_mtime
                summary = dict(candidate=cid, window=window, dispatch_utc=start,
                    finished_utc=datetime.fromtimestamp(finished, timezone.utc).isoformat(),
                    total_service_seconds=finished-datetime.fromisoformat(start).timestamp(),
                    qualified=result['qualified'], audit_status=result['audit_status'],
                    native_result_and_audit_hashes_verified=True)
                seen.add((cid, window)); completions.append(summary)
                (v/'production_completions.json').write_text(json.dumps(completions, indent=2)+'\n', encoding='utf-8')
                print(json.dumps(summary), flush=True)
        except psutil.NoSuchProcess:
            break
        time.sleep(10)

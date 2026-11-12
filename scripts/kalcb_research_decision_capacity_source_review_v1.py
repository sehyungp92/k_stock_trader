"""Independent retained-observation check of failed C18 label bars; no replay."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
import sqlite3
import time
import pandas as pd

ROOT = Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1')
OUT = ROOT / 'capacity_source_review_v1'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    started = time.monotonic()
    reg = read(OUT / 'registration.json')
    assert digest(__file__) == reg['program']['sha256']
    for binding in reg['bindings']:
        assert digest(binding['path']) == binding['sha256']
    source_reg = read(reg['minute_registration']['path'])
    rows = list(csv.DictReader((ROOT / 'label_integrity_review_v1/violations.csv').open()))
    plan = read(ROOT / 'native_v1/is/plans/off.json')
    database = Path(plan['database'])
    assert digest(database) == plan['input_sha256'][str(database)]
    conn = sqlite3.connect('file:'+database.as_posix()+'?mode=ro', uri=True)
    results, minute_rows, sources = [], [], {}
    symbols = sorted({row['symbol'] for row in rows})
    for symbol in symbols:
        binding = source_reg['sources'].get(symbol)
        frame = None
        if binding:
            assert digest(binding['path']) == binding['sha256']
            sources[symbol] = binding
            frame = pd.read_parquet(binding['path'])
            frame['artifact_row'] = range(1, len(frame)+1)
        keys = sorted({row['bar'] for row in rows if row['symbol'] == symbol})
        for stamp in keys:
            dt = datetime.fromisoformat(stamp)
            prior = (dt-timedelta(minutes=5)).isoformat()
            current = conn.execute('SELECT symbol,ts,o,h,l,c,v FROM bars WHERE symbol=? AND ts=?', (symbol,stamp)).fetchone()
            previous = conn.execute('SELECT symbol,ts,o,h,l,c,v FROM bars WHERE symbol=? AND ts=?', (symbol,prior)).fetchone()
            assert current is not None
            matching = [row for row in rows if row['symbol']==symbol and row['bar']==stamp]
            assert all(float(row['printed_quantity']) == current[6] for row in matching)
            segment = frame.loc[(frame.timestamp>=dt)&(frame.timestamp<dt+timedelta(minutes=5))] if frame is not None else None
            if segment is not None:
                for rec in segment.to_dict('records'):
                    minute_rows.append({'symbol':symbol,'bar':stamp,'artifact_row':rec['artifact_row'],
                                        'timestamp':str(rec['timestamp']),'volume':rec['volume']})
            results.append({'symbol':symbol,'bar':stamp,'source_5m_volume':current[6],
                'previous_5m_volume':previous[6] if previous else None,
                'native_baseline_budget':int(previous[6]*.01) if previous else 0,
                'native_combined_budget':int(previous[6]*.005) if previous else 0,
                'retained_minute_rows':len(segment) if segment is not None else 0,
                'retained_minute_volume_sum':int(segment.volume.sum()) if segment is not None else None,
                'minute_source_complete_five_rows':bool(segment is not None and len(segment)==5),
                'affected_label_rows':';'.join(row['scenario']+':'+row['source_row'] for row in matching),
                'invalid_label_count':len(matching)})
    conn.close()
    for name,data in [('bars.csv',results),('minute_rows.csv',minute_rows)]:
        with (OUT/name).open('x',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]))
            writer.writeheader();writer.writerows(data)
    with (OUT/'receipt.json').open('x',encoding='utf-8') as handle:
        json.dump({'passed':True,'utc':datetime.now(timezone.utc).isoformat(),'seconds':time.monotonic()-started,
            'native_evaluations':0,'bars':len(results),'minute_sources':sources,
            'database':{'path':str(database),'sha256':plan['input_sha256'][str(database)]},
            'scope':'Source identity/volume comparison only; no future-volume fill cap or changed labels.',
            'unavailable':'Sparse minute observations do not establish queue or broker arrival latency.',
            'files':{p.name:{'path':str(p),'sha256':digest(p)} for p in OUT.iterdir() if p.name!='receipt.json'}},handle,indent=2)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()

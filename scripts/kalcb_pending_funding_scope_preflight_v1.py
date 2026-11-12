"""Read-only full opening correspondence for the newly narrowed core scope."""
import argparse
from collections import defaultdict
from datetime import datetime,timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
REPO=Path(__file__).resolve().parents[1]
BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')


def main(output):
    def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
    def digest(p):
        with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    def bound(p):return {'path':str(Path(p).resolve()),'sha256':digest(p)}
    p=read(BASE/'v23_pending_funding_preflight/protocol.json');code=Path(p['runtime']).resolve();manifest=read(p['runtime_manifest']['path'])
    output.mkdir(parents=True,exist_ok=False)
    sys.path[:]=[str(code)]+[x for x in sys.path if x and not Path(x).resolve().is_relative_to(REPO)]
    spec=importlib.util.spec_from_file_location('backtests.strategies.kalcb._scope_preflight_io',REPO/'backtests/strategies/kalcb/pending_funding_io_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=Path(p['cases'][0]['profile']['manifest']['path']).parent/'files/strategy_kalcb/core/logic.py'
    text=module.patch_core(original.read_text(encoding='utf-8'));patched=output/'scoped_core.py';patched.write_text(text,encoding='utf-8')
    from strategy_kalcb.core import logic
    # Test the explicitly bound new source in the frozen dependency environment.
    exec(compile(text,str(patched),'exec'),logic.__dict__)
    from dataclasses import replace
    from strategy_kalcb.models import KALCBDailySnapshot
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    from strategy_kalcb.core import preview
    preview.step_kalcb_core=logic.step_kalcb_core
    preview_entry=preview.preview_entry
    from strategy_common.market import MarketBar
    from scripts.kalcb_executed_value_study import config_for
    expected={(r['name'],r['period'],r['scenario'],r['day']):r for r in read(BASE/'v23_pending_funding_preflight/prepared/batches.json')}
    count=changed=0
    for case in p['cases']:
        cfg=config_for(read(case['profile']['plan']['path']),case['scenario'])
        snapshots={r['trade_date']:r for r in read(case['snapshot']['path'])};groups=defaultdict(list)
        for r in read(case['evidence']['path'])['entry_exposures']:
            if r['action']['metadata']['entry_route']=='first30_open_anchor':groups[r['decision_available_at']].append(r)
        with sqlite3.connect('file:'+Path(p['databases'][int(case['period']=='oos')]).as_posix()+'?mode=ro',uri=True) as conn:
            for stamp,entries in sorted(groups.items()):
                at=datetime.fromisoformat(stamp);day=str(at.date());snap=KALCBDailySnapshot.from_json_dict(snapshots[day]);symbols={c.symbol for c in snap.candidates}
                bars=[MarketBar(s,datetime.fromisoformat(t),'5m',o,h,l,c,v) for s,t,o,h,l,c,v in conn.execute('SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day=? ORDER BY ts,symbol',(day,)) if s in symbols and datetime.fromisoformat(t)+timedelta(minutes=5)<=at]
                account=KALCBPortfolioView(**entries[0]['account_before']['portfolio'])
                control=preview_entry(bars,snap,cfg,account,available_at=at)
                candidate=preview_entry(bars,snap,replace(cfg,risk_max_gross_equity_pct=2),account,available_at=at)
                if any('pending_funding_reservation' not in r['metadata'] for r in candidate['intents']):raise ValueError('New scoped source was not executed')
                actual={r['action']['symbol']:r['action']['qty'] for r in entries}
                if {r['symbol']:r['quantity'] for r in control['intents']}!=actual:raise ValueError('Disabled control drift')
                prior=expected[(case['name'],case['period'],case['scenario'],day)];wanted=dict(actual)
                for r in prior['changes']:
                    if r['new_qty']:wanted[r['symbol']]=r['new_qty']
                    else:wanted.pop(r['symbol'])
                if {r['symbol']:r['quantity'] for r in candidate['intents']}!=wanted:raise ValueError('Narrowed opening action drift')
                count+=1;changed+=bool(prior['changes'])
    origins=[]
    for name,mod in list(sys.modules.items()):
        file=getattr(mod,'__file__',None)
        if not file or name.split('.')[0] not in ('scripts','backtests','strategy_kalcb','strategy_common','kis_core'):continue
        path=Path(file).resolve()
        if name=='backtests.strategies.kalcb._scope_preflight_io':continue
        if not path.is_relative_to(code) or manifest['files'].get(path.relative_to(code).as_posix())!=digest(path):raise ValueError('Frozen dependency changed '+name)
        origins.append(bound(path))
    value={'complete':True,'batches':count,'changed_batches':changed,'controls_exact':True,'opening_quantities_exact':True,
        'scoped_core_sha256':digest(patched),'tested_core':bound(patched),'original_core':bound(original),
        'programs':[bound(__file__),bound(REPO/'backtests/strategies/kalcb/pending_funding_io_v1.py')],
        'frozen_dependency_origins':origins,'explicit_source_override':'Compiled tested_core replaces logic during this preflight only; no native parent/registered sources edited.',
        'scope':'No fills, future outcomes, funding return or portfolio qualification. Full544 registered opening batches; checkpoint scope tested separately.'}
    (output/'completion.json').write_text(json.dumps(value,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in value.items() if k not in ('frozen_dependency_origins','programs')}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();main(a.output.resolve())

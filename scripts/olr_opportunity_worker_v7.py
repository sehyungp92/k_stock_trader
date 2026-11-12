"""Isolated native feature ledger. It reads no trade or portfolio payoff files."""
from pathlib import Path
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace
import copy
import importlib.util
import json
import os
import pickle
import sys
import tempfile


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    repo = Path(__file__).resolve().parents[1]
    own = load('opportunity_contract', repo / 'scripts/olr_opportunity_v7.py')
    root, out = own.ROOT, own.OUT
    registration = own.read(root / 'registration.json')
    for name, digest in registration['files'].items():
        assert own.sha(repo / name) == digest, name
    for name, digest in registration['inputs'].items():
        assert own.sha(name) == digest, name
    assert not (root / 'ledger_complete.json').exists(), 'Never rerun a completed ledger'
    temporary = root / 'runtime_tmp'
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ['TEMP'] = os.environ['TMP'] = tempfile.tempdir = str(temporary)
    ready = own.read(registration['source']['ready'])
    files = Path(ready['files']).resolve()
    cache = Path(registration['source']['cache'])
    index = own.read(cache / 'index.json')
    h = load('opportunity_cache_identity', repo / 'scripts/olr_snapshot_acceleration.py')
    signature = index.pop('index_signature')
    assert signature == h.signature(index)
    assert index['identity']['base']['source_hash'] == registration['source']['source_hash']
    assert index['identity']['base']['implementation_sha256'] == own.sha(repo / 'scripts/olr_snapshot_acceleration.py')
    assert index['identity']['streaming_sha256'] == own.sha(repo / 'scripts/olr_snapshot_streaming.py')
    assert own.sha(index['parent_receipt']) == index['parent_receipt_sha256']
    parent_receipt = own.read(index['parent_receipt'])
    assert parent_receipt['identity'] == index['identity']['base']
    assert parent_receipt['inputs'] == index['inputs']
    # Authenticate source bytes against the completed generation, even though
    # the cache was created before later additive entry-artifact bindings.
    for name, digest in index['inputs'].items():
        assert ready['manifest']['files'].get(name) == own.sha(files / name) == digest, name
    database = files / ready['study']['database']
    assert own.sha(database) == index['identity']['base']['database_sha256'] == ready['study']['database_hash']
    isolation = load('opportunity_native_isolation', repo / 'scripts/olr_structural_native_test_isolated_v6.py')
    guard = isolation.NativeOnly(files, ready['manifest']['files'])
    sys.meta_path.insert(0, guard)
    sys.path[:] = [str(files)] + [p for p in sys.path if not Path(p or os.getcwd()).resolve().is_relative_to(repo)]
    os.chdir(files)
    from backtests.core.input_manifest import InputManifest
    from backtests.strategies.olr import research_sweep as native
    from strategy_olr import research
    from strategy_olr.config import OLRConfig
    from strategy_common.clock import KST
    from strategy_common.execution_session import session_cutoff
    from strategy_common.sector_intraday import AFTERNOON_CUTOFF
    streaming = load('opportunity_native_panels', repo / 'scripts/olr_snapshot_streaming.py')
    generation = InputManifest(ready['study']['manifest'])
    request = own.read(files / ready['study']['olr_request'])
    candidates = own.read(own.PARENT / 'candidate_plan.json')['candidates']
    mutations = {k:v for k,v in candidates[0]['mutations'].items() if not k.startswith('capacity_')}
    mutations['olr.research.minute_exit'] = False
    request['mutations'] = mutations
    request['config'].update(initial_mutations=copy.deepcopy(mutations), initial_equity=100000000,
                             date_range=ready['study']['configuration']['date_range'])
    cfg = OLRConfig.from_mapping(request['config'], mutations)
    original = {s['trade_date']:s for s in own.read(registration['source']['snapshot'])}
    days = index['identity']['base']['output_dates']
    assert days == sorted(original) and max(days) <= '2026-03-31'

    def payload(name):
        assert own.sha(cache / name) == index['files'][name], name
        with (cache / name).open('rb') as stream:
            return pickle.load(stream)

    static = payload('static.pickle')
    universe = sorted(static['symbols'])
    assert len(universe) == 103 and len(days)*len(universe) <= registration['scope']['max_rows']
    context_days = tuple(date.fromisoformat(d) for d in index['identity']['base']['context_dates'])
    dataset = SimpleNamespace(config=request['config'], trading_dates=context_days,
        sector_map=static['sectors'], sector_intraday_panel_1430=static['panel'])
    rows, daily_checks = [], []
    with generation.activate():
        for day_text in days:
            day = date.fromisoformat(day_text)
            snapshot, daily_panel, intraday_panel, bars = payload(day_text + '.pickle')
            assert snapshot.trade_date == day
            for symbol in snapshot.symbols.values():
                for field in ('daily_rows', 'flow_rows', 'foreign_flow_rows', 'institutional_flow_rows'):
                    for row in getattr(symbol, field, ()):
                        assert research._row_date(row) < day, (day_text, field)
            cutoff = datetime.combine(day, session_cutoff(day, time(14,30)), KST)
            prefix = {key:tuple(b for b in values if b.timestamp + timedelta(minutes=5) <= cutoff)
                      for key,values in bars.items()}
            with streaming.native_panel_cache(research, day, snapshot, daily_panel, intraday_panel,
                                               static['sectors'], AFTERNOON_CUTOFF):
                morning = native.snapshots_for_experiment(dataset, mutations, research_snapshots={day:snapshot})[day]
                contexts = research.build_afternoon_contexts(morning, prefix, cfg, static['sectors'])
                full_contexts = research.build_afternoon_contexts(morning, bars, cfg, static['sectors'])
                assert contexts == full_contexts, 'Future intraday bars changed decision features'
                final = research.afternoon_selection_from_contexts(morning, contexts, cfg).to_json_dict()
            expected = original[day_text]
            assert final['candidates'] == expected['candidates'], 'Raw selected candidate drift: ' + day_text
            for key in ('rejected_symbols', 'afternoon_rejected_symbols', 'candidate_pool_symbols', 'overflow_symbols'):
                assert final['metadata'].get(key) == expected['metadata'].get(key), (day_text, key)
            selected = {c['symbol'] for c in final['candidates']}
            morning_candidates = {c.symbol:c for c in morning.candidates}
            for symbol in universe:
                candidate = morning_candidates.get(symbol)
                ctx = contexts.get(symbol)
                hard = research._afternoon_reject_reasons(ctx, cfg) if ctx is not None else []
                reasons = final['metadata']['afternoon_rejected_symbols'].get(symbol, [])
                role = own.cohort(symbol in selected, candidate is not None, ctx is not None, hard, reasons)
                score, raw, exhaustion = research._afternoon_score_details(ctx, cfg) if ctx is not None else (None,None,None)
                features = research._afternoon_rule_feature_values(ctx,score,exhaustion) if ctx is not None else None
                rows.append(dict(day=day_text, symbol=symbol, cohort=role, available_at=cutoff.isoformat(),
                    morning_selected=candidate is not None, selected=symbol in selected,
                    daily_rejections=morning.metadata.get('rejected_symbols',{}).get(symbol,[]),
                    daily_overflow=symbol in morning.metadata.get('overflow_symbols',[]),
                    source_prefix_excluded=symbol in expected['metadata'].get('feature_prefix_exclusions',[]),
                    afternoon_hard_rejections=hard, afternoon_final_rejections=reasons,
                    score=score, raw_score=raw, exhaustion=exhaustion, features=features,
                    candidate=candidate.to_json_dict() if candidate is not None else None))
            daily_checks.append(dict(day=day_text, raw_selected_exact=True, rejection_metadata_exact=True,
                                     future_intraday_invariance=True, cached_payload_sha256=index['files'][day_text+'.pickle']))
            print(json.dumps(dict(day=day_text, rows=len(rows), selected=len(selected))), flush=True)
    imports = {}
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] not in guard.roots or not getattr(module,'__file__',None):
            continue
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(files), name
        relative = path.relative_to(files).as_posix()
        assert ready['manifest']['files'].get(relative) == own.sha(path), name
        imports[relative] = own.sha(path)
    assert len(rows) == len(days)*103 and len({(r['day'],r['symbol']) for r in rows}) == len(rows)
    summary = own.summarize(rows, days)
    own.write(root / 'opportunities_train.json', rows)
    own.write(out / 'supply_summary.json', summary)
    own.write(root / 'ledger_complete.json', dict(utc=own.utc(), registration_sha256=own.sha(root/'registration.json'),
        ledger_sha256=own.sha(root/'opportunities_train.json'), summary_sha256=own.sha(out/'supply_summary.json'),
        imports=imports, daily_checks=daily_checks, source_files_authenticated=len(index['inputs']),
        no_performance_files_read=True, promotion=False))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

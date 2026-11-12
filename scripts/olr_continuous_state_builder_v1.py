"""Intentional current-panel feature treatment over authenticated native caches.

This is a model/data-flow change, not a parity-preserving accelerator. All
daily research and full-history panels retain their original source receipts.
"""
from contextlib import contextmanager
from datetime import date
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import time


def build(manifest, database, request, days, *, output_days, cache_root, source_hash, profile):
    helper_path=Path(__file__).with_name('olr_snapshot_streaming.py')
    helper_spec=importlib.util.spec_from_file_location('olr_panel_native_streaming',helper_path)
    streaming=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(streaming)
    helper,read_payload,native_panel_cache=streaming.helper,streaming.read_payload,streaming.native_panel_cache
    from backtests.core.input_manifest import InputManifest
    from backtests.strategies.olr import research_sweep as native
    from strategy_olr import research
    from strategy_olr.config import OLRConfig
    from strategy_common.sector_intraday import AFTERNOON_CUTOFF
    from strategy_common.source_quality import POLICY
    h=helper();start=time.perf_counter();generation=InputManifest(manifest)
    descriptor=json.loads((Path(cache_root)/'source.json').read_text())
    directory=Path(descriptor['cache']);index_path=directory/'index.json'
    assert h.digest(index_path)==descriptor['index_sha256']
    index=json.loads(index_path.read_text());signed=index.pop('index_signature')
    assert h.signature(index)==signed
    base=index['identity']['base']
    assert base['database_sha256']==h.digest(database)
    assert base['context_dates']==list(map(str,days)) and base['output_dates']==list(map(str,output_days))
    for name,expected in index['inputs'].items():
        assert generation.digest(generation.root/name)==expected
        assert h.digest(generation.root/name)==expected
    static=read_payload(directory/'static.pickle',index['files']['static.pickle'],h.digest)
    dataset=SimpleNamespace(config=request['config'],trading_dates=tuple(days),sector_map=static['sectors'],sector_intraday_panel_1430=static['panel'])
    cfg=OLRConfig.from_mapping(request['config'],request['mutations']);result={};bridge=[]
    for day in output_days:
        name=day.isoformat()+'.pickle'
        snapshot,daily_panel,same_day_panel,bars=read_payload(directory/name,index['files'][name],h.digest)
        assert snapshot.trade_date==day
        with native_panel_cache(research,day,snapshot,daily_panel,same_day_panel,static['sectors'],AFTERNOON_CUTOFF):
            morning=native.snapshots_for_experiment(dataset,request['mutations'],research_snapshots={day:snapshot})
            result[day]=research.afternoon_selection_from_snapshot(morning[day],bars,cfg,static['sectors'],sector_intraday_panel=static['panel'] if cfg.research_continuous_intraday_panel else None)
        if cfg.require_complete_feature_prefix:
            excluded=[x.split(':')[0] for x in static['exclusions'] if x.endswith(':'+day.isoformat())]
            assert not set(excluded)&{c.symbol for c in result[day].candidates}
            result[day].metadata.update(feature_prefix_policy=POLICY,feature_prefix_exclusions=excluded)
        result[day].metadata['current_sector_history_policy']='continuous_prior_20_complete_positive_sessions' if cfg.research_continuous_intraday_panel else 'legacy_same_session'
        result[day].metadata['current_sector_history_builder_sha256']=h.digest(__file__)
    assert set(result)==set(output_days)
    profile.update(feature_treatment='continuous_current_afternoon_panel_v1',native_invariant_cache=str(directory),
        native_invariant_index_sha256=h.digest(index_path),source_generation_hash=source_hash,
        context_sessions=len(days),output_sessions=len(output_days),total_seconds=time.perf_counter()-start)
    print(json.dumps(profile),flush=True)
    return result

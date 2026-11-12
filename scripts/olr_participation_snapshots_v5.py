"""Native two-pass selector on authenticated cached source features."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time


def build(manifest, database, request, days, *, output_days, cache_root, source_hash, profile):
    path=Path(__file__).with_name('olr_snapshot_streaming.py')
    spec=importlib.util.spec_from_file_location('olr_participation_streaming',path)
    streaming=importlib.util.module_from_spec(spec); spec.loader.exec_module(streaming)
    h=streaming.helper()
    from backtests.core.input_manifest import InputManifest
    from backtests.strategies.olr import research_sweep as native
    from strategy_olr import research
    from strategy_olr.config import OLRConfig
    from strategy_olr.participation_rescue_v5 import select
    from strategy_common.sector_intraday import AFTERNOON_CUTOFF
    from strategy_common.source_quality import POLICY
    start=time.perf_counter(); generation=InputManifest(manifest)
    descriptor=json.loads((Path(cache_root)/'source.json').read_text())
    directory=Path(descriptor['cache']); index_path=directory/'index.json'
    assert h.digest(index_path)==descriptor['index_sha256']
    assert h.digest(descriptor['source_receipt'])==descriptor['source_receipt_sha256']
    index=json.loads(index_path.read_text()); signed=index.pop('index_signature')
    assert h.signature(index)==signed
    base=index['identity']['base']
    assert base['database_sha256']==h.digest(database)==generation.digest(database)
    assert base['context_dates']==list(map(str,days)) and base['output_dates']==list(map(str,output_days))
    for name,expected in index['inputs'].items():
        assert generation.digest(generation.root/name)==expected==h.digest(generation.root/name)
    payload=lambda name: streaming.read_payload(directory/name,index['files'][name],h.digest)
    static=payload('static.pickle')
    dataset=SimpleNamespace(config=request['config'],trading_dates=tuple(days),sector_map=static['sectors'],sector_intraday_panel_1430=static['panel'])
    cfg=OLRConfig.from_mapping(request['config'],request['mutations'])
    assert cfg.research_empty_participation_rescue and not cfg.research_continuous_intraday_panel
    base_mutations={**request['mutations'],'olr.research.empty_participation_rescue':False}
    alternate_mutations={**base_mutations,'olr.research.min_sector_participation':0.0}
    base_cfg=OLRConfig.from_mapping(request['config'],base_mutations)
    alternate_cfg=OLRConfig.from_mapping(request['config'],alternate_mutations)
    result={}; additions=[]
    for day in output_days:
        snapshot,daily_panel,same_day_panel,bars=payload(day.isoformat()+'.pickle')
        assert snapshot.trade_date==day
        with streaming.native_panel_cache(research,day,snapshot,daily_panel,same_day_panel,static['sectors'],AFTERNOON_CUTOFF):
            morning=native.snapshots_for_experiment(dataset,base_mutations,research_snapshots={day:snapshot})[day]
            original=research.afternoon_selection_from_snapshot(morning,bars,base_cfg,static['sectors'],sector_intraday_panel=None)
            if original.candidates:
                alternate=original
            else:
                morning=native.snapshots_for_experiment(dataset,alternate_mutations,research_snapshots={day:snapshot})[day]
                alternate=research.afternoon_selection_from_snapshot(morning,bars,alternate_cfg,static['sectors'],sector_intraday_panel=None)
            result[day]=select(original,alternate,cfg)
        if cfg.require_complete_feature_prefix:
            excluded=[x.split(':')[0] for x in static['exclusions'] if x.endswith(':'+day.isoformat())]
            assert not set(excluded)&{c.symbol for c in result[day].candidates}
            result[day].metadata.update(feature_prefix_policy=POLICY,feature_prefix_exclusions=excluded)
        result[day].metadata.update(current_sector_history_policy='legacy_same_session',
            current_sector_history_builder_sha256=h.digest(__file__))
        if not original.candidates:
            additions.extend({'day':str(day),'symbol':c.symbol} for c in result[day].candidates)
    assert set(result)==set(output_days)
    profile.update(native_invariant_cache=str(directory),native_invariant_index_sha256=h.digest(index_path),
        source_generation_hash=source_hash,context_sessions=len(days),output_sessions=len(output_days),
        total_seconds=time.perf_counter()-start,added_candidates=additions,
        future_returns_consumed=False,incumbent_selection_unchanged=True)
    print(json.dumps(profile),flush=True)
    return result

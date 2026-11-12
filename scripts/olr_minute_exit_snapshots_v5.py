"""Fresh native snapshots for the corrected source generation; no P&L input."""
import copy,importlib.util
from pathlib import Path


def entry_request(request):
    """Keep execution-only exit mode out of executable entry artifact identity."""
    result=copy.deepcopy(request)
    result['mutations']['olr.research.minute_exit']=False
    result['config']['initial_mutations']=copy.deepcopy(result['mutations'])
    return result


def build(manifest,database,request,days,*,output_days,cache_root,source_hash,profile):
    path=Path(__file__).with_name('olr_snapshot_streaming.py')
    spec=importlib.util.spec_from_file_location('minute_native_streaming',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from strategy_olr.config import OLRConfig
    request=entry_request(request)
    cfg=OLRConfig.from_mapping(request['config'],request['mutations'])
    if cfg.research_continuous_intraday_panel:raise ValueError('Minute study preserves the retained legacy entry panel')
    result=module.build(manifest,database,request,days,output_days=output_days,
        cache_root=cache_root,source_hash=source_hash,profile=profile)
    for snap in result.values():
        snap.metadata['current_sector_history_policy']='legacy_same_session'
        snap.metadata['current_sector_history_builder_sha256']=module.helper().digest(__file__)
    return result

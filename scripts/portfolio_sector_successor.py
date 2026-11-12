"""Preserve the pre-return sector-check attempt and freeze its narrow correction."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.portfolio_source_repair import ROOT
from scripts.portfolio_targeted_validation import read, bound
from backtests.strategies.kalcb.executed_value import digest, signature, write_json


def build():
    old = ROOT / 'sector_complete'
    new = ROOT / 'sector_complete_v2'
    if (old / 'validation').exists(): raise ValueError('Only a pre-replay mechanical correction is allowed')
    manifest = read(old / 'forward_inputs/manifest.json')
    new.mkdir(exist_ok=False)
    for name in ('protocol.json', 'screen.json', 'bridge.json', 'source_audit.json', 'development_selection.json',
                 'frozen_inference_bridge.json', 'olr_inference_bridge.json', 'development/study.json', 'battery/study.json',
                 'repair_registration.json', 'flow_basis_audit.json', 'volume_revision_audit.json',
                 'lookback_overlap_probe.json', 'lookback_overlap_verified.json', 'inference_import_audit.json',
                 'software_tests.xml', 'forward_dependency_tests.xml', 'sector_dependency_tests.xml', 'real_merge_regression.json'):
        destination = new / name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old / name, destination)
    shutil.copyfile(old / 'prior_forward_program_dependency_review.json', new / 'forward_program_dependency_review.json')
    for name in ('repaired_sessions', 'minute_sources', 'flow_sources'):
        for source in (old / name).rglob('*'):
            if source.is_file():
                target = new / source.relative_to(old); target.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, target)
    helper = 'scripts/kalcb_executed_value_study.py'
    target = new / 'forward_inputs/files'; target.mkdir(parents=True)
    files = {}
    for name, sha in manifest['files'].items():
        source = old / 'forward_inputs/files' / name
        if digest(source) != sha: raise ValueError('Changed predecessor generation ' + name)
        destination = target / name; destination.parent.mkdir(parents=True, exist_ok=True)
        if name == helper: shutil.copyfile(REPO / name, destination)
        else: os.link(source, destination)
        files[name] = digest(destination)
    if {n for n in files if files[n] != manifest['files'][n]} != {helper}:
        raise ValueError('Correction must change only the sector coverage assertion')
    old_source = (old / 'forward_inputs/files' / helper).read_text(encoding='utf-8')
    new_source = (target / helper).read_text(encoding='utf-8')
    before = "missing_sectors = [s for s in context_symbols if"
    after = "missing_sectors = [s for s in universe if"
    if old_source.count(before) != 1 or old_source.replace(before, after) != new_source:
        raise ValueError('Unexpected strategy-helper changes')
    command = 'from kis_core.trading_calendar import get_trading_calendar;get_trading_calendar();from backtests.auto.shared.cache_keys import source_generation_hash;print(source_generation_hash())'
    source_hash = subprocess.check_output([sys.executable, '-B', '-c', command], cwd=target, text=True).strip()
    payload = {**{k:v for k,v in manifest.items() if k != 'manifest_hash'}, 'files': files, 'source_generation_hash': source_hash}
    write_json(new / 'forward_inputs/manifest.json', {**payload, 'manifest_hash': signature(payload)})
    repair = read(old / 'repair_generation.json')
    repair['sector_preflight_predecessor'] = bound(old / 'forward_inputs/manifest.json')
    repair['new_manifest'] = bound(new / 'forward_inputs/manifest.json')
    repair['narrow_assertion_correction'] = {'before': before, 'after': after, 'strategy_formulas_changed': False}
    write_json(new / 'repair_generation.json', repair)
    sector = read(old / 'sector_dependency_repair.json')
    sector['restored'] = bound(target / 'data/krx_daily_parquet/tables/sector_map.parquet')
    sector['historical_bridge_assertion_scope'] = 'Require known sectors for all traded-universe stocks; retain original unknown auxiliary context symbols'
    sector['prior_bridge_attempt'] = bound(old / 'inference_feature-bridge.log')
    sector['successor_program'] = bound(Path(__file__))
    write_json(new / 'sector_dependency_repair.json', sector)
    overlap = read(old / 'lookback_development_verified.json')
    if not overlap['pass'] or overlap['repaired_manifest'] != bound(old / 'forward_inputs/manifest.json'):
        raise ValueError('Unqualified predecessor overlap check')
    for item in overlap['inputs']:
        previous = Path(item['repaired']['path'])
        name = previous.relative_to(old / 'forward_inputs/files').as_posix()
        if item['repaired']['sha256'] != files[name] or files[name] != manifest['files'][name]:
            raise ValueError('Cannot reuse changed market-data overlap proof')
        item['repaired'] = bound(target / name)
    overlap['repaired_manifest'] = bound(new / 'forward_inputs/manifest.json')
    overlap['reuse_basis'] = 'Every compared market input is byte-identical; only a traded-universe coverage assertion changed'
    overlap['prior_completed_audit'] = bound(old / 'lookback_development_verified.json')
    write_json(new / 'lookback_development_verified.json', overlap)
    print(json.dumps({'successor': str(new), 'source_generation_hash': source_hash, 'market_bytes_unchanged': True}), flush=True)


if __name__ == '__main__': build()

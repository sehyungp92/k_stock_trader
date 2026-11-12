"""Register paired continuation execution experiments without changing production.

The supplied policy fixes candidate admission and quantity metadata. These are
execution ablations, not newly fitted policies: a successful change must receive
new action labels and past-only model fits before a value-policy promotion.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev
from scripts.kalcb_executed_value_study import load_plan


def variants(original):
    acceptance = deepcopy(original)
    acceptance.update(
        exit_failed_followthrough_bars=0,
        quick_exit_enabled=False,
        exit_vwap_fail_bars=3,
        exit_vwap_fail_pct=.002,
        exit_vwap_fail_after_mfe_r=0,
    )
    confirmed = deepcopy(original)
    confirmed['entry_window_start'] = '09:00:00'
    anchors = 0
    for route in confirmed['entry_plan_routes']:
        if route['mode'] == 'first30_open':
            anchors += 1
            route.update(name='completed_post_open_confirmation', mode='post_or_momentum',
                         after_bar=0, max_signal_bars=3, max_session_trades=1,
                         min_bar_ret=0, min_vwap_ret=0,
                         min_or_position=.65, min_close_location=.6)
        elif route['mode'] == 'pullback_acceptance':
            # Preserve its regular-session earliest 10:25 signal after moving
            # the shared entry window to permit the confirmation route.
            route['after_bar'] = max(route.get('after_bar', 0), 11)
    if anchors != 1:
        raise ValueError('Expected one opening anchor')
    modes = list(confirmed.get('exit_path_quality_entry_route_modes') or [])
    if 'first30_open' in modes and 'post_or_momentum' not in modes:
        modes.append('post_or_momentum')
    confirmed['exit_path_quality_entry_route_modes'] = modes
    return {'acceptance_exit': acceptance, 'confirmed_entry': confirmed}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    parent = load_plan(a.plan)
    original = json.loads(Path(parent['config']).read_text())
    from strategy_kalcb.config import KALCBConfig
    generated = variants(original)
    for config in generated.values():
        KALCBConfig.from_mapping(config)
    a.output.mkdir(parents=True, exist_ok=False)
    for name, config in generated.items():
        cfg_path = (a.output / (name + '.config.json')).resolve()
        ev.write_json(cfg_path, config)
        plan = deepcopy(parent)
        plan['input_sha256'].pop(parent['config'])
        plan['config'] = str(cfg_path)
        plan['input_sha256'][str(cfg_path)] = ev.digest(cfg_path)
        plan['execution_ablation'] = name
        ev.write_json(a.output / (name + '.plan.json'), plan)
    ev.write_json(a.output / 'protocol.json', {
        'parent_plan_sha256': ev.digest(a.plan),
        'variants': list(generated),
        'selection': 'Declared before paired execution outcomes',
        'comparison': 'Identical candidate policy, quantity reductions, capital, costs and dates',
        'acceptance_exit': 'Three completed closes 20bp below session VWAP replace the two time-based exits; other exits retained',
        'confirmed_entry': 'First three completed post-opening bars; positive bar, above VWAP, at least 65% of opening range and 60% of signal bar',
        'limitation': 'Old value/risk forecasts describe old execution; this is a paired ablation only. Regenerate labels before refitting.',
        'production_promotion': False,
    })


if __name__ == '__main__':
    main()

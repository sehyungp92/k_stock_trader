"""Research-only decision value contracts; no fitting, policy or broker mutation.

Keep calibration value, ranking score, exact action and funded outcome separate.
An isolated action's return is never scaled into a portfolio counterfactual.
"""
from dataclasses import dataclass
from datetime import datetime
import math

SCENARIOS = ('baseline', 'combined')
SCALED_FEATURES = ('action_notional_equity', 'action_quantity_last_volume',
                   'action_quantity_opening_volume', 'action_low_volume_exit_slices')


def finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Nonfinite decision value')
    return result


def aware(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError('Timestamp must include availability timezone')
    return result


def action_features(row, full_quantity):
    """Reconstruct the parent's exact integer quantity transform, not q*x."""
    q = finite(row['quantity_scale'])
    if q not in (.25, .5, 1.) or not isinstance(full_quantity, int) or full_quantity <= 0:
        raise ValueError('Unregistered quantity')
    quantity = math.floor(full_quantity * q)
    if quantity <= 0 or quantity != row['reference_quantity']:
        raise ValueError('Reference quantity mismatch')
    features = dict(row['features'])
    if features['action_quantity_scale'] != 1.:
        raise ValueError('Expected unscaled source features')
    ratio = quantity / full_quantity
    for key in SCALED_FEATURES:
        features[key] = finite(features[key]) * ratio
    features['action_quantity_scale'] = q
    if not math.isclose(features['action_notional_equity'], row['notional_equity'],
                        rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Integer notional mismatch')
    return features


@dataclass(frozen=True)
class DecisionValue:
    action_id: str
    origin: str
    quantity: int
    scale: float
    notional_equity: float
    available_at: str
    base_valid_from: str
    calibration_valid_from: str
    core: dict
    final: dict
    supported: bool
    retained: bool
    selection_score: float

    def notional_bps(self, field, scenario):
        if field not in ('core', 'final') or scenario not in SCENARIOS:
            raise ValueError('Unknown value field')
        return finite(getattr(self, field)[scenario]) / self.notional_equity


def reconcile_value(row, counterpart, origin, proof, selection_score):
    """Reproduce the existing parent veto; this function chooses no new action."""
    if origin not in ('frozen_core', 'new_forecast'):
        raise ValueError('Unknown origin')
    day = row['day']
    if counterpart['action_id'] != row['action_id'] or counterpart['day'] != day:
        raise ValueError('Counterpart action mismatch')
    clock = aware(row['available_at'])
    if clock.date().isoformat() != day or aware(counterpart['available_at']) > clock:
        raise ValueError('Future counterpart')
    if row['base_valid_from'] > day or counterpart['valid_from'] > day:
        raise ValueError('Future model')
    if proof['family'] != 'expanding' or not proof['valid_from'] <= day <= proof['valid_through']:
        raise ValueError('Calibration clock mismatch')
    if proof['observed_through'] >= proof['valid_from']:
        raise ValueError('Calibration embargo missing')
    latest = proof['latest_label_available_at']
    if latest and aware(latest) >= aware(proof['valid_from'] + 'T00:00:00+09:00'):
        raise ValueError('Immature calibration')
    field = 'calibrated_core' if origin == 'frozen_core' else 'calibrated_support'
    core = {s: finite(row[field][s]) for s in SCENARIOS}
    own = int(min(core.values()) > 0)
    other = int(min(finite(counterpart[field][s]) for s in SCENARIOS) > 0)
    q = finite(row['quantity_scale'])
    tier = row['descriptor']['tier']
    if q not in (.25, .5, 1.) or tier not in ('top2', 'rest'):
        raise ValueError('Unknown cohort')
    key = f'{origin}/q{q:g}/self{own}_other{other}/{tier}'
    cells = {s: proof['cohorts'][s + '/' + key] for s in SCENARIOS}
    supported = all(c['supported'] for c in cells.values())
    for c in cells.values():
        if not c['supported'] and c['offset_bps'] != 0:
            raise ValueError('Unsupported map must be identity')
    final = {s: core[s] + finite(cells[s]['offset_bps']) for s in SCENARIOS}
    notional = finite(row['notional_equity'])
    if notional <= 0 or row['reference_quantity'] <= 0:
        raise ValueError('Invalid action denominator')
    return DecisionValue(row['action_id'], origin, row['reference_quantity'], q, notional,
                         row['available_at'], row['base_valid_from'], proof['valid_from'],
                         core, final, supported, not (supported and min(final.values()) <= 0),
                         finite(selection_score))


def unique_position_links(row, parent, scenario, seen):
    """Attach a lifetime once, only to the parent's selected quantity action."""
    positions = row['actual'].get(parent + '/' + scenario, [])
    if parent not in row['selected']:
        return []
    if row['selected'][parent]['action_id'] != row['action_id']:
        raise ValueError('Selection identity mismatch')
    result = []
    for position in positions:
        identity = (parent, row['period'], scenario, position['position_id'])
        if identity in seen:
            raise ValueError('Duplicate lifetime PnL join')
        seen.add(identity)
        if position['route'] != 'first30_open_anchor' or position['entry_day'] != row['day']:
            raise ValueError('Lifetime outside opening action')
        result.append({**position,
                       'reference_quantity_equal': position['qty'] == row['reference_quantity'],
                       'isolated_to_funded_exchangeability_proven': False})
    return result

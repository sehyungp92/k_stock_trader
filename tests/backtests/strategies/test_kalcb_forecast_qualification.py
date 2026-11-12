from copy import deepcopy
import pytest
from backtests.strategies.kalcb.executed_value import signature
from scripts.kalcb_recovery_forecast_qualification_v1 import validate_training_proof


def fixture():
    rows = [{'trade_date': '2026-03-04', 'opportunity_id': 'a'},
            {'trade_date': '2026-03-05', 'opportunity_id': 'b'}]
    targets, weights = [2.0, -3.0], [.5, 1.]
    proof = {'valid_from': '2026-03-10', 'train_from': '2026-03-04',
             'train_through': '2026-03-05', 'train_rows': 2,
             'training_keys_hash': signature(['a','b']),
             'training_target_hash': signature(targets),
             'training_weights_hash': signature(weights)}
    return proof, rows, targets, weights


def test_exact_frozen_training_population_is_accepted_without_mutation():
    inputs = fixture()
    before = deepcopy(inputs)
    validate_training_proof(*inputs)
    assert inputs == before


@pytest.mark.parametrize('mutation', ['future', 'order', 'target', 'weights', 'missing_weights', 'extent'])
def test_wrong_training_population_or_target_is_rejected(mutation):
    proof, rows, targets, weights = fixture()
    if mutation == 'future': rows[-1]['trade_date'] = '2026-03-10'
    elif mutation == 'order': rows.reverse()
    elif mutation == 'target': targets[-1] = 30.
    elif mutation == 'weights': weights[0] = 1.
    elif mutation == 'missing_weights': weights = None
    else: proof['train_from'] = '2026-03-03'
    with pytest.raises(ValueError):
        validate_training_proof(proof, rows, targets, weights)

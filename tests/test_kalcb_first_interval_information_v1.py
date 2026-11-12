from copy import deepcopy
import numpy as np
import pytest
from scripts.kalcb_first_interval_information_v1 import features, train, predict


def test_first_interval_future_prefix_is_rejected():
    bars=[[f'2025-08-08T09:{i*5:02d}:00+09:00',100,102,99,101,1000] for i in range(7)]
    row={'raw_prefix':bars,'completion_at':'2025-08-08T09:35:00+09:00'}
    assert len(features(row,{'daily_atr':3,'first30_rel_volume':2}))==12
    bad=deepcopy(row);bad['raw_prefix'][-1][0]='2025-08-08T09:35:00+09:00'
    with pytest.raises(ValueError,match='Future'):
        features(bad,{'daily_atr':3,'first30_rel_volume':2})


def test_training_only_degrees_of_freedom_matching_and_no_target_in_prediction():
    rng=np.random.default_rng(17)
    rows=[dict(day=f'd{i//2:02d}',features=rng.normal(size=12).tolist(),target_bps=float(i%7)) for i in range(80)]
    models=train(rows)
    assert abs(models['N']['effective_df']-models['B']['effective_df'])<1e-9
    a=rows[0];b={**a,'target_bps':1000000}
    assert predict(models['N'],a)==predict(models['N'],b)

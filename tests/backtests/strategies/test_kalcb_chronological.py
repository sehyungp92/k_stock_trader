from datetime import date, timedelta

import pytest

from backtests.strategies.kalcb.chronological import past_only_folds, validate_predictor_columns


def test_every_training_label_precedes_validation_and_sessions_stay_together():
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(15) for _ in range(2)]
    ends = [d + timedelta(days=2) for d in days]
    folds = past_only_folds(days, min_train_sessions=4, validation_sessions=3, embargo_sessions=1, label_end_dates=ends)
    for f in folds:
        assert max(ends[i] for i in f.train_indices) < min(days[i] for i in f.valid_indices)
        assert not set(f.train_indices) & set(f.valid_indices)
        for i in f.valid_indices:
            assert i ^ 1 in f.valid_indices
    # Appending future observations cannot alter already-complete earlier folds.
    extended = days + [date(2025, 1, 25)] * 2
    more = past_only_folds(extended, min_train_sessions=4, validation_sessions=3, embargo_sessions=1,
        label_end_dates=ends + [date(2025, 1, 25)] * 2)
    assert more[:len(folds) - 1] == folds[:-1]


def test_warmup_is_not_filled_with_future_fitted_predictions():
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(10)]
    folds = past_only_folds(days, min_train_sessions=4, validation_sessions=2, embargo_sessions=1)
    assert not set(range(5)) & {i for f in folds for i in f.valid_indices}
    assert folds[0].train_indices == (0, 1, 2, 3)
    assert folds[0].valid_indices == (5, 6)


def test_outcome_features_are_rejected():
    validate_predictor_columns(["first30_ret", "daily_return_20d"])
    with pytest.raises(ValueError, match="entered_shared_core"):
        validate_predictor_columns(["first30_ret", "entered_shared_core"])


def test_oof_fit_does_not_train_on_future_outcomes_or_impute_warmup(monkeypatch):
    import numpy as np
    import pandas as pd
    from scripts import kalcb_oof_interaction_route_optimizer as runner

    class MeanModel:
        def fit(self, x, y):
            self.mean = np.mean(y)
        def predict(self, x):
            return np.full(len(x), self.mean)

    days = pd.date_range("2025-01-01", periods=85).strftime("%Y-%m-%d")
    frame = pd.DataFrame({"trade_date": days, "first30_ret": range(85),
        "positive_r": np.arange(85, dtype=float), "net_eod_r_proxy": np.arange(85, dtype=float),
        "dataset_available": 1.0})
    spec = runner.OOFRankerSpec("test", "base_all_context", "ridge", "alpha_quality", "dataset", "top16", 16, 16)
    monkeypatch.setattr(runner, "oof_ranker_specs", lambda: [spec])
    monkeypatch.setattr(runner, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.base, "feature_columns", lambda *args: ["first30_ret"])
    monkeypatch.setattr(runner.base, "make_model", lambda kind: MeanModel())
    monkeypatch.setattr(runner.base, "target_values", lambda df, target: df.positive_r.to_numpy())
    result = runner.fit_oof_rankers(frame, frame.head(0))["test"]
    assert np.isnan(result["train_score"][:61]).all()
    assert result["train_score"][61] == pytest.approx(np.mean(np.arange(60)))
    changed = frame.copy()
    changed.loc[61:, "positive_r"] = 999999
    newer = runner.fit_oof_rankers(changed, changed.head(0))["test"]
    np.testing.assert_equal(result["train_score"][:66], newer["train_score"][:66])
    for fold in result["fold_proxy_rows"]:
        assert fold["train_through"] < fold["valid_from"]
        assert fold["label_through"] < fold["valid_from"]


def test_route_predictors_exclude_realized_entry_and_selection():
    import pandas as pd
    from scripts.kalcb_interaction_route_conversion_optimizer import feature_columns
    df = pd.DataFrame(columns=["first30_ret", "entered_shared_core", "selected_first30", "oracle_score"])
    assert feature_columns(df, "interaction_route") == ["first30_ret"]


def test_recovery_window_overrides_archived_nested_range_without_mutating_request():
    from scripts.kalcb_recovery_replay import bind_requested_window
    original = {"start": "2025-05-12", "end": "2026-03-31", "date_range": {
        "start": "2025-05-12", "end": "2026-03-31"}}
    bound = bind_requested_window(original, "2025-06-02T09:00:00+09:00", "2025-06-02T15:30:00+09:00")
    assert bound["date_range"] == {"start": "2025-06-02", "end": "2025-06-02"}
    assert original["date_range"]["start"] == "2025-05-12"
    with pytest.raises(ValueError, match="empty"):
        bind_requested_window(original, "2027-01-01")


@pytest.mark.parametrize("participation,slippage,name", [(.01,3,"research-baseline"),
    (.005,3,"participation-stress"),(.01,13,"cost-stress"),(.005,13,"combined-stress"),
    (.007,3,"unregistered-recovery-research")])
def test_recovery_capacity_preserves_declared_policy_identity(participation, slippage, name):
    from scripts.kalcb_recovery_replay import research_capacity_scenario
    assert research_capacity_scenario(participation, slippage) == {"name": name, "participation": participation}

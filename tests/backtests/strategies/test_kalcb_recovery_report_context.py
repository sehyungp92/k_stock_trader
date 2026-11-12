from copy import deepcopy

import pytest

from backtests.strategies.kalcb.recovery_full_diagnostics_v1 import recovery_report_context


@pytest.fixture
def profile():
    return {
        "date_range": {"start": "2026-04-01", "end": "2026-07-10", "sessions": 69},
        "recovery_report_context": {
            "period": "oos", "recipe": "decay60_atr5",
            "independent_validation": False, "june_july_used_for_research": True,
            "training_contract": {
                "minimum_training_sessions": 60, "refit_sessions": 20,
                "embargo_sessions": 1, "continuous_original_fitting_clock": True,
                "mature_labels_only": True, "earlier_oof_calibration_only": True,
            },
        },
    }


def test_full_oos_decay_description_cannot_reuse_old_is_caption(profile):
    before = deepcopy(profile)
    context = recovery_report_context(profile)
    assert "60-session half-life" in context["training"]
    assert "April-July 2026" in context["validation_scope"]
    assert "excluded" not in context["validation_scope"]
    assert "no independent validation claim" in context["validation_scope"]
    assert context["canonical_replaced"] is False
    assert context["production_enabled"] is False
    assert profile == before


def test_expanding_is_reports_actual_recipe_and_shared_research_usage(profile):
    profile["date_range"] = {"start": "2025-08-07", "end": "2026-03-31", "sessions": 156}
    profile["recovery_report_context"].update(period="is", recipe="expanding_atr5")
    context = recovery_report_context(profile)
    assert context["training"].startswith("Expanding history;")
    assert "separate KRW100m accounts" in context["validation_scope"]


@pytest.mark.parametrize("change", ["june_only", "independent", "excluded", "reset_clock", "unknown_recipe"])
def test_inaccurate_recovery_context_fails_before_report(profile, change):
    context = profile["recovery_report_context"]
    if change == "june_only":
        profile["date_range"]["start"] = "2026-06-01"
    elif change == "independent":
        context["independent_validation"] = True
    elif change == "excluded":
        context["june_july_used_for_research"] = False
    elif change == "reset_clock":
        context["training_contract"]["continuous_original_fitting_clock"] = False
    else:
        context["recipe"] = "invented_winner"
    with pytest.raises(ValueError):
        recovery_report_context(profile)

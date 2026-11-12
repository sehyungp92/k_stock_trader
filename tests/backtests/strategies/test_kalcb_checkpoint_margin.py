from scripts.kalcb_checkpoint_variant import meets_value_margin


def test_cost_margin_uses_equity_bps_and_scales_with_notional():
    row={'features':{'action_notional_equity':.1}}
    assert meets_value_margin(row,{'baseline_value_bps':8,'combined_value_bps':5},48)
    assert not meets_value_margin(row,{'baseline_value_bps':8,'combined_value_bps':4.8},48)
    row['features']['action_notional_equity']=.05
    assert meets_value_margin(row,{'baseline_value_bps':4,'combined_value_bps':2.5},48)
    assert not meets_value_margin(row,{'baseline_value_bps':2,'combined_value_bps':3},48)

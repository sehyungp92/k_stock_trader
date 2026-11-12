import pytest
from scripts.kalcb_entry_lifecycle_panel_v2 import entry_target


def test_invalid_cash_evidence_is_not_a_valid_economic_label():
    assert entry_target({'net_value_krw':None,'label_valid':False},10000) is None
    assert entry_target({'net_value_krw':-20,'label_valid':True},10000)==-20
    with pytest.raises(ValueError,match='Missing valid'):
        entry_target({'net_value_krw':None,'label_valid':True},10000)

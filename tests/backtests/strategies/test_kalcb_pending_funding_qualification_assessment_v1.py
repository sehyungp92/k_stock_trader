import pytest
from backtests.strategies.kalcb.pending_funding_qualification_assessment_v1 import meaningful,sacrifice_bands,completion_gate


def test_one_share_and_tiny_profits_do_not_create_meaningful_activity():
    p={'qty':2,'entry_cost_krw':2000000.,'entry_equity_krw':100000000.,'net_pnl_krw':10000.}
    rows=[p,{**p,'qty':1},{**p,'net_pnl_krw':1.},{**p,'entry_cost_krw':999999.}]
    assert meaningful(rows)['0.01/1.0']['positions']==1
    assert meaningful(rows)['0.01/5.0']['positions']==0


def test_no_invented_global_stress_return():
    for s in ('participation','cost'):
        v=sacrifice_bands(75,76,'is',s)
        assert v['local_return_delta_pp']==-1 and v['global_return_delta_pp'] is None
    assert sacrifice_bands(75,76,'is','baseline')['global_return_delta_pp']==pytest.approx(75-73.1303145472)


def test_live_supervisor_blocks_assessment(tmp_path):
    (tmp_path/'supervisor_start.json').write_text('{}')
    with pytest.raises(ValueError,match='supervisor'):completion_gate(tmp_path,alive=lambda i:True)

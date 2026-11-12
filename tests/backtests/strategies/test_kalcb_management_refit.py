import json
import pytest
from scripts import kalcb_management_refit as refit


def test_invalid_returns_remain_in_feasibility_target():
    row={'status':'filled','risk_available_at':'2025-08-07T15:35:00+09:00',
        'physical_fill_failures':[],'deadline_violation':False,'unresolved':False,
        'label_valid':False,'net_value_equity_bps':None}
    grid={'baseline':{'A':row},'combined':{'A':{**row,'deadline_violation':True,
        'risk_available_at':'2025-08-08T15:35:00+09:00'}}}
    assert refit.failure_outcomes(grid)['A']=={'unsafe':True,'risk_available_at':'2025-08-08T15:35:00+09:00'}
    with pytest.raises(ValueError):refit.failure_outcomes({'baseline':{'A':row},'combined':{}})


def test_management_grid_cannot_mix_exit_policies_or_omit_full_quantity(tmp_path,monkeypatch):
    (tmp_path/'request.json').write_text(json.dumps({'arguments':{'plan':'other.json'}}))
    plan={k:k for k in ('config','database','seed','q3')}
    plan['input_sha256']={k:'same' for k in plan}
    monkeypatch.setattr(refit,'load_plan',lambda _: {**plan,'input_sha256':{**plan['input_sha256'],'config':'different'}})
    with pytest.raises(ValueError,match='config'):refit.changed_grid([tmp_path],tmp_path,plan)
    monkeypatch.setattr(refit,'load_plan',lambda _:plan)
    monkeypatch.setattr(refit.ev,'digest',lambda _:'hash')
    monkeypatch.setattr(refit,'load_grid',lambda *_:({'baseline':{'A/q0.5':{}},'combined':{'A/q0.5':{}}},{}))
    with pytest.raises(ValueError,match='Complete'):refit.changed_grid([tmp_path],tmp_path,plan,{'A/q0.5','A/q1'})

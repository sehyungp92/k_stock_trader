import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('independent_holding_review',Path(__file__).resolve().parents[3]/'scripts/kalcb_checkpoint_holding_model_review_v1.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def fixture():
    row={'day':'2026-01-05','eligibility':'eligible','held_quantity':10,'pending_buy_quantity':0,'quantity_scale':.25,
         'own_features':{'own':1.},'state_features':{'state':0.},'context_features':{'context':1.}}
    proof={'valid_from':'2026-01-05','valid_through':'2026-01-30','status':'supported',
           'quantity_support':{'0.25':{'supported':True}},'models':{}}
    for s in r.SCENARIOS:proof['models'][s]={'center':[0.,0.,0.],'scale':[1.,1.,1.],'minimum':[0.,0.,0.],
       'maximum':[2.,0.,2.],'coefficients':{'own_state':[10.,2.,1.],'peer_context':[10.,2.,1.,3.]}}
    return {s:copy.deepcopy(row) for s in r.SCENARIOS},proof


def test_independent_prediction_distinguishes_incremental_context():
    pair,p=fixture()
    a=r.prediction(pair,p,'own_state',['own','state','context'],2)
    b=r.prediction(pair,p,'peer_context',['own','state','context'],2)
    assert a['values']['baseline']==12 and b['values']['baseline']==15


@pytest.mark.parametrize('recipe',r.RECIPES)
def test_independent_domain_rejects_context_for_both_models(recipe):
    pair,p=fixture();pair['combined']['context_features']['context']=3.
    a=r.prediction(pair,p,recipe,['own','state','context'],2)
    assert not a['supported'] and a['reason']=='prior_feature_envelope'


def test_prediction_poison_rejected():
    with pytest.raises(ValueError):r.check_prediction({'supported':True,'values':{'baseline':11.,'combined':12.}},
                                                      {'supported':True,'values':{'baseline':12.,'combined':12.}})


def test_pending_order_does_not_receive_prediction():
    pair,p=fixture();pair['baseline']['pending_buy_quantity']=1
    assert not r.prediction(pair,p,'own_state',['own','state','context'],2)['supported']


def test_independent_calibration_identity_and_support():
    points=[((f'2026-01-{i+1:02d}','A'),float(i),float(i)) for i in range(20)]
    p=r.affine_equations(points);assert p['supported'] and p['offset']==0 and p['slope']==1
    assert not r.affine_equations(points[:19])['supported']


def test_calibration_does_not_inspect_future_labels():
    row={'day':'2026-02-01','labels':None}
    assert r.affine_points([row],{'2026-01-05'},'baseline')==[]


def test_metric_is_not_quantity_replication_weighted():
    a={'day':'2026-01-05','symbol':'A','prediction':0.,'target':10.}
    b={'day':'2026-01-05','symbol':'B','prediction':0.,'target':0.}
    one=r.metric([a,b]);replicated=r.metric([a,a,a,b]);assert one['mse_bps2']==pytest.approx(replicated['mse_bps2'])
    assert replicated['mse_bps2']==pytest.approx(50.)


def test_unmatched_native_target_is_never_profit_evidence():
    row={'day':'2026-01-05','symbol':'A','matched_predecision_both':False,
         'calibrated':{'supported':True,'values':{'baseline':100.,'combined':100.}},'labels':{}}
    p=r.positive_metrics([row],'calibrated');assert p['unknown_label_proposals']==1 and p['exact_label_proposals']==0

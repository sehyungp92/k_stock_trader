from copy import deepcopy
import numpy as np
from scripts.kalcb_structural_information_v1 import balance,demean,mature,renewal_features,shadow_select


def test_group_contrast_is_invariant_to_common_session_quantity_payoff_shift():
    rows=[{'day':'a','base_id':'a/x','quantity_scale':.25},{'day':'a','base_id':'a/y','quantity_scale':.25},
          {'day':'a','base_id':'a/x','quantity_scale':.5},{'day':'b','base_id':'b/x','quantity_scale':.25}]
    a=np.array([10.,-20.,80.,5.]);b=np.array([110.,80.,-50.,90.])
    assert np.allclose(demean(a,rows),demean(b,rows))
    w=balance(rows)
    assert np.isclose(w[:3].sum(),w[3])
    assert np.isclose(w[0]+w[2],w[1])


def test_mature_rejects_future_and_embargoed_outcomes():
    days=['2026-04-01','2026-04-02','2026-04-03']
    r={'day':days[0],'label_valid':True,'label_available_at':'2026-04-01T15:35:00+09:00'}
    assert mature(r,days,2)
    r['day']=days[1];assert not mature(r,days,2)
    r['day']=days[0];r['label_available_at']='2026-04-03T01:00:00+09:00';assert not mature(r,days,2)


def test_renewal_uses_current_state_and_ignores_future_outcome():
    r={'features35':list(range(12)),'quantity_scale':.5,'pending_qty_at35':7,'quantity_at35':3,'planned_qty':10}
    a={'calibrated_core':{'baseline':2,'combined':1},'notional_equity':.2}
    f=renewal_features(r,a,'N');r['remaining_net_after35_krw']=1e20;r['future_other_scenario_qty']=500
    assert renewal_features(r,a,'N')==f
    assert f[-2:]==[.7,.3] and f[7:9]==[10.,5.]


def test_shadow_selection_respects_quantity_quota_cash_and_unique_name():
    def row(k,n,q,selected=False):
        return {'day':'a','base_id':k.split('/q')[0],'action_id':k,'quantity_scale':q,'notional_equity':n,
                'physical_risk_pass':True,'capacity_pass':True,'selected':{'control':{}} if selected else {}}
    rr=[row('x/q.5',.2,.5,True),row('y/q.25',.1,.25,True),row('z/q.5',.3,.5),row('z/q.25',.1,.25)]
    pp={r['action_id']:{'A':{'baseline':i+1,'combined':i+1}} for i,r in enumerate(rr)}
    result=shadow_select(rr,pp,'A')
    assert len({x.split('/q')[0] for x in result})==len(result)
    assert sum(r['notional_equity'] for r in rr if r['action_id'] in result)<=.3+1e-9
    assert len(result)<=2

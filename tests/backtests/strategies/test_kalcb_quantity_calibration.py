from copy import deepcopy
from datetime import date,timedelta
from scripts.kalcb_quantity_calibration import corrections

def test_calibration_is_out_of_fold_past_only_and_embargoed():
    days=[(date(2025,1,1)+timedelta(days=i)).isoformat() for i in range(50)]
    rows=[];labels={'baseline':{},'combined':{}}
    for i,d in enumerate(days):
        key=d+'/A/q0.5'
        rows.append({'trade_date':d,'opportunity_id':key,'quantity_scale':.5,
            'baseline_value_bps':10,'combined_value_bps':8,'failure_probability':0,'capacity_ok':True})
        for s in labels:labels[s][key]={'label_valid':True,'label_available_at':d+'T15:35:00+09:00','net_value_equity_bps':-2}
    # A delayed old observation must also be excluded, not just future dates.
    for s in labels:labels[s][rows[2]['opportunity_id']]['label_available_at']=days[45]+'T15:35:00+09:00'
    expected,proof=corrections(rows,labels,days,days[40])
    assert proof['cohorts']['baseline/q0.5']['observations']==38
    assert expected[('baseline',.5)]==-12*38/138
    assert expected[('baseline',.25)]==0
    for i,r in enumerate(rows):
        if i>=39 or i==2:
            for s in labels:labels[s][r['opportunity_id']]['net_value_equity_bps']=1e12
    assert corrections(rows,labels,days,days[40])==(expected,proof)

def test_no_calibration_before_sufficient_observations():
    days=['2025-01-01','2025-01-02']
    offsets,proof=corrections([],{'baseline':{},'combined':{}},days,days[-1])
    assert not any(offsets.values())
    assert proof['latest_label_available_at'] is None


def test_builder_rescues_calibrated_value_without_bypassing_stress_risk_or_capacity(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts import kalcb_quantity_calibration as module
    from scripts import kalcb_action_value_study as actions
    from backtests.strategies.kalcb import executed_value as ev
    day='2025-01-15';prepared=tmp_path/'prepared';prepared.mkdir();parent=tmp_path/'parent';parent.mkdir()
    rows=[];values={}
    for symbol in ('000100','000270','000660','000720'):
        key=day+'/'+symbol
        row={'opportunity_id':key,'trade_date':day,'symbol':symbol,'candidate':{'metadata':{}},
            'features':{k:1. for k in ('action_notional_equity','action_quantity_last_volume','action_quantity_opening_volume','action_low_volume_exit_slices')},
            'action':{'quantity':1000000 if symbol=='000720' else 20,'ready':True},
            'feature_available_at':day+'T09:30:00+09:00','opening_bars':[{'volume':10000} for _ in range(6)]}
        rows.append(row)
        for q in (.25,.5,1):
            action=key+f'/q{q:g}'
            values[action]={'opportunity_id':action,'baseline_value_bps':-1.,
                'combined_value_bps':-4. if symbol=='000270' else -2.,
                'failure_probability':.002 if symbol=='000660' else 0.,'valid_from':day}
    (prepared/'opportunities.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    meta={'sessions':[day],'opportunities_sha256':ev.digest(prepared/'opportunities.jsonl'),
        'parent_prepared_sha256':'parent','original_prepared_sha256':'original'}
    (prepared/'prepared.json').write_text(json.dumps(meta));(parent/'chronology.json').write_text('{"folds":[]}')
    seed=tmp_path/'seed.json';seed.write_text('{"mutations":{}}')
    manifest={'prepared_sha256':ev.digest(prepared/'prepared.json'),'evaluation_from':day,'evaluation_through':day}
    (parent/'policies_manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(module,'forecast_values',lambda *args:(manifest,values))
    label_rows={r['opportunity_id']:{'label_valid':True} for r in rows}
    monkeypatch.setattr(module,'compact_labels',lambda *args,**kwargs:({'baseline':deepcopy(label_rows),'combined':{}},{}))
    monkeypatch.setattr(module,'config_for',lambda plan:None)
    monkeypatch.setattr(module,'corrections',lambda *args:({(s,q):3. for s in ('baseline','combined') for q in (.25,.5,1)}, {'cohorts':{}}))
    monkeypatch.setattr(ev,'outer_windows',lambda *args,**kwargs:[(day,day)])
    monkeypatch.setattr(ev,'eligible',lambda *args,**kwargs:True)
    monkeypatch.setattr(ev,'snapshot_for',lambda d,selected,**kwargs:SimpleNamespace(to_json_dict=lambda:{'day':d,'selected':[r['opportunity_id'] for r,score in selected]}))
    real_admit=actions.admit_actions
    monkeypatch.setattr(actions,'admit_actions',lambda *args,**kwargs:real_admit(*args,**kwargs,ready=False))
    admitted={}
    for mode in ('positive_offset','offset_all'):
        out=tmp_path/mode;out.mkdir()
        module.run({'seed':str(seed),'q3':str(seed)},prepared,parent,'model',[],[],out,mode)
        policy=json.loads((out/'policies_manifest.json').read_text())['policies'][0]
        admitted[mode]=ev.read_jsonl(out/(policy+'.admission.jsonl'))
    assert admitted['positive_offset']==[]
    assert [r['opportunity_id'] for r in admitted['offset_all']]==[day+'/000100']
    assert admitted['offset_all'][0]['score']==2.

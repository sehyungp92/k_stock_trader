import json,os
from pathlib import Path
import pytest
from scripts import kalcb_retained_funding_v1 as study


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v),encoding='utf-8')


def test_complete_assessor_on_real_identity_accounts_and_rejection(tmp_path):
    controls={};ledger=[]
    for period in ('is','oos'):
        for scenario in ('baseline','combined'):
            source=study.C/'cases/veto'/period/scenario
            assert (source/'independent_account_verification.json').exists()
            controls[period+'/'+scenario]=str(source)
            for mode in ('queue','forecast'):
                name=mode+'_'+period+'_'+scenario;out=tmp_path/'cases'/name;out.mkdir(parents=True)
                for p in source.iterdir():
                    if p.is_file():os.link(p,out/p.name)
                ledger.append(dict(case=dict(id=name,mode=mode,period=period,scenario=scenario),verdict=dict(qualified=True)))
    write(tmp_path/'protocol.json',dict(controls=controls,prior_controls=controls));write(tmp_path/'completed_cases.json',ledger)
    study.assess(tmp_path);result=study.read(tmp_path/'assessment.json')
    assert len(result['cells'])==len(result['versus_prior100'])==8 and len(result['forecast_minus_queue'])==4
    for cell in list(result['cells'].values())+list(result['forecast_minus_queue'].values())+list(result['versus_prior100'].values()):
        assert cell['net_delta_krw']==0 and cell['useful_days']==cell['control_useful_days']
        assert cell['metrics']['return_pct']==cell['control_metrics']['return_pct']
        assert cell['activity_delta']['net_pnl_delta_krw']==0
    ledger[-1]['verdict']=dict(qualified=False,reason='Unresolved native account')
    write(tmp_path/'completed_cases.json',ledger);study.assess(tmp_path)
    result=study.read(tmp_path/'assessment.json')
    assert len(result['cells'])==7 and len(result['forecast_minus_queue'])==3
    assert 'forecast_oos_combined' in result['rejected'] and not result['economic_objective_achieved']


def test_activation_checks_real_quantity_omission_and_full_submissions(tmp_path):
    import gzip
    from tests.test_kalcb_retained_funding_v1 import rows
    rr=rows();book,left,priority=study.reserve(rr,100,'forecast');at='2026-01-05T09:30:00+09:00'
    record=dict(day='2026-01-05',available_at=at,mode='forecast',reason='supported_overbooked',
        changed=True,intents=rr,room=100,book=book,remaining_room=left,priority=priority)
    write(tmp_path/'reservation_actions.json',[record])
    entries=[dict(event='submit',order=dict(symbol=s,side='BUY',qty=r['qty'],metadata=dict(entry_route='first30_open_anchor',decision_available_at=at)))
             for s,r in book.items() if r['qty']]
    def journal():
        with gzip.open(tmp_path/'orders.jsonl.gz','wt',encoding='utf-8') as f:
            for row in entries:f.write(json.dumps(row)+'\n')
    journal();result=study.activation(tmp_path,'forecast');assert result['omitted_requests']==1 and result['changed_batches']==1
    entries[0]['order']['qty']-=1;journal()
    with pytest.raises(ValueError,match='activation mismatch'):study.activation(tmp_path,'forecast')


def test_probe_arguments_keep_actual_worker_inputs_and_bridge_is_not_financial():
    item=dict(plan=dict(path='plan'),policy=dict(path='policies/policies_manifest.json'),manifest=dict(path='manifest'))
    cmd=study.command(item,'forecast','combined',Path('output'))
    probe=[cmd[0],'-B','-c',study.PROBE]+cmd[5:]
    assert probe[4:6]==['--plan','plan'] and '--mode' in probe and '--scenario' in probe
    assert 'financial_replay_executed=False' in study.PROBE

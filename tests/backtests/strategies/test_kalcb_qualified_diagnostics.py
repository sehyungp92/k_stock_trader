import json
import math
from datetime import datetime, timedelta
import pytest
from strategy_common.clock import KST
from backtests.auto.shared.cache_keys import stable_signature
from backtests.strategies.kalcb.full_diagnostics import _equity_risk_metrics, _num, _final_metrics
from backtests.strategies.kalcb.qualified_artifacts import digest, load_qualified_profile, validate_case


def test_intraday_sampling_cannot_inflate_sortino():
    start = datetime(2026, 1, 5, 15, 0, tzinfo=KST)
    daily = [100, 101, 99, 103, 102]
    stamps = [start + timedelta(days=i) for i in range(4)]
    sparse = _equity_risk_metrics(daily, timestamps=stamps)
    dense, times = [100], []
    for i, end in enumerate(daily[1:]):
        dense.extend([daily[i] * 1.05, daily[i] * .97, end])
        times.extend([stamps[i] - timedelta(minutes=10), stamps[i] - timedelta(minutes=5), stamps[i]])
    assert _equity_risk_metrics(dense, timestamps=times) == sparse
    returns = [b/a - 1 for a,b in zip(daily,daily[1:])]
    expected = sum(returns)/len(returns) / math.sqrt(sum(min(r,0)**2 for r in returns)/len(returns))*math.sqrt(252)
    assert sparse['sortino'] == pytest.approx(expected)


def test_unknown_sampling_is_unavailable_and_bad_alignment_rejected():
    assert _equity_risk_metrics([100, 101])['sortino'] is None
    assert _num(None) == 'n/a'
    assert _num(1000) == '1000.00'
    with pytest.raises(ValueError):
        _equity_risk_metrics([100, 101], timestamps=[datetime(2026,1,5,tzinfo=KST)]*2)


def test_frequency_includes_declared_cash_only_period():
    # Empty trade days are part of the measured opportunity window.
    from strategy_common.events import TradeOutcome
    stamp = datetime(2026,2,2,10,tzinfo=KST)
    trade = TradeOutcome('kalcb','005930',1,stamp,stamp,100,stamp,101,1,0,1,True)
    result = _final_metrics({}, [trade], {'date_range': {'start':'2026-01-01','end':'2026-03-31'}})
    assert result['trades_per_month'] == pytest.approx(30.4375 / 90)
    with pytest.raises(ValueError, match='Reversed diagnostic date range'):
        _final_metrics({}, [trade], {'date_range': {'start':'2026-03-31','end':'2026-01-01'}})


def test_canonical_round_r_deducts_costs():
    from strategy_common.events import TradeOutcome
    stamp = datetime(2026,2,2,10,tzinfo=KST)
    trade = TradeOutcome('kalcb','005930',100,stamp,stamp,100,stamp,101,
        100,30,70,True,route_metadata={'risk_per_share':10})
    result = _final_metrics({'expected_total_r':.1,'avg_r':.1}, [trade], {})
    assert result['expected_total_r'] == pytest.approx(.07)
    assert result['avg_r'] == pytest.approx(.07)
    assert result['gross_expected_total_r'] == pytest.approx(.1)


def test_modified_export_is_rejected_before_scoring(tmp_path):
    export = tmp_path/'result.json'
    export.write_text('{"return_pct": 73.13}')
    manifest = {'version':'kalcb-qualified-native-exports-v1','profile':'irrelevant',
                'files':{str(export):digest(export)}}
    path=tmp_path/'acceptance.json'
    path.write_text(json.dumps({**manifest,'manifest_hash':stable_signature(manifest)}))
    export.write_text('{"return_pct": 999}')
    with pytest.raises(ValueError,match='Changed qualified artifact'):
        load_qualified_profile(path)


@pytest.mark.parametrize('status', ['failed','unverified'])
def test_attractive_but_unqualified_replay_cannot_be_promoted(tmp_path,status):
    for name,data in {'completion.json':{'complete':True},'request.json':{},
        'result.json':{'return_pct':999,'audit_status':status,'execution_validity_pass':True},
        'execution_audit.json':{'audit_status':status,'audit_pass':False}}.items():
        (tmp_path/name).write_text(json.dumps(data))
    with pytest.raises(ValueError,match='cannot be promoted'):
        validate_case(tmp_path,{},'baseline')


def test_changed_receipt_is_rejected_even_with_passing_labels(tmp_path):
    data={'completion.json':{'complete':True,'source_generation_hash':'bound'},
        'request.json':{'source_generation_hash':'bound'},
        'result.json':{'audit_status':'passed','execution_validity_pass':True,'source_generation_hash':'bound'},
        'execution_audit.json':{'audit_status':'passed','audit_pass':True,'audit_receipt':{'receipt_hash':'fabricated'}}}
    for name,value in data.items():(tmp_path/name).write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Changed execution audit receipt'):
        validate_case(tmp_path,{'execution_source':{'source_generation_hash':'bound'}},'baseline')


def test_failed_qualification_stops_public_promotion_before_round_creation(tmp_path):
    from backtests.strategies.kalcb.full_diagnostics import main
    evidence = tmp_path/'account_check.json'
    evidence.write_text(json.dumps({'passed': False}))
    profile = tmp_path/'profile.json'
    profile.write_text(json.dumps({**{k:str(evidence) for k in (
        'configuration','audit_manifest','plan','all_trials','attribution','scope_preparation_parity')},
        'metrics': {'baseline': {'return_pct': 999}}}))
    manifest = {'version':'kalcb-qualified-native-exports-v1', 'profile':str(profile),
                'files':{str(p):digest(p) for p in (profile,evidence)}}
    capsule = tmp_path/'acceptance.json'
    capsule.write_text(json.dumps({**manifest,'manifest_hash':stable_signature(manifest)}))
    output = tmp_path/'output'
    with pytest.raises(ValueError,match='Failed qualification evidence'):
        main(['--acceptance-manifest',str(capsule),'--output-root',str(output)])
    assert not output.exists()


def test_legacy_command_cannot_overwrite_accepted_round(tmp_path):
    from backtests.strategies.kalcb.full_diagnostics import main
    target = tmp_path/'kalcb'/'round_1'
    target.mkdir(parents=True)
    config = target/'optimized_config.json'
    config.write_text('{"locked":true}')
    with pytest.raises(FileExistsError, match='Preserve/archive'):
        main(['--output-root',str(tmp_path)])
    assert config.read_text() == '{"locked":true}'

import pytest
from scripts.kalcb_population_assessment_v1 import account,paired_attribution,paired_blocks,write


def make_account(folder):
    days=['2026-01-30','2026-02-02','2026-02-03'];values=[10_000_000.,-11_000_000.,1_000_000.]
    trades=[{'entry_decision_time':day+'T09:30:00+09:00','exit_fill_time':day+'T14:45:00+09:00',
             'symbol':str(i),'net_pnl':value,'gross_pnl':value+1000,'commission':1000} for i,(day,value) in enumerate(zip(days,values))]
    write(folder/'trades.json',trades)
    write(folder/'daily.json',[{'day':day,'equity':value} for day,value in zip(days,[110_000_000,99_000_000,100_000_000])])
    write(folder/'result.json',{'return_pct':0.,'drawdown_pct':10.,'trades':3,'active_days':3,'audit_status':'passed'})
    write(folder/'independent_account_verification.json',{'passed':True})
    for name in ('equity.json','fills.json'):write(folder/name,[])


def test_monthly_returns_use_actual_capital_and_cash(tmp_path):
    make_account(tmp_path);a=account(tmp_path)
    assert a['monthly']['2026-01']['return_pct']==pytest.approx(10)
    assert a['monthly']['2026-02']['return_pct']==pytest.approx(-100/11)
    assert a['monthly']['2026-02']['cash_change_krw']==-10_000_000
    assert a['net_pnl_krw']==0 and a['explicit_costs_krw']==3000
    comparison=paired_attribution(a,a)
    assert comparison['new_traded_issuer_days']==0
    assert comparison['same_issuer_days_variant_pnl_krw']==10_000_000


def test_block_comparison_is_paired_and_compounded():
    same=paired_blocks([.1,-.1,.02],[.1,-.1,.02],2,draws=20)
    assert same['return_delta_pp_2_5']==same['return_delta_pp_97_5']==0
    gain=paired_blocks([.01]*3,[0]*3,2,draws=20)
    assert gain['return_delta_pp_50']==pytest.approx(3.0301)
    with pytest.raises(ValueError,match='Unpaired'):paired_blocks([0],[0,0],2)


def test_account_refuses_unreconciled_trade_cash(tmp_path):
    make_account(tmp_path)
    write(tmp_path/'trades.json',[])
    # Net cash happens to match zero; fee/return checks alone would miss this.
    # Monthly attribution must still reject cancelling monthly discrepancies.
    with pytest.raises(ValueError,match='Monthly cash mismatch'):account(tmp_path)

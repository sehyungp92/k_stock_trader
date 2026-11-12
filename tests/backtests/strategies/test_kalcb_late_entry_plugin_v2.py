from types import SimpleNamespace
from backtests.strategies.kalcb.late_entry_plugin_v2 import LateEntryPluginV2


def test_shared_end_of_round_report_contract_keeps_screening_label(tmp_path):
    plugin=object.__new__(LateEntryPluginV2)
    plugin.diagnostic=lambda:{'trials':[],'study_complete':False}
    plugin.compute_final_metrics=lambda _: {'diagnostic_score':0}
    result=plugin.write_full_diagnostics(SimpleNamespace(cumulative_mutations={}),str(tmp_path))
    text=(tmp_path/'round_final_diagnostics.txt').read_text()
    assert 'interim screening ledger' in text
    assert result['complete'] and not result['existing_engine_final_reports_complete']


def test_final_rank_preview_skips_blocked_candidate_and_never_spends_two_slots(monkeypatch):
    from datetime import datetime
    from backtests.strategies.kalcb import late_policy_v2 as module
    at=datetime.fromisoformat('2026-04-01T10:00:00+09:00');seen=[]
    monkeypatch.setattr(module,'append_snapshot',lambda latest,row,value,flat:SimpleNamespace(generated_at=at,symbol=row['symbol'],locked=latest,flat=flat))
    def preview(bars,snap,cfg,account,available_at):
        seen.append(snap.symbol)
        assert snap.locked=='locked_incumbents' and snap.flat and available_at==at
        return {'intents':[] if snap.symbol=='A' else [{'symbol':snap.symbol}]}
    choices=[({'symbol':symbol,'opening_bars':[]},{'baseline':score,'combined':score}) for symbol,score in [('C',1),('A',3),('B',2)]]
    selected=module.select_executable(choices,'locked_incumbents',None,True,preview_fn=preview)
    assert selected[0]['symbol']=='B' and seen==['A','B']

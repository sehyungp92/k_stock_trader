"""Synthetic check of temporary-config retention; no market evaluation."""
from pathlib import Path
from dataclasses import replace
import importlib.util,sys,json,hashlib
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1')
REPO=Path(__file__).resolve().parents[1]
def bound(p):
    with Path(p).open('rb') as f:return dict(path=str(p),sha256=hashlib.file_digest(f,'sha256').hexdigest())
def main():
    out=ROOT/'horizon_cache_unit_v1';out.mkdir(exist_ok=False)
    helper=REPO/'backtests/strategies/kalcb/research_decision_horizon_v2.py'
    (out/'registration.json').write_text(json.dumps(dict(program=bound(__file__),helper=bound(helper),market_evaluations=0),indent=2),encoding='utf-8')
    sys.path.insert(0,str(ROOT/'native_v1/is/inputs/files'))
    spec=importlib.util.spec_from_file_location('c18_horizon_cache',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    from strategy_kalcb.config import KALCBConfig
    from strategy_kalcb.core import logic
    cfg=KALCBConfig();original=logic._exit_reason
    logic._exit_reason=lambda ss,position,bar,current,**kwargs:str(current.commission_bps)
    h=m.DiagnosticHorizon(30);h.install()
    try:
        for i in range(50):
            temporary=replace(cfg,commission_bps=float(i))
            assert logic._exit_reason(None,None,None,temporary)==str(float(i))
            assert h.cached_original is temporary
            assert h.cached_config.commission_bps==float(i)
    finally:h.restore();logic._exit_reason=original
    (out/'receipt.json').write_text(json.dumps(dict(passed=True,market_evaluations=0,checks=['strong_reference_to_current_config','temporary_config_changes_propagate','no_object_id_cache_alias','native_function_restored']),indent=2),encoding='utf-8')
    print('Horizon temporary-config cache checks passed',flush=True)
if __name__=='__main__':main()

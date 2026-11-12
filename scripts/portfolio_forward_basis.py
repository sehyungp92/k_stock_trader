"""Quotation-only checks of original/adjusted OHLCV across the reserved boundary.

Raw responses remain separate from the frozen input generation. No strategy
performance, order submission, or automatic source replacement occurs here.
"""
from datetime import datetime
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from scripts.portfolio_targeted_validation import ROOT,REPO,read,bound
from backtests.strategies.kalcb.executed_value import write_json


def collect(root):
    import pandas as pd
    import yaml
    from scripts.update_kis_intraday import _load_dotenv,KST
    from kis_core import KoreaInvestAPI,KoreaInvestEnv,build_kis_config_from_env
    from loguru import logger
    _load_dotenv(REPO/'.env');logger.remove()
    api=KoreaInvestAPI(KoreaInvestEnv(build_kis_config_from_env()))
    destination=root/'provider_basis_audit';destination.mkdir(exist_ok=True)
    symbols=[str(s).zfill(6) for s in yaml.safe_load((REPO/'config/universe_103.yaml').read_text())['symbols']]
    results=[];fields={'open':'stck_oprc','high':'stck_hgpr','low':'stck_lwpr','close':'stck_clpr','volume':'acml_vol'}
    for i,symbol in enumerate(symbols):
        controls={}
        for adjusted in (False,True):
            path=destination/(symbol+('_adjusted.json' if adjusted else '_original.json'))
            if not path.exists():
                params={'FID_COND_MRKT_DIV_CODE':'J','FID_INPUT_ISCD':symbol,'FID_INPUT_DATE_1':'20260529',
                    'FID_INPUT_DATE_2':'20260710','FID_PERIOD_DIV_CODE':'D','FID_ORG_ADJ_PRC':'0' if adjusted else '1'}
                endpoint='/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice'
                response=api._url_fetch(endpoint,'FHKST03010100',params)
                if response is None or not response.is_ok():raise ValueError('Daily basis request failed for '+symbol)
                write_json(path,{'symbol':symbol,'adjusted':adjusted,'params':params,'endpoint':endpoint,
                    'queried_at':datetime.now(KST).isoformat(),'status':'ok','rows':getattr(response.get_body(),'output2',[])})
            receipt=read(path)
            controls[adjusted]={datetime.strptime(r['stck_bsop_date'],'%Y%m%d').date().isoformat():
                               {k:int(r[v]) for k,v in fields.items()} for r in receipt['rows']}
        source=next((REPO/'data/krx_daily_parquet/daily_ohlcv'/symbol).glob('*_20260710.parquet'))
        frame=pd.read_parquet(source);frame['day']=pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
        for row in frame[frame.day.between('2026-05-29','2026-07-10')].to_dict('records'):
            day=row['day'];archive={k:float(row[k]) for k in fields}
            if any(day not in values for values in controls.values()):raise ValueError('Missing provider basis date '+symbol+' '+day)
            original,adjusted=controls[False][day],controls[True][day]
            results.append({'symbol':symbol,'date':day,'archive':archive,'original':original,'adjusted':adjusted,
                'matches_original':archive==original,'matches_adjusted':archive==adjusted,
                'original_differs_from_adjusted':original!=adjusted})
        if (i+1)%10==0:print(json.dumps({'provider_basis_symbols':i+1,'total':len(symbols)}),flush=True)
    exceptions=[r for r in results if not r['matches_original']]
    result={'symbols':len(symbols),'sessions':results,'non_original_sessions':exceptions,
        'all_original':not exceptions,'original_and_adjusted_controls':{p.name:bound(p) for p in destination.glob('*.json') if p.name!='audit.json'},
        'strategy_performance_evaluated':False,'source_gate_pass':not exceptions,'program':bound(Path(__file__))}
    write_json(destination/'audit.json',result)
    print(json.dumps({'provider_basis_audit_complete':True,'symbols':len(symbols),'non_original_sessions':len(exceptions),
        'affected_symbols':sorted({r['symbol'] for r in exceptions})}),flush=True)


if __name__=='__main__':collect(ROOT)

"""Separate feature history, native-label history and scored IS boundaries."""
from collections import defaultdict
import json,math
from pathlib import Path
import sqlite3,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_full_oos_recovery import ROOT,FULL,read,write,bound

def coverage(path,start,end):
    keys=('daily_return_5d','daily_return_20d','daily_return_60d','daily_atr_pct',
        'daily_adv20_krw_log','prior_afternoon_return','prior_intraday_return','prior_path_available')
    counts=defaultdict(int);total=0;first=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if not start<=row['trade_date']<=end:continue
            total+=1
            for k in keys:
                v=row['features'].get(k)
                if v is None or not math.isfinite(float(v)):counts[k]+=1
            if row['features'].get('prior_path_available')!=1:counts['prior_path_unavailable']+=1
            if row['trade_date']==start:first.append({'symbol':row['symbol'],'prior_path_available':row['features'].get('prior_path_available')})
    return {'rows':total,'missing':dict(counts),'first_day_prior_path':first}

def main():
    import pyarrow.parquet as pq
    repo=Path(__file__).resolve().parents[1];intraday=[]
    for p in sorted((repo/'data/kis_intraday_parquet').rglob('*_5m_*.parquet')):
        f=pq.ParquetFile(p);idx=f.schema.names.index('timestamp');ranges=[]
        if not f.metadata.num_rows:continue
        for i in range(f.metadata.num_row_groups):
            stat=f.metadata.row_group(i).column(idx).statistics
            if stat and stat.has_min_max:ranges.append((str(stat.min),str(stat.max)))
            else:
                values=[v for v in f.read_row_group(i,columns=['timestamp'])['timestamp'].to_pylist() if v is not None]
                if values:ranges.append((str(min(values)),str(max(values))))
        if ranges:intraday.append({'path':str(p),'earliest':min(a for a,b in ranges),'latest':max(b for a,b in ranges),'rows':f.metadata.num_rows})
    daily=[]
    for name in ('daily_ohlcv','daily_flow','index_ohlcv'):
        p=repo/'data/krx_daily_parquet/tables'/(name+'.parquet');f=pq.ParquetFile(p)
        column=next((k for k in ('date','day','timestamp') if k in f.schema.names),None)
        if not column:raise ValueError('Unknown daily date schema')
        idx=f.schema.names.index(column);ranges=[]
        for i in range(f.metadata.num_row_groups):
            st=f.metadata.row_group(i).column(idx).statistics
            if st and st.has_min_max:ranges.append((str(st.min),str(st.max)))
        if not ranges:
            values=pq.read_table(p,columns=[column])[column].to_pylist();ranges=[(str(min(values)),str(max(values)))]
        daily.append({'path':str(p),'earliest':min(a for a,b in ranges),'latest':max(b for a,b in ranges)})
    history=Path('E:/CodexResearch/k_stock_trader/kalcb_continuation_20261006/prepared_v1')
    meta=read(history/'prepared.json');sessions=meta['sessions']
    conn=sqlite3.connect('file:C:/Users/sehyu/.codex/tmp/kalcb_calibrated_execution_20261007/market.sqlite?mode=ro',uri=True)
    extent=conn.execute('select min(day),max(day),count(distinct day),count(distinct symbol) from bars').fetchone();conn.close()
    selected=read(repo/'config/research/kalcb_intraday_advantage_20261007/selected_research.json')
    proofs=read(FULL/'frozen_inference_bridge.json')['models']
    model_ranges=[{k:p['training_proof'][k] for k in ('name','train_from','train_through','valid_from','label_available_through','train_rows')}|{'kind':p['kind'],'target':p['target']} for p in proofs]
    coverage_result={
        'training_prefix':coverage(history/'opportunities.jsonl',sessions[0],sessions[60]),
        'scored_IS':coverage(history/'opportunities.jsonl',sessions[61],sessions[-1]),
        'full_OOS':coverage(FULL/'forward_features/continuation/opportunities.jsonl','2026-04-01','2026-07-10')}
    if sessions[61]!='2025-08-07' or any(p['train_from']!='2025-05-12' for p in model_ranges):raise ValueError('Unexpected model-history clock')
    payload={'scored_IS_start':sessions[61],'training_history_start':sessions[0],'min_train_sessions':60,'embargo_sessions':1,
        'first_scored_fit_training_through':sessions[59],'first_scored_fit_embargo_day':sessions[60],
        'history_sessions':len(sessions),'database_extent':extent,'prepared':bound(history/'prepared.json'),
        'current_frozen_models':model_ranges,'local_5m_files':intraday,'local_5m_earliest':min(p['earliest'] for p in intraday),
        'older_daily_history':daily,'feature_coverage':coverage_result,'program':bound(__file__),
        'conclusions':[
            'The August score boundary does not truncate the model training data to August: pre-August native outcomes are used.',
            'Older daily prices support long-lookback predictors but cannot substitute for missing 5-minute opening, capacity and execution outcomes.',
            'The finite intraday-label sample can affect statistical generalization; absence of boundary leakage does not prove more older history would be useless.',
            'Shortening warm-up changes how much IS is scored, not the mature March training prefix of an expanding model.',
            'OOS staleness and history weighting are tested separately by causal terminal, expanding, rolling120 and decay60 model experiments.']}
    write(ROOT/'history_boundary_audit.json',payload)
    lines=['KALCB HISTORY AND SCORING BOUNDARY AUDIT',json.dumps({k:v for k,v in payload.items() if k not in ('local_5m_files','feature_coverage','current_frozen_models')},indent=2),
        'Feature availability: '+json.dumps(coverage_result,indent=2),'Current model windows: '+json.dumps(model_ranges,indent=2)]
    (ROOT/'history_boundary_audit.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'local_5m_files':len(intraday),'earliest':payload['local_5m_earliest'],'score_start':sessions[61],
        'training_start':sessions[0],'feature_missing':{k:v['missing'] for k,v in coverage_result.items()}}),flush=True)

if __name__=='__main__':main()

"""O5 bounded measurement: existing ready actions, temporal cash and source support.

No model fits, new native accounts, or strategy policy mutation.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time

REPO = Path(__file__).resolve().parents[1]
BASE = Path('E:/CodexResearch/k_stock_trader')
ALIGN = BASE/'kalcb_selected_target_alignment_20261010/v2'
AUDIT = BASE/'kalcb_late_entry_20261011/v26_opening_decision_audit_v2'
OLD = BASE/'kalcb_lifecycle_readiness_20261016/first_interval_v1'
DESIGN = REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0033.md'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, obj, replace=False):
    with Path(p).open('w' if replace else 'x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def bound(p):
    return {'path': str(Path(p).resolve()), 'sha256': sha(p)}


def verify(b):
    if sha(b['path']) != b['sha256']:
        raise ValueError('Changed source: '+b['path'])


def lines(p):
    opener = gzip.open if str(p).endswith('.gz') else open
    with opener(p, 'rt', encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def stamp(x):
    d = datetime.fromisoformat(x)
    if d.utcoffset() != timedelta(hours=9):
        raise ValueError('Non-Korean source clock')
    return d


def cash_partition(fills, at, close):
    """Exclude same-clock next-open fills from the completed-bar decision state."""
    cut = stamp(at)
    qty = before_qty = 0
    total = before = fees = 0.0
    bought = sold = before_buy = 0
    for f in fills:
        q = int(f['qty']); price = float(f['price']); fee = float(f['metadata']['fee'])
        if q <= 0 or price <= 0 or fee < 0 or f['side'] not in ('BUY', 'SELL'):
            raise ValueError('Invalid fill')
        sign = 1 if f['side'] == 'BUY' else -1
        qty += sign*q
        if qty < 0:
            raise ValueError('Negative inventory')
        value = -sign*q*price-fee
        total += value; fees += fee
        bought += q if sign == 1 else 0; sold += q if sign == -1 else 0
        if stamp(f['timestamp']) < cut:
            before += value; before_qty += sign*q
            before_buy += q if sign == 1 else 0
    marked = before+before_qty*close
    return {'terminal_inventory_reconstructed': qty, 'net_reconstructed_krw': total,
            'fees_reconstructed_krw': fees, 'buy_qty_reconstructed': bought, 'sell_qty_reconstructed': sold,
            'quantity_at35': before_qty, 'buy_qty_before35': before_buy,
            'marked_net_at35_krw': marked, 'remaining_net_after35_krw': total-marked}


def prefix_features(opening, new, day, atr, relvol):
    bars = opening+[new]
    cut = stamp(day+'T09:35:00+09:00')
    if len(bars) != 7 or atr <= 0:
        raise ValueError('Prefix/ATR')
    times = [stamp(b['timestamp']) for b in bars]
    if times[0] != cut-timedelta(minutes=35) or any(t+timedelta(minutes=5)>cut for t in times):
        raise ValueError('Future or incorrectly anchored prefix')
    if any(b-a != timedelta(minutes=5) for a,b in zip(times,times[1:])):
        raise ValueError('Noncontiguous prefix')
    for b in bars:
        if min(b[k] for k in ('open','high','low','close','volume')) <= 0:
            raise ValueError('Nonpositive bar')
        if b['low']>min(b['open'],b['close']) or b['high']<max(b['open'],b['close']):
            raise ValueError('Incoherent OHLC')
    hi=max(b['high'] for b in opening);lo=min(b['low'] for b in opening);c=opening[-1]['close']
    vol=sum(b['volume'] for b in opening)
    pv=sum((b['high']+b['low']+b['close'])/3*b['volume'] for b in opening)
    vwap=pv/vol; vwap35=(pv+(new['high']+new['low']+new['close'])/3*new['volume'])/(vol+new['volume'])
    values=[(c-opening[0]['open'])/atr,(hi-lo)/atr,(c-vwap)/atr,(c-lo)/max(hi-lo,1e-9),
            math.log1p(relvol),atr/c,(new['open']-c)/atr,(new['close']-new['open'])/atr,
            (new['high']-hi)/atr,(new['low']-lo)/atr,
            (new['close']-new['low'])/max(new['high']-new['low'],1e-9),math.log1p(new['volume']/(vol/6))]
    states={'below_vwap':new['close']<vwap35,'below_opening_low':new['close']<lo,
            'above_opening_high':new['close']>hi,'negative_interval':new['close']<new['open'],
            'volume_acceleration':new['volume']>vol/6}
    return values, states


def state_at35(event, at, before_buy):
    """Rebuild pending original BUY orders using only orders available by decision."""
    cut=stamp(at);submitted={};cancelled=set();sell_intent=False
    for e in event['orders']:
        order=e['order'];meta=order.get('metadata',{})
        if e['event'] in ('cancel','cancelled'):
            if not e.get('timestamp'):raise ValueError('Untimed cancellation')
            # Native cancellation callbacks use completed source bar start times.
            available=(stamp(e['timestamp'])+timedelta(minutes=5)).isoformat()
        else:
            available=meta.get('decision_available_at')
            if not available and e['event']=='submit':
                available=(stamp(order['submitted_at'])+timedelta(minutes=5)).isoformat()
        if not available or stamp(available)>cut:
            continue
        oid=order.get('order_id')
        if e['event']=='submit':
            if order['side']=='BUY':submitted[oid]=int(order['qty'])
            else:sell_intent=True
        elif e['event'] in ('cancel','cancelled'):
            cancelled.add(oid)
    # The exact pending flag uses buy fill counts by order, never final fill totals.
    used=Counter()
    for f in event['fills']:
        if f['side']=='BUY' and stamp(f['timestamp'])<cut:used[f['order_id']]+=int(f['qty'])
    pending=sum(max(0,q-used[oid]) for oid,q in submitted.items() if oid not in cancelled)
    return {'pending_qty_at35':pending,'original_exit_intent_by35':sell_intent,
            'submitted_buy_qty_by35':sum(submitted.values())}


def register(out):
    inputs={};jobs=[];prepared={};databases={}
    def add(p, expected=None):
        p=Path(p);key=str(p.resolve())
        b={'path':key,'sha256':expected or sha(p)}
        if key in inputs and inputs[key]['sha256']!=b['sha256']:raise ValueError('Conflicting binding')
        inputs[key]=b
    audit=read(AUDIT/'completion.json')
    b=next(x for x in audit['artifacts'] if Path(x['path']).name=='decisions.jsonl.gz')
    add(b['path'],b['sha256']);add(AUDIT/'completion.json')
    for period in ('is','oos'):
        planpath=ALIGN/(period+'_opening_plan.json');plan=read(planpath);add(planpath)
        db=Path(plan['database']);expected=next(v for k,v in plan['input_sha256'].items() if Path(k)==db)
        add(db,expected);databases[period]=str(db)
        cell=next(x for x in read(ALIGN/'protocol.json')['cells'] if x['kind']=='opening' and x['period']==period)
        p=Path(cell['prepared']);meta=read(p/'prepared.json');add(p/'prepared.json');add(p/'opportunities.jsonl',meta['opportunities_sha256'])
        prepared[period]=str(p/'opportunities.jsonl')
        for scenario in ('baseline','combined'):
            folder=ALIGN/'labels'/(period+'_opening_'+scenario);manifest=read(folder/'grid_manifest.json');add(folder/'grid_manifest.json')
            for day in manifest['sessions']:
                if period=='oos' and day<'2026-04-01':continue
                lp=folder/(day+'.jsonl');ep=folder/(day+'.events.jsonl.gz')
                add(lp,manifest['files'][lp.name]);add(ep,manifest['files'][ep.name])
                jobs.append({'period':period,'scenario':scenario,'day':day,'labels':str(lp),'events':str(ep)})
    for p in [OLD/'registration.json',OLD/'completion.json']:
        add(p)
    for p in OLD.glob('*models.json'):add(p)
    for p in OLD.glob('*predictions.json'):add(p)
    add(ALIGN/'protocol.json')
    reg={'at_utc':datetime.now(timezone.utc).isoformat(),'program':bound(__file__),'design':bound(DESIGN),
         'tests':bound(REPO/'tests/test_kalcb_entry_lifecycle_panel_v1.py'),
         'inputs':list(inputs.values()),'jobs':jobs,'prepared':prepared,'databases':databases,
         'native_cases':0,'fits':0,'stage':'C9-A','evidence_level':'reused-development isolated native lifecycle diagnosis'}
    write(out/'registration.json',reg)
    return reg


def stats(rows):
    days=defaultdict(list)
    valid=[r for r in rows if r['label_valid'] and r.get('prefix_supported')]
    for r in valid:days[r['day']].append(r)
    def sessionmean(key):
        return sum(sum(r[key] for r in rs)/len(rs) for rs in days.values())/len(days) if days else None
    return {'actions':len(rows),'issuer_days':len({r['base_id'] for r in rows}),
            'sessions':len({r['day'] for r in rows}),'valid_with_prefix':len(valid),
            'entry_value_planned_notional_bps':sessionmean('entry_value_planned_bps'),
            'remaining_value_planned_notional_bps':sessionmean('remaining_value_planned_bps'),
            'already_earned_planned_notional_bps':sessionmean('earned_value_planned_bps'),
            'pending_at35':sum(r.get('pending_qty_at35',0)>0 for r in rows),
            'flat_at35':sum(r.get('quantity_at35',0)==0 for r in rows),
            'positive_remaining':sum(r['remaining_net_after35_krw']>0 for r in valid),
            'negative_remaining':sum(r['remaining_net_after35_krw']<0 for r in valid),
            'positive_remaining_mass_reference_krw':sum(max(0,r['remaining_net_after35_krw']) for r in valid),
            'negative_remaining_mass_reference_krw':sum(min(0,r['remaining_net_after35_krw']) for r in valid)}


def run(out):
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    import psutil
    me=psutil.Process()
    write(out/'process.json',{'pid':me.pid,'created_epoch':me.create_time(),'argv':me.cmdline()})
    reg=register(out)
    for i,b in enumerate(reg['inputs']):
        verify(b)
        if i%40==0:write(out/'status.json',{'phase':'authentication','verified':i+1,'total':len(reg['inputs']),'elapsed_s':time.monotonic()-started},True)
    audit={r['action_id']:r for r in lines(AUDIT/'decisions.jsonl.gz')}
    sources={p:{r['opportunity_id']:r for r in lines(f) if p=='is' or r['trade_date']>='2026-04-01'} for p,f in reg['prepared'].items()}
    bar35={}
    for period,db in reg['databases'].items():
        with sqlite3.connect('file:'+Path(db).as_posix()+'?mode=ro',uri=True) as conn:
            for sym,ts,day,o,h,l,c,v in conn.execute("SELECT symbol,ts,day,o,h,l,c,v FROM bars WHERE substr(ts,12,5)='09:30'"):
                key=day+'/'+sym
                if key in sources[period]:bar35[key]={'timestamp':ts,'open':o,'high':h,'low':l,'close':c,'volume':v}
    groups=defaultdict(list);counts=Counter();max_error=0.;row_count=0
    with gzip.open(out/'panel.jsonl.gz','xt',encoding='utf-8') as target:
        for j,job in enumerate(reg['jobs']):
            labels={r['opportunity_id']:r for r in lines(job['labels'])};seen=set()
            for ev in lines(job['events']):
                key=ev['opportunity_id'];base=key.rsplit('/q',1)[0];label=labels[key];source=sources[job['period']][base]
                if key in seen:raise ValueError('Duplicate event')
                seen.add(key);day=job['day'];at=day+'T09:35:00+09:00'
                old=audit.get(key);q=float(label['quantity_scale']);planned=math.floor(int(source['action']['quantity'])*q)
                planned_notional=planned*source['opening_bars'][-1]['close']
                if planned_notional<=0:raise ValueError('Nonpositive planned amount')
                row={'action_id':key,'base_id':base,'day':day,'period':'prior' if day<'2025-08-07' else job['period'],
                     'scenario':job['scenario'],'quantity_scale':q,'planned_qty':planned,'planned_notional':planned_notional,
                     'features30':source['features'],'available_at30':source['feature_available_at'],
                     'selected_prior100':bool(old and 'control' in old['selected']),
                     'actual_prior100':bool(old and old['actual'].get('control/'+job['scenario'])),
                     'original_parent_selected':bool(old and old['original_parent_selected']),
                     'physical_capacity_pass':bool(old and old['physical_risk_pass'] and old['capacity_pass']) if old else None,
                     'positive_core':bool(old and min(old['calibrated_core'].values())>0) if old else None,
                     'forecast_core':old['calibrated_core'][job['scenario']] if old else None,
                     'label_valid':label['label_valid'],'label_available_at':label.get('label_available_at'),
                     'deadline_violation':label['deadline_violation'],'unresolved':label['unresolved'],
                     'native_status':label['status'],'prefix_supported':False}
                if old and (old['reference_quantity']!=planned or abs(old['notional_equity']*1e8-planned_notional)>.01):
                    raise ValueError('Quantity source bridge')
                new=bar35.get(base)
                if new:
                    values,states=prefix_features(source['opening_bars'],new,day,float(source['candidate']['daily_atr']),float(source['features']['first30_rel_volume']))
                    states['prior20_positive']=source['features']['daily_return_20d']>0
                    prevclose=source['opening_bars'][0]['open']/(1+source['features']['first30_gap'])
                    states['below_prior_close']=new['close']<prevclose
                    row.update(prefix_supported=True,features35=values,states=states,close35=new['close'])
                else:counts['missing_prefix']+=1
                close=new['close'] if new else 0.
                part=cash_partition(ev['fills'],at,close);row.update(part)
                error=abs(part['net_reconstructed_krw']-label['net_cash_flow_krw']);max_error=max(max_error,error)
                if error>.02 or abs(part['fees_reconstructed_krw']-label['fees_and_tax_krw'])>.02:
                    raise ValueError('Cash/fees do not reconcile '+key)
                if part['buy_qty_reconstructed']!=label['buy_qty'] or part['sell_qty_reconstructed']!=label['sell_qty'] or part['terminal_inventory_reconstructed']!=label['terminal_inventory']:
                    raise ValueError('Quantity does not reconcile '+key)
                row.update(state_at35(ev,at,part['buy_qty_before35']))
                row['entry_value_planned_bps']=10000*label['net_value_krw']/planned_notional
                row['remaining_value_planned_bps']=10000*part['remaining_net_after35_krw']/planned_notional
                row['earned_value_planned_bps']=10000*part['marked_net_at35_krw']/planned_notional
                target.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
                # Retain small analysis records only; broad source features remain in the panel.
                small={k:v for k,v in row.items() if k not in ('features30','features35')}
                groups[(row['period'],job['scenario'],q)].append(small)
                counts[row['native_status']]+=1;row_count+=1
            if seen!=set(labels):raise ValueError('Missing/extra native events')
            elapsed=time.monotonic()-started
            status={'phase':'extracting','jobs_complete':j+1,'jobs_total':len(reg['jobs']),'rows':row_count,'elapsed_s':elapsed,
                    'last_job':job,'max_cash_error_krw':max_error}
            write(out/'status.json',status,True)
            if j==9:write(out/'pilot_checkpoint.json',{**status,'passed':True,'checkpoint':'integrity/resource only'})
            if (j+1)%20==0:print(json.dumps(status),flush=True)
    assessment={}
    for (period,s,q),rows in groups.items():
        key=f'{period}/{s}/q{q}'
        cohorts={'all_ready':rows,'physical_capacity':[r for r in rows if r['physical_capacity_pass']],
                 'core_rejected':[r for r in rows if r['physical_capacity_pass'] and not r['positive_core']],
                 'positive_not_selected':[r for r in rows if r['physical_capacity_pass'] and r['positive_core'] and not r['selected_prior100']],
                 'selected':[r for r in rows if r['selected_prior100']],
                 'actually_traded':[r for r in rows if r['actual_prior100']]}
        cell={'cohorts':{k:stats(v) for k,v in cohorts.items()},'states':{},'monthly':{}}
        for cohort in ('physical_capacity','selected'):
            rr=cohorts[cohort]
            for state in ('below_vwap','below_opening_low','above_opening_high','negative_interval','volume_acceleration','prior20_positive','below_prior_close'):
                for yes in (False,True):
                    vv=[r for r in rr if r.get('states',{}).get(state)==yes]
                    cell['states'][f'{cohort}/{state}/{yes}']=stats(vv)
            for m in sorted({r['day'][:7] for r in rr}):cell['monthly'][cohort+'/'+m]=stats([r for r in rr if r['day'].startswith(m)])
        assessment[key]=cell
    write(out/'assessment.json',{'cells':assessment,'rows':row_count,'counts':dict(counts),'max_cash_error_krw':max_error,
         'scope':'Repeated quantities/stresses dependent. Isolated native outcomes, not full funded policy. All reused development.',
         'economic_objective_achieved':False,'next_stage':'Support attribution and prespecified information comparison; no automatic candidate.'})
    for b in (reg['program'],reg['design'],reg['tests']):verify(b)
    # Input hashes were authenticated immediately before reads. Post-read metadata is not represented as rehashing.
    write(out/'completion.json',{'complete':True,'at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-started,
          'registration':bound(out/'registration.json'),'panel':bound(out/'panel.jsonl.gz'),'assessment':bound(out/'assessment.json')})
    write(out/'status.json',{'phase':'complete','rows':row_count,'elapsed_s':time.monotonic()-started},True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:run(args.output)
    except Exception as exc:
        if args.output.exists() and not (args.output/'failure.json').exists():
            write(args.output/'failure.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'error':repr(exc)})
        raise

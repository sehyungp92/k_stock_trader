"""Independent saved-ledger and economic attribution, without changing inputs."""
import argparse
from collections import defaultdict,Counter
from datetime import datetime,timezone,timedelta
from functools import lru_cache
import gzip,hashlib,json,math
from pathlib import Path
import sqlite3,sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify
from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
from backtests.strategies.kalcb.population_research_io_v1 import bridge_accounts

def near(a,b,tol=.011):
    if abs(a-b)>tol:raise ValueError(f'Independent saved account differs: {a} vs {b}')


@lru_cache(maxsize=12)
def gz(binding_path,sha):
    verify([{'path':binding_path,'sha256':sha}])
    with gzip.open(binding_path,'rt',encoding='utf-8') as f:return json.load(f)


def check_ledger(record,detail,source,costs,participation):
    """Rebuild branch cash, inventory, costs, capacity and every native MTM mark."""
    from datetime import datetime
    dt=datetime.fromisoformat
    raw={(b['symbol'],dt(b['timestamp'])):b for batch in source for b in batch}
    fills=defaultdict(list);seen=set();used=Counter();fees=0.
    for f in detail['fills']:
        m=f['metadata'];stamp=dt(f['timestamp']);key=(f['order_id'],m['execution_id'])
        if key in seen or type(f['qty']) is not int or f['qty']<=0:raise ValueError('Duplicate or invalid branch fill')
        seen.add(key);b=raw[(f['symbol'],stamp)]
        if any(m['execution_bar'][k]!=b[k] for k in ('open','high','low','close','volume')):raise ValueError('Raw branch execution bar differs')
        if stamp<dt(m['decision_available_at']):raise ValueError('Anticipatory saved action')
        phase=m['execution_phase']
        if phase=='open':
            previous=raw[(f['symbol'],stamp-timedelta(minutes=5))]
            volume=previous['volume'];base=b['open']
        elif phase=='auction':volume=b['volume'];base=b['close']
        else:raise ValueError('Unsupported native research phase')
        near(m['base_price'],base,1e-7)
        price=base*(1+(1 if f['side']=='BUY' else -1)*costs['slippage_bps']/10000)
        near(f['price'],price,1e-7)
        used[(f['symbol'],stamp,phase)]+=f['qty']
        if used[(f['symbol'],stamp,phase)]>math.floor(volume*participation):raise ValueError('Saved branch exceeds causal capacity')
        fee=f['qty']*price*(costs['commission_bps']+(costs['tax_bps_on_sell'] if f['side']=='SELL' else 0))/10000
        near(fee,m['fee'],1e-7);fees+=fee;fills[stamp].append((f,fee))
    cash=record['initial_cash'];quantities=defaultdict(int,record['initial_positions']);marks=dict(record['initial_marks'])
    peak=record['boundary']['running_peak'];dd=max(0.,1-record['boundary']['equity']/peak);gross=[];cash_path=[]
    curve=detail['value']['equity_path']
    future=source[record['source_suffix_offset']:]
    if len(curve)!=len(future):raise ValueError('Truncated branch mark path')
    for (timestamp,equity),batch in zip(curve,future):
        stamp=dt(timestamp)
        if any(dt(b['timestamp'])!=stamp for b in batch):raise ValueError('Changed suffix clock')
        for f,fee in fills.pop(stamp,[]):
            signed=f['qty'] if f['side']=='BUY' else -f['qty'];quantities[f['symbol']]+=signed
            cash-=signed*f['price']+fee
            if quantities[f['symbol']]<0:raise ValueError('Unowned branch sale')
        marks.update({b['symbol']:b['close'] for b in batch})
        inventory=sum(q*marks[s] for s,q in quantities.items() if q);near(cash+inventory,equity)
        peak=max(peak,equity);dd=max(dd,1-equity/peak);gross.append(inventory/equity);cash_path.append(cash)
    if fills or any(quantities.values()) or detail['terminal_pending'] or detail['terminal_inventory']:
        raise ValueError('Unresolved saved branch')
    near(cash,detail['terminal_cash']);near(cash,detail['value']['terminal_equity'])
    near(dd,detail['value']['max_drawdown_from_inherited_peak'],1e-10)
    return {'terminal_cash':cash,'dd':dd,'fees':fees,'marks':len(curve),'fills':len(seen),
        'mean_gross_inventory_fraction':sum(gross)/len(gross),'mean_cash_krw':sum(cash_path)/len(cash_path)}


def review_case(folder,item,scenario,database):
    validate_native(folder,item,scenario,save=False)
    cfg=read(read(item['plan']['path'])['config']);costs={k:cfg[k] for k in ('commission_bps','tax_bps_on_sell')}
    costs['slippage_bps']=13. if scenario in ('cost','combined') else 3.
    participation=.005 if scenario in ('participation','combined') else .01
    control_fills=read(folder/'fills.json');control_equity=read(folder/'equity.json')
    identities={r['day']:r['exact_hold_eod_state'] for r in read(folder/'hold_identity.json')}
    conn=sqlite3.connect('file:'+Path(database).as_posix()+'?mode=ro',uri=True)
    day=None;rows=[];marks=0;prefixes=set();holds=set();reasons=Counter()
    for line in (folder/'observations.jsonl').open(encoding='utf-8'):reasons[json.loads(line)['reason']]+=1
    for line in (folder/'labels.jsonl').open(encoding='utf-8'):
        r=json.loads(line);obs=r['observation'];d=obs['day']
        if d!=day:
            day=d;raw={(str(s),datetime.fromisoformat(t)):(o,h,l,c,v) for s,t,o,h,l,c,v in conn.execute(
                'SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day=?',(d,))}
            source=gz(**{'binding_path':r['source']['path'],'sha':r['source']['sha256']})
            for batch in source:
                for b in batch:
                    if tuple(b[k] for k in ('open','high','low','close','volume'))!=raw[(b['symbol'],datetime.fromisoformat(b['timestamp']))]:
                        raise ValueError('Saved branch source differs from bound raw database')
        if identities.get(d)!=r['hold_eod_state']:raise ValueError('Unverified EOD hold identity')
        pre=gz(r['prefix']['path'],r['prefix']['sha256'])
        digest=hashlib.sha256(json.dumps(pre['state'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if digest!=r['boundary']['state_sha256'] or digest!=pre['sha256']:raise ValueError('Changed account boundary')
        prefixes.add(r['prefix']['sha256']);hold=gz(r['hold']['path'],r['hold']['sha256'])
        if r['hold']['sha256'] not in holds:
            hv=check_ledger(r,hold,source,costs,participation);marks+=hv['marks'];holds.add(r['hold']['sha256'])
            # Exact disabled suffix against the independently qualified account.
            n=r['boundary']['marks'];curve=[e for e in control_equity[n:] if e[0][:10]==d]
            if [(datetime.fromisoformat(t),v) for t,v in curve]!=[(datetime.fromisoformat(t),v) for t,v in hold['value']['equity_path']]:
                raise ValueError('Saved hold differs from full sequential MTM')
            expected=[f for f in control_fills[r['boundary']['fills']:] if f['timestamp'][:10]==d]
            normalize=lambda fs:[{**f,'timestamp':datetime.fromisoformat(f['timestamp']).isoformat()} for f in fs]
            if normalize(expected)!=normalize(hold['fills']):raise ValueError('Saved hold differs from full sequential fills')
        else:hv={'terminal_cash':hold['terminal_cash'],'dd':hold['value']['max_drawdown_from_inherited_peak']}
        for arm,c in r['contrasts'].items():
            detail=gz(c['ledger']['path'],c['ledger']['sha256']);v=check_ledger(r,detail,source,costs,participation);marks+=v['marks']
            value=(v['terminal_cash']-hv['terminal_cash'])/r['boundary']['equity']*10000
            dd=(v['dd']-hv['dd'])*10000;near(value,c['target']['value_bps'],1e-7);near(dd,c['target']['dd_change_bps'],1e-7)
            rows.append({'day':d,'symbol':obs['symbol'],'clock':obs['available_at'],'entry_order_id':obs['entry_order_id'],
                'age_minutes':obs['features']['age_minutes'],'qty':obs['qty'],'arm':arm,**c['target'],
                'first_fill':c['first_fill'],'last_fill':c['last_fill'],'mean_gross_inventory_fraction':v['mean_gross_inventory_fraction']})
    conn.close()
    return {'passed':True,'labels':len(rows)//2,'prefixes':len(prefixes),'marks':marks,'observations':dict(reasons),'actions':rows}


def main(root):
    p=read(root/'protocol.json');done=read(root/'completion.json')
    if not done['complete']:raise ValueError('Native study incomplete')
    verify(p['programs']+p['bindings']+p['database_bindings']);out=root/'independent_assessment_v1';out.mkdir(exist_ok=False)
    summaries={};all_rows=[]
    for period in ('is','oos'):
        for scenario in p['scenarios']:
            folder=root/'cases/control'/period/scenario;item=read(folder/'input_profile.json')
            bridge=bridge_accounts(folder,Path(p['references'][period][scenario]))
            if not bridge['passed']:raise ValueError('Independent parent control bridge failed')
            checked=review_case(folder,item,scenario,p['databases'][int(period=='oos')])
            write(out/(period+'_'+scenario+'.json'),checked)
            summaries[period+'/'+scenario]={k:v for k,v in checked.items() if k!='actions'}
            all_rows.extend({**r,'period':period,'scenario':scenario} for r in checked['actions'])
    if done['economic_readiness']:
        for period in ('is','oos'):
            for scenario in p['scenarios']:
                folder=root/'cases/selective'/period/scenario
                validate_native(folder,read(folder/'input_profile.json'),scenario,save=False)
    result={'passed':True,'control_and_suffix_checks':summaries,'execution_completion':done,
        'largest_harmful_local_actions':sorted(all_rows,key=lambda r:r['value_bps'])[:50],
        'largest_helpful_local_actions':sorted(all_rows,key=lambda r:-r['value_bps'])[:50],
        'local_values_are_not_additive_strategy_returns':True,'reused_OOS_is_development':True,
        'economic_frontier':read(root/'account_frontier.json'),'models':bound(root/'models/manifest.json')}
    write(out/'verification.json',result)
    write(out/'receipt.json',{'complete':True,'report':bound(out/'verification.json'),'program':bound(__file__),
        'thorough_final_interpretation_still_required':True,'at_utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();main(args.root)

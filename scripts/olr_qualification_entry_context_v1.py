"""Final bounded entry-context postmortem; no financial replay or candidate rule."""
from collections import defaultdict
from datetime import datetime,timedelta,timezone
from statistics import median,mean
from pathlib import Path
import argparse,json,sqlite3,yaml
from olr_qualification_disposition_v1 import read,write,sha


def group(rows, field, value):
    result={}
    for g in sorted({r[field] for r in rows}):
        members=[r for r in rows if r[field]==g];days=defaultdict(list)
        for r in members:days[r['day']].append(r[value])
        date_values={d:mean(v) for d,v in days.items()}
        result[g]=dict(n=len(members),dates=len(days),net_pnl=sum(r['pnl'] for r in members),
            winners=sum(r['pnl']>0 for r in members),losers=sum(r['pnl']<0 for r in members),
            date_equal_bps=mean(date_values.values()),
            excluding_largest_absolute_date=mean(v for d,v in date_values.items() if d!=max(date_values,key=lambda d:abs(date_values[d]))) if len(days)>1 else None,
            months={m:dict(n=sum(r['day'][:7]==m for r in members),date_equal_bps=mean(v for d,v in date_values.items() if d[:7]==m)) for m in sorted({d[:7] for d in days})})
    return result


def run(root):
    out=root/'qualification_disposition_v1'/'entry_context_v1';out.mkdir(exist_ok=False)
    repo=Path(__file__).resolve().parents[1];design=repo/'docs/optimisation/olr/experiments/olr-exp-0036.md'
    (out/'design_at_freeze.md').write_bytes(design.read_bytes());(out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    a=read(root/'assessment.json');lane_path=root.parent/'context_population_v1/entry_rows.json';sources={str(lane_path):sha(lane_path)}
    for w in ['train','oos']:
        ready=read(root/'native_engines'/w/'ready.json');files=Path(ready['files']);case=Path(a['summaries'][w+'_nominal_control']['case'])
        for p in [root/'native_engines'/w/'ready.json',files/ready['study']['database'],files/'kis_core/data/krx_sessions.yaml',case/'trades.json',case/'funded_population_receipt.json']:
            sources[str(p)]=sha(p)
    write(out/'protocol.json',dict(at=datetime.now(timezone.utc).isoformat(),sources=sources,script_sha256=sha(__file__),design_sha256=sha(design),
        method='Same natural-zero own/market complete-prefix representation as0034 at actual entry availability; all nominal acquired trades and existing six untraded lanes. No threshold fitting.'))
    rows=[];unsupported=[]
    for w in ['train','oos']:
        ready=read(root/'native_engines'/w/'ready.json');files=Path(ready['files']);case=Path(a['summaries'][w+'_nominal_control']['case'])
        trades=read(case/'trades.json');pop=read(case/'funded_population_receipt.json')['entry_contexts']
        universe={str(s).zfill(6) for s in yaml.safe_load((files/'config/universe_103.yaml').read_text(encoding='utf8'))['symbols']}
        calendar=yaml.safe_load((files/'kis_core/data/krx_sessions.yaml').read_text(encoding='utf8'))
        con=sqlite3.connect((files/ready['study']['database']).as_uri()+'?mode=ro',uri=True);con.execute('PRAGMA query_only=ON')
        by_day=defaultdict(list)
        for t in trades:by_day[t['entry_decision_time'][:10]].append(t)
        for day,ts in sorted(by_day.items()):
            bars=defaultdict(list)
            for s,stamp,o,c,v in con.execute('SELECT symbol,ts,o,c,v FROM bars WHERE day=? ORDER BY ts',(day,)):
                if s in universe:bars[s].append((stamp,o,c,v))
            clock=calendar.get('exceptions',{}).get(day,{}).get('open',calendar['regular']['open'])
            opening=datetime.fromisoformat(day+'T'+clock+':00+09:00')
            for t in ts:
                label=datetime.fromisoformat(t['route_metadata']['entry_submission_time']);at=label+timedelta(minutes=5)
                proposals=[p for p in pop if p['at']==at.isoformat() and any(x['symbol']==t['symbol'] for x in p['proposals'].values())]
                assert proposals,(w,t['symbol'],at)
                expected=[(opening+timedelta(minutes=5*i)).isoformat() for i in range(int((label-opening).total_seconds()/300)+1)]
                own={}
                for s,bs in bars.items():
                    prefix=[b for b in bs if b[0]<=label.isoformat()]
                    if [b[0] for b in prefix]!=expected or sum(b[3] for b in prefix)<=0:continue
                    own[s]=dict(opening=prefix[0][1],close=prefix[-1][2],ret=prefix[-1][2]/prefix[0][1]-1)
                peers=[x['ret'] for s,x in own.items() if s!=t['symbol']]
                if t['symbol'] not in own or len(peers)<68:
                    unsupported.append(dict(window=w,symbol=t['symbol'],day=day,peers=len(peers)));continue
                x=own[t['symbol']];m=median(peers)
                assert x['close']==t['route_metadata']['entry_submission_close']
                rows.append(dict(window=w,symbol=t['symbol'],day=day,available_at=at.isoformat(),source_last_label=label.isoformat(),
                    own_return=x['ret'],market_median=m,peer_count=len(peers),
                    market='decline' if m<0 else 'nondecline',state=('own_decline' if x['ret']<0 else 'own_nondecline')+'/'+('market_decline' if m<0 else 'market_nondecline'),
                    rank=t['route_metadata']['candidate_rank'],score=t['route_metadata']['candidate_score'],pnl=t['net_pnl'],
                    acquired_notional=t['qty']*t['entry_price'],notional_bps=1e4*t['net_pnl']/(t['qty']*t['entry_price'])))
        con.close()
    lane_rows=[]
    for r in read(lane_path):
        c=r['context']
        if r['incumbent_disposition']!='recorded_admission_without_fill' or not c.get('available') or c['market_peers']<68:continue
        lane_rows.append(dict(day=r['day'],symbol=r['symbol'],scenario=r['scenario'],budget=r['budget'],pnl=r['pnl'],request_bps=r['request_bps'],
            market='decline' if c['market_peer_median']<0 else 'nondecline'))
    groups={w:dict(market=group([x for x in rows if x['window']==w],'market','notional_bps'),
        interaction=group([x for x in rows if x['window']==w],'state','notional_bps')) for w in ['train','oos']}
    lanes={f'{s}_{b}':group([x for x in lane_rows if x['scenario']==s and x['budget']==b],'market','request_bps') for s,b in sorted({(x['scenario'],x['budget']) for x in lane_rows})}
    write(out/'assessment.json',dict(at=datetime.now(timezone.utc).isoformat(),rows=rows,unsupported=unsupported,groups=groups,unfilled_lanes=lanes,
        limits=['Acquired notional returns and requested-budget isolated returns have different denominators.',
                'Same-date issuers/lanes and adaptively reused histories are dependent observations.',
                'Entry contexts are observed before orders; outcomes include the original whole lifecycle and are not funded alternative policies.',
                'Unconfirmed/outside-slot names are not covered by the six confirmed-opportunity lanes.']))
    print(json.dumps(dict(output=str(out),supported=len(rows),unsupported=unsupported,groups=groups,unfilled_lanes=lanes)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();run(args.root)

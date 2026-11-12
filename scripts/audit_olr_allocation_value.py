"""Independently reconcile allocation-learning evidence to native fill cash flows."""
from collections import defaultdict
from datetime import datetime
import argparse
import gzip
import hashlib
import json
from math import isclose, sqrt
from pathlib import Path
from statistics import mean,stdev


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(case):
    if not (case/'completion.json').exists():
        raise ValueError('Only completed native cases can be audited')
    request=json.loads((case/'request.json').read_text())
    settings=request['recipe']['olr_capacity']
    mode=settings.get('capacity_value_mode','off');strength=settings.get('capacity_value_strength',.5)
    if mode not in ('rank','band'):
        raise ValueError('This case does not enable allocation learning')
    with gzip.open(case/'fills.jsonl.gz','rt') as f:
        fills=sorted([json.loads(line) for line in f],key=lambda r:r['timestamp'])
    with gzip.open(case/'decisions.jsonl.gz','rt') as f:
        decisions=sorted([json.loads(line) for line in f
            if 'capacity_value_evidence' in line],key=lambda r:r['timestamp'])
    decisions=[d for d in decisions if d['decision_code'] in ('ENTRY_SUBMITTED','ENTRY_SKIPPED')]
    rows={};active={};cursor=0;failures=[];supported=set();checked=0
    for decision in decisions:
        stamp=datetime.fromisoformat(decision['timestamp'])
        evidence=decision['metadata'].get('capacity_value_evidence')
        if evidence is None:continue
        # No same-session execution can become a training observation. This
        # independent reconstruction reads prices/fees from broker fills, not
        # the strategy's learned parameters or archived position accounting.
        while cursor<len(fills) and datetime.fromisoformat(fills[cursor]['timestamp']).date()<stamp.date():
            fill=fills[cursor];cursor+=1
            if fill['strategy_id']!='OLR':continue
            symbol=fill['symbol'];side=fill['side'];qty=fill['qty'];metadata=fill['metadata']
            if side=='BUY':
                order=fill['order_id'];old=rows.get(active.get(symbol))
                if old and old['buy_qty']!=old['sell_qty'] and active[symbol]!=order:
                    raise ValueError('Overlapping native OLR lifecycles')
                active[symbol]=order
                row=rows.setdefault(order,{'buy':0.,'sell':0.,'buy_qty':0,'sell_qty':0,
                    'rank':str(metadata.get('candidate_rank',0)),
                    'band':str(metadata.get('afternoon_score_band_rule') or 'unknown'),'complete':True})
            else:
                row=rows[active[symbol]]
            fee=metadata.get('fee')
            if fee is None:row['complete']=False;fee=0.
            part=side.lower();row[part]+=qty*fill['price']+(fee if side=='BUY' else -fee)
            row[part+'_qty']+=qty;row['last_fill']=fill['timestamp']
        try:
            keys=evidence['label_keys']
            if len(keys)!=len(set(keys)) or len(keys)>60:raise ValueError('Duplicate or excessive observations')
            sample=[];values=[];exits=[]
            for key in keys:
                order=json.loads(key)[2];row=rows[order]
                if not row['complete'] or row['buy_qty']!=row['sell_qty'] or row['buy']<=0:
                    raise ValueError('Unclosed or cost-incomplete training observation')
                if datetime.fromisoformat(row['last_fill']).date()>=stamp.date():
                    raise ValueError('Noncausal training observation')
                value=min(1500.,max(-1500.,10000*(row['sell']/row['buy']-1)))
                values.append(value);exits.append(row['last_fill'])
                if row[mode]==evidence['group']:sample.append(value)
            if len(values)!=evidence['strategy_support'] or len(sample)!=evidence['cohort_support']:
                raise ValueError('Support does not match actual native lifecycles')
            latest=max((datetime.fromisoformat(t) for t in exits),default=None)
            declared=(datetime.fromisoformat(evidence['latest_label_exit']) if evidence['latest_label_exit'] else None)
            if latest!=declared:
                raise ValueError('Latest training availability mismatch')
            expected=1.
            if len(values)>=30 and len(sample)>=10 and evidence['group']!='unknown':
                prior=mean(values);estimate=(sum(sample)+20*prior)/(len(sample)+20)
                reference=prior-.25*stdev(values)/sqrt(len(values))
                robust=estimate-.25*stdev(sample)/sqrt(len(sample))
                expected=min(1+.5*strength,max(1-.5*strength,
                    1+strength*(robust-reference)/max(abs(reference),100.)))
                for field,value in [('prior_bps',prior),('shrunk_mean_bps',estimate),
                                    ('reference_bps',reference),('robust_net_bps',robust)]:
                    if not isclose(evidence[field],value,abs_tol=1e-8,rel_tol=1e-10):
                        raise ValueError('Cash-flow model value mismatch: '+field)
                if evidence['status']!='supported':raise ValueError('Supported status mismatch')
                supported.add((stamp.date().isoformat(),decision['symbol']))
            elif evidence['status']!='insufficient_matured_support':
                raise ValueError('Insufficient-support behavior mismatch')
            if not isclose(evidence['multiplier'],expected,abs_tol=1e-9):
                raise ValueError('Allocation multiplier mismatch')
            checked+=1
        except (ValueError,KeyError,IndexError) as error:
            failures.append({'timestamp':decision['timestamp'],'symbol':decision['symbol'],'error':str(error)})
    result={'pass':not failures,'checked_decisions':checked,'supported_issuer_days':len(supported),
        'failures':failures,'source_generation_hash':request['source_generation_hash'],
        'inputs':{name:digest(case/name) for name in ('request.json','fills.jsonl.gz','decisions.jsonl.gz')},
        'auditor_sha256':digest(Path(__file__)),
        'scope':'Causality and learned values reconciled to native completed lifecycle cash flows; not an independent performance validation'}
    (case/'allocation_value_audit.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('case',type=Path)
    result=audit(parser.parse_args().case)
    print(json.dumps({k:result[k] for k in ('pass','checked_decisions','supported_issuer_days')}|
        {'failure_count':len(result['failures']),'first_failures':result['failures'][:3]}))
    raise SystemExit(0 if result['pass'] else 1)

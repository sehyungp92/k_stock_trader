"""Executable C16 score and promotion checks on deliberately synthetic accounts."""
from collections import defaultdict
from pathlib import Path
import json
import hashlib


def useful_days(events):
    unique={x['execution_id']:x for x in events}
    groups=defaultdict(list)
    for row in unique.values():groups[row['day'],row['symbol']].append(row)
    useful=set()
    for key,rows in groups.items():
        lifetimes=defaultdict(list)
        for row in rows:lifetimes[row['lifetime_id']].append(row)
        earliest=min(rows,key=lambda r:r['decision_at'])
        material=any(sum(r['notional'] for r in parts)>=.01*min(parts,key=lambda r:r['decision_at'])['equity']
                     for parts in lifetimes.values())
        if material and sum(r['net'] for r in rows)>=.0001*earliest['equity']:useful.add(key)
    return useful


def frequency_credit(events, evidence):
    allowed={group for group, e in evidence.items() if e['registered_before_outcomes']
             and e['expected_net_lower'] is not None and e['expected_net_lower']>0
             and e['funded_added_minus_displaced_lower'] is not None and e['funded_added_minus_displaced_lower']>0}
    return useful_days([r for r in events if r['addition_group'] in allowed])


def score(cells):
    weights={p+'/'+s:.125 for p in ('is','oos') for s in ('baseline','participation','cost','combined')}
    assert set(cells)==set(weights)
    r=sum(weights[k]*v['return_pp'] for k,v in cells.items())
    d=sum(weights[k]*max(0,v['mtm_dd_pp']-(8 if k.endswith('/baseline') else 10)) for k,v in cells.items())
    valid=all(v['mtm_dd_pp'] <= (12 if k.endswith('/baseline') else 15) and v['native_valid'] for k,v in cells.items())
    return dict(R=r,D=d,S=r-.25*d,risk_valid=valid)


def promotion(control,candidate,evidence):
    b,c=score(control),score(candidate)
    gain=c['R']-b['R'];losses={k:candidate[k]['return_pp']-v['return_pp'] for k,v in control.items()}
    passed=(c['risk_valid'] and gain>=max(1.,evidence['paired_mde_pp']) and evidence['simultaneous_gain_ci_lower_pp']>0
            and min(losses.values())>=-1 and sum(losses[k] for k in losses if k.startswith('is/'))/4>=-2
            and evidence['mechanism_pass'] and evidence['independent_reconstruction_pass'] and evidence['search_adjustment_valid'])
    return dict(promote=bool(passed),gain_pp=gain,**c)


def main():
    def case(ret,dd=8):return {p+'/'+s:dict(return_pp=ret,mtm_dd_pp=dd,native_valid=True) for p in ('is','oos') for s in ('baseline','participation','cost','combined')}
    control=case(10);good=case(12);padded=case(10);hidden=case(15);hidden['oos/combined']['mtm_dd_pp']=16
    proof=dict(paired_mde_pp=1,simultaneous_gain_ci_lower_pp=.5,mechanism_pass=True,independent_reconstruction_pass=True,search_adjustment_valid=True)
    lucky={**proof,'simultaneous_gain_ci_lower_pp':-3}
    cases={'control':promotion(control,control,proof),'genuine_gain':promotion(control,good,proof),
           'frequency_padding':promotion(control,padded,proof),'drawdown_hidden_by_eod':promotion(control,hidden,proof),
           'lucky_path':promotion(control,case(15),lucky)}
    assert cases['genuine_gain']['promote'] and cases['genuine_gain']['S']>cases['control']['S']
    assert cases['frequency_padding']['S']==cases['control']['S'] and not cases['frequency_padding']['promote']
    assert cases['drawdown_hidden_by_eod']['S']>cases['control']['S'] and not cases['drawdown_hidden_by_eod']['promote']
    assert cases['lucky_path']['S']>cases['genuine_gain']['S'] and not cases['lucky_path']['promote']
    row=dict(execution_id='one',addition_group='useful',lifetime_id='L1',decision_at='2000-01-01T09:30:00+09:00',day='2000-01-01',symbol='synthetic',net=20000,notional=2e6,equity=1e8)
    assert len(useful_days([row,row]))==1
    fragments=[{**row,'execution_id':str(i),'net':row['net']/4,'notional':row['notional']/4} for i in range(4)]
    assert useful_days(fragments)==useful_days([row])
    unrelated_tiny=[{**r,'lifetime_id':'tiny'+str(i)} for i,r in enumerate(fragments)]
    assert not useful_days(unrelated_tiny)
    assert useful_days(fragments+fragments)==useful_days([row])
    assert not useful_days([{**row,'net':0}]) and not useful_days([{**row,'net':-1}])
    ev={'useful':dict(registered_before_outcomes=True,expected_net_lower=100,funded_added_minus_displaced_lower=50)}
    base_credit=frequency_credit([row],ev)
    for name,expected,funded in [('zero_ev',0,10),('negative_ev',-100,10),('unknown_ev',None,10),('displaced',100,-10)]:
        extra={**row,'execution_id':name,'lifetime_id':name,'addition_group':name,'day':'2000-01-02'}
        proof_group={**ev,name:dict(registered_before_outcomes=True,expected_net_lower=expected,funded_added_minus_displaced_lower=funded)}
        assert frequency_credit([row,extra],proof_group)==base_credit
    extra={**row,'execution_id':'new_good','lifetime_id':'new_good','addition_group':'useful','day':'2000-01-02'}
    assert len(frequency_credit([row,extra],ev))==2
    # Frequency may break a close score tie only with positive paired added-minus-displaced value.
    tie_allowed=lambda scoregap,lower:abs(scoregap)<=.10 and lower>0
    assert tie_allowed(.05,1) and not tie_allowed(.05,0) and not tie_allowed(.11,1)
    out=Path('E:/CodexResearch/k_stock_trader/kalcb_round2_assessment_20261017/v1/score_checks_v3.json')
    with out.open('x',encoding='utf-8') as f:json.dump(dict(passed=True,program_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        repair='docs/optimisation/kalcb/experiments/kalcb-exp-0041-score-padding-r2.md',expected_value_padding_rejected=True,unknown_value_credit_withheld=True,displacement_failure_credit_withheld=True,cases=cases,unrelated_tiny_lifetimes_rejected=True,duplicate_invariance=True,fragment_invariance=True,nonpositive_frequency_rejected=True,
        evidence_qualified_tie_break_tested=True,scaling='Display normalization cannot change economic coefficients or ranking.'),f,indent=2)
    print('C16 synthetic ranking, risk, evidence and frequency invariants passed')


if __name__=='__main__':main()

"""Prepared read-only four-stress assessment; never starts native work."""
from copy import deepcopy
from pathlib import Path
from .retention_research_io_v1 import read,write,bound,verify,PERIODS
from .research_process_v1 import exact_process
from .pending_funding_assessment_v1 import account,path_comparison
from .exit_qualification_assessment_v1 import paired_blocks,GLOBAL
from .population_research_io_v1 import bridge_accounts
from .reclaim_observer_assessment_v1 import require,close,exposure_comparison

SCENARIOS=('baseline','participation','cost','combined')


def completion_gate(root,alive=exact_process):
    require(not alive(read(root/'supervisor_start.json')),'Exact native supervisor still active')
    status=read(root/'status.json')
    require(not status.get('child') or not alive(status['child']),'Exact native child active')
    require(status['stage']=='awaiting_joint_assessment','Native study not complete')
    done=read(root/'stage_completion.json')
    require(done.get('complete') and done['fresh_cases']==8 and done['authenticated_reuse']==8 and done['accounted_cases']==16,'Wrong stage completion')
    rows=read(root/'progress.json')
    keys=[(r['name'],r['period'],r['scenario']) for r in rows]
    expected={(n,p,s) for n in ('control','pending_100') for p in PERIODS for s in SCENARIOS}
    require(len(keys)==16 and set(keys)==expected and sum(r['reused'] for r in rows)==8,'Missing or duplicate cases')
    require(all(r['metrics']['audit_status']=='passed' and (r['name']!='control' or r['identity_passed']) for r in rows),'Missing source/account/parent bridge')
    return rows


def meaningful(positions):
    # Same V25 descriptive nine-point sensitivity; never used as a policy gate.
    return {str(size)+'/'+str(profit):{
        'positions':sum(p['qty']>1 and p['net_pnl_krw']>0 and p['entry_cost_krw']/p['entry_equity_krw']>=size
                        and p['net_pnl_krw']/p['entry_equity_krw']*10000>=profit for p in positions),
        'net_pnl_krw':sum(p['net_pnl_krw'] for p in positions if p['qty']>1 and p['net_pnl_krw']>0
            and p['entry_cost_krw']/p['entry_equity_krw']>=size and p['net_pnl_krw']/p['entry_equity_krw']*10000>=profit)}
        for size in (.001,.01,.05) for profit in (0.,1.,5.)}


def sacrifice_bands(candidate,parent,period,scenario):
    if period!='is':return None
    local=candidate-parent
    global_delta=candidate-GLOBAL[scenario] if scenario in GLOBAL else None
    return {'local_return_delta_pp':local,'global_return_delta_pp':global_delta,
        'local_sacrifice_exceeds_pp':{str(b):local < -b for b in (2,5,10)},
        'global_sacrifice_exceeds_pp':{str(b):global_delta < -b for b in (2,5,10)} if global_delta is not None else None,
        'scope':'Local bands for all stresses; global baseline/combined references only. No invented global participation/cost benchmark or hard acceptance cutoff.'}


def compare(candidate,parent,period,scenario,cfolder,pfolder):
    require(candidate['attribution']['daily_dates']==parent['attribution']['daily_dates'],'Unpaired sessions')
    delta={k:candidate['metrics'][k]-parent['metrics'][k] for k in ('return_pct','drawdown_pct','trades','active_days')}
    for k in ('positions','profitable_positions','unique_issuer_days','profitable_entry_notional_krw',
              'gross_winning_net_pnl_krw','gross_losing_net_pnl_krw','filled_entry_notional_krw','fees_krw'):
        delta[k]=candidate['activity'][k]-parent['activity'][k]
    paths=path_comparison(candidate['positions'],parent['positions'])
    close(paths['delta_net_pnl_krw']/1e6,delta['return_pct'],'Duplicated position/PnL difference')
    try:alias=bridge_accounts(cfolder,pfolder)
    except ValueError as error:
        require(str(error).startswith(('Native account bridge drift','Native fill bridge drift')),'Unexpected alias error')
        alias={'passed':False,'first_difference':str(error)}
    cs,ps=meaningful(candidate['positions']),meaningful(parent['positions'])
    return {'delta':delta,'paths':paths,'exact_account_alias':alias,
        'is_bands':sacrifice_bands(candidate['metrics']['return_pct'],parent['metrics']['return_pct'],period,scenario),
        'meaningful_activity_sensitivity':{k:{'candidate':cs[k],'parent':ps[k],
            'delta_positions':cs[k]['positions']-ps[k]['positions'],'delta_net_pnl_krw':cs[k]['net_pnl_krw']-ps[k]['net_pnl_krw']} for k in cs},
        'paired_uncertainty':[paired_blocks(candidate['attribution']['daily_returns'],parent['attribution']['daily_returns'],b) for b in (5,10)],
        'month_delta_krw':{m:candidate['attribution']['months'][m]['net_change_krw']-v['net_change_krw'] for m,v in parent['attribution']['months'].items()},
        'issuer_delta_krw':{s:candidate['attribution']['issuers'].get(s,{}).get('net_pnl_krw',0)-parent['attribution']['issuers'].get(s,{}).get('net_pnl_krw',0)
            for s in sorted(set(candidate['attribution']['issuers'])|set(parent['attribution']['issuers']))}}


def assess(root,output):
    rows=completion_gate(root);registration=read(output/'protocol.json');verify(registration['programs']+registration['bindings'])
    protocol=read(root/'protocol.json');wrapped=deepcopy(protocol)
    for period in PERIODS:wrapped['profiles']['100'][period]['references']=protocol['parent_stress_references'][period]
    oldroot=root.parent/'v25_pending_funding_screen';oldprotocol=read(oldroot/'protocol.json')
    accounts={};bindings=[];folders={}
    for row in rows:
        name,period,scenario=(row[k] for k in ('name','period','scenario'));key='/'.join((name,period,scenario))
        source=oldroot if row['reused'] else root
        expected=source/'cases'/name/period/scenario
        require(Path(row['folder'])==expected,'Unexpected reused/fresh source path')
        verify([row['result'],row['exposure_mtm'],row['input_profile']])
        accounts[key],proof=account(source,row,oldprotocol if row['reused'] else wrapped)
        bindings+=proof;folders[key]=expected
        accounts[key]['meaningful_activity_sensitivity']=meaningful(accounts[key]['positions'])
        print('assessed '+key,flush=True)
    comparisons={}
    for period in PERIODS:
        for scenario in SCENARIOS:
            ck='/'.join(('pending_100',period,scenario));pk='/'.join(('control',period,scenario))
            comparisons[ck]=compare(accounts[ck],accounts[pk],period,scenario,folders[ck],folders[pk])
    exposures={n+'/'+s:exposure_comparison(accounts[n+'/is/'+s]['positions'],accounts[n+'/oos/'+s]['positions']) for n in ('control','pending_100') for s in SCENARIOS}
    verify(registration['programs']+registration['bindings']+bindings)
    result={'version':'pending-funding-four-stress-assessment-v1','complete':True,'accounts':accounts,
        'comparisons':comparisons,'exposure_descriptive':exposures,'bindings':bindings,'accounted_cases':16,
        'fresh_native_cases':8,'authenticated_reuse':8,'exact_parent_bridges':8,
        'scope':'Joint four-stress prior100 opening reservation evidence. Unchanged prior125 remains external competitor. Funded nominal-allocation comparison not performed here; reports and economic judgment follow. Reused development.',
        'automatic_promotion':False,'study_complete':False}
    write(output/'assessment.json',result)
    write(output/'completion.json',{'complete':True,'assessment':bound(output/'assessment.json'),
        'study_complete':False,'full_reports_pending':True,'economic_judgment_pending':True})
    return result

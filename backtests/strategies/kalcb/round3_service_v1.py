"""Finite C17 evaluation/qualification service; no optimiser or adoption loop."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
import json
import shutil
import numpy as np
from backtests.auto.shared.cache_keys import stable_signature
from .round3_native_io_v1 import ROOT,SOURCE,profile_for,read,write,bound
from .round3_native_tasks_v1 import execute
from .round3_evidence_v1 import seal,authenticate
from .round3_contract_v1 import SCENARIOS,CAPITALS,scalar,qualify,selectable_score,verify_acceptance
from .round3_analysis_v1 import account,compare,native_reach,finer_marks,funding_chronology,funded_decomposition
from .late_artifact_validation_v2 import validate_case


class RoundService:
    def __init__(self,registration):
        self.registration=Path(registration);self.registration_data=read(registration)
        self.out=self.registration.parent;self.runtime=ROOT/'native_v5'
        self.profiles=read(self.runtime/'preparation.json')['profiles'];self.arms=read(self.runtime/'arms.json')
        bridge=read(ROOT/'preflight_v2/bridge_completion.json')
        if bridge.get('passed') is not True or len(bridge['receipts'])!=8:raise ValueError('Incomplete baseline bridge')
        self.control_receipts={r['item']['period']+'/'+r['process']['scenario']:r for r in bridge['receipts']}
        self.control={k:account(r) for k,r in self.control_receipts.items()}
        self.reach=read(self.registration_data['reach']['path'])
        if self.reach.get('complete') is not True:raise ValueError('Incomplete native reach preflight')
        self.required_policy=dict(frozen_runtime=bound(self.runtime/'preparation.json'),
            initial_control='C12-B_VETO_plus_cap',score=bound(ROOT/'freeze/score.json'),
            qualification=bound(ROOT/'freeze/qualification.json'),all_history_consumed=True)
        self.cost_policy={k:read(Path(r['folder'])/'execution_audit.json')['audit_receipt']['qualification']['capacity']['policy']
            for k,r in self.control_receipts.items()}
        self._receipts={'v6_h0':self.control_receipts};self._raw={'v6_h0':self.control};self._evaluations={}
        reused=read(self.registration_data['reused_controls']['path'])
        for name,cases in reused['controls'].items():
            self._receipts[name]={k:read(b['path']) for k,b in cases.items()}
            for b in cases.values():
                if bound(b['path'])['sha256']!=b['sha256']:raise ValueError('Changed reused receipt')
            self._raw[name]={k:account(r) for k,r in self._receipts[name].items()}
        baseline=dict(native_valid=True,accepted_control=True,score=scalar(self.control),
            qualification=dict(promote=False,failures=['accepted_control_not_new_improvement']),
            receipt=str(self.out/'evaluations/v6_h0.json'))
        self._evaluations['v6_h0']=baseline
        if not (self.out/'evaluations/v6_h0.json').exists():
            seal(self.out/'evaluations/v6_h0.json',baseline,list(self.control_receipts.values()),[bound(self.registration)])
        authenticate(self.out/'evaluations/v6_h0.json')

    def check_frozen(self):
        for binding in self.registration_data['programs']+[self.registration_data[k] for k in ('reach','freeze','reused_controls','capital_controls')]:
            if bound(binding['path'])['sha256']!=binding['sha256']:raise ValueError('Frozen controller/evidence changed '+binding['path'])

    def name_for_config(self,config):
        matches=[n for n,a in self.arms.items() if (a['selectable'] or n=='v6_h0') and a['config']==config and a['mode']=='veto']
        if len(matches)!=1:raise ValueError('Unregistered or ambiguous effective selectable configuration')
        return matches[0]

    def reached(self,name):
        return name=='v6_h0' or self.reach['arms'].get(name,{}).get('eligible_for_full_evaluation') is True

    def _batch(self,name,capitals=(100000000,),delay=0,progress=None):
        self.check_frozen();jobs=[];result={}
        for p in ('is','oos'):
            for s in SCENARIOS:
                for capital in capitals:
                    item=self.profiles[p][name+'/'+str(capital)]
                    folder=self.out/'accounts'/name/p/s/str(capital)/('delay5' if delay else 'ordinary')
                    jobs.append((p,s,capital,item,folder))
        self._reserve(jobs)
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending={pool.submit(execute,item,s,folder,delay=delay):(p,s,c) for p,s,c,item,folder in jobs}
            for future in as_completed(pending):
                p,s,c=pending[future];receipt=future.result();key=f'{p}/{s}/{c}';result[key]=receipt
                if progress:progress(dict(event='native_account_complete',arm=name,cell=key,delay=delay,
                    valid=receipt['passed'],completed=len(result),planned=len(jobs)))
        return result

    def _reserve(self,jobs):
        """Reserve worst-case worker cost before launch; resumption never retries a failed trial."""
        path=self.out/'budget_claims.jsonl';claims=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        known={x['folder'] for x in claims};fresh=[j for j in jobs if str(j[-1]) not in known]
        envelope=self.registration_data['resource_envelope']
        total=len(claims)+len(fresh)
        if envelope['spent_account_equivalents']+total>512 or envelope['spent_native_seconds']+1200*total>200*3600:
            raise ValueError('Registered prospective resource ceiling would be exceeded; no automatic expansion')
        with path.open('a',encoding='utf-8') as stream:
            for p,s,c,item,folder in fresh:
                stream.write(json.dumps(dict(folder=str(folder),period=p,scenario=s,capital=c,arm=item['arm'],reserved_seconds=1200))+'\n')

    def baseline_delay(self,progress=None):
        target=self.out/'baseline_delay.json'
        if target.exists():return authenticate(target)
        receipts=self._batch('v6_h0',delay=5,progress=progress);cells={};bindings=[]
        for key,r in receipts.items():
            if not r.get('passed'):continue
            cell=account(r);mark=self.out/'qualification/baseline_delay'/(key.replace('/','_')+'.json')
            fm=self._minute(r,mark);bindings.append(bound(mark));cell['finer_dd_pp']=fm.get('retained_minute_dd_pct');cell['finer_valid']=fm['passed'];cells[key]=cell
        result=dict(native_valid=all(r.get('passed') for r in receipts.values()),cells=cells,
            qualification_pass=len(cells)==8 and all(c['finer_valid'] and max(c['mtm_dd_pp'],c['finer_dd_pp'])<=15 for c in cells.values()),
            scope='Mandatory baseline delay diagnostic, never a selectable treatment',receipts=receipts)
        seal(target,result,list(receipts.values()),[bound(self.registration),*bindings]);return result

    def _minute(self,receipt,path):
        if path.exists():return read(path)
        try:return finer_marks(receipt,path)
        except ValueError as exc:
            result=dict(passed=False,error=repr(exc),scope='Unavailable or inconsistent finer qualification; no substituted DD or promotion.')
            write(path,result);return result

    def _main(self,name,progress=None):
        if name in self._raw:return self._raw[name]
        actual=self._batch(name,progress=progress)
        receipts={k.rsplit('/',1)[0]:v for k,v in actual.items()}
        self._receipts[name]=receipts
        if any(r.get('passed') is not True for r in receipts.values()):
            self._raw[name]=None;return None
        self._raw[name]={k:account(r) for k,r in receipts.items()}
        return self._raw[name]

    def receipts(self,name):
        self._main(name);return self._receipts[name]

    def _qualification_profile(self,name,remove,period):
        """Registered component removal on an endogenous candidate account."""
        identifier=name+'__remove_'+remove
        if identifier+'/100000000' in self.profiles[period]:return identifier
        if remove not in ('veto','cap'):raise ValueError('Unregistered integrated removal')
        parent=self.profiles[period][name+'/100000000'];plan=deepcopy(read(parent['plan']['path']))
        config=deepcopy(self.arms[name]['config']);mode=self.arms[name]['mode']
        if remove=='cap':config['risk_max_gross_equity_pct']=0
        else:mode='disabled'
        out=self.out/'qualification_inputs'/identifier/period
        if (out/'profile.json').exists():item=read(out/'profile.json')
        else:
            out.mkdir(parents=True,exist_ok=False);write(out/'config.json',config)
            plan['input_sha256'].pop(plan['config']);plan['config']=str(out/'config.json');plan['input_sha256'][plan['config']]=bound(plan['config'])['sha256']
            plan['round3_execution']={**plan['round3_execution'],'mode':mode,'arm':identifier,'registration':bound(self.registration)}
            write(out/'plan.json',plan)
            mp=Path(parent['manifest']['path']);payload=read(mp);payload.pop('manifest_hash');code=mp.parent/'files'
            for n in ('config.json','plan.json'):
                rel=f'research/c17/integrated/{identifier}/{n}';dest=code/rel
                if dest.exists():raise ValueError('Preserve qualified config generation')
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(out/n,dest);payload['files'][rel]=bound(dest)['sha256']
            payload['evidence']['C17_integrated_removal']=dict(registration=bound(self.registration),parent=name,remove=remove,
                source_unchanged=True,config=bound(out/'config.json'))
            newmanifest=mp.parent/(identifier+'.manifest.json')
            write(newmanifest,{**payload,'manifest_hash':stable_signature(payload)})
            item={**parent,'arm':identifier,'plan':bound(out/'plan.json'),'manifest':bound(newmanifest)}
            write(out/'profile.json',item)
        self.profiles[period][identifier+'/100000000']=item
        return identifier

    def evaluate(self,name,progress=None):
        self.check_frozen()
        if name in self._evaluations:
            authenticate(self._evaluations[name]['receipt']);return self._evaluations[name]
        path=self.out/'evaluations'/(name+'.json')
        if path.exists():
            value=authenticate(path)
            # Rebuild and authenticate cached native receipts before use.
            self._main(name,progress);self._evaluations[name]=value;return value
        if not self.reached(name):raise ValueError('Inactive/unsupported arm cannot enter full evaluation')
        cells=self._main(name,progress)
        if cells is None:
            result=dict(native_valid=False,accepted_control=False,qualification=dict(promote=False,failures=['native_execution_invalid']),receipt=str(path))
            seal(path,result,list(self._receipts[name].values()),[bound(self.registration)])
            self._evaluations[name]=result;return result
        stats=compare(self.control,cells);score=scalar(cells)
        reach={k:native_reach(r['folder'],self.control_receipts[k]['folder']) for k,r in self.receipts(name).items()}
        components={k:funded_decomposition(self.control_receipts[k]['folder'],r['folder']) for k,r in self.receipts(name).items()}
        signature=any(r['changed_physical_fills'] and r['changed_action_records'] for r in reach.values())
        if self.arms[name]['config'].get('research_opening_failure_min_hold_bars'):
            signature=signature and any(r['opening_failure_exit_decisions'] for r in reach.values())
        checks=dict(source=True,flat=True,independent_accounting=True,mechanism=signature,
            ablation=False,neighbourhood=False,delay=False,cost=True,retained_components=False)
        capcells={k+'/100000000':v for k,v in cells.items()}
        preliminary=qualify(self.control,cells,capcells,stats,checks,diagnostic=not self.arms[name]['selectable'])
        definitive=[f for f in preliminary['failures'] if f not in ('missing_capital_qualification',
            'qualification:ablation','qualification:neighbourhood','qualification:delay','qualification:retained_components')]
        detail={};evidence_receipts=list(self.receipts(name).values());evidence_bindings=[bound(self.registration)]
        if not definitive:
            extra=self._batch(name,(95000000,105000000),progress=progress)
            for key,r in extra.items():
                if not r.get('passed'):capcells[key]=dict(native_valid=False,mtm_dd_pp=None);continue
                capcells[key]=account(r)
            allreceipts={**{k+'/100000000':v for k,v in self.receipts(name).items()},**extra}
            for key,r in allreceipts.items():
                if not r.get('passed'):continue
                markpath=self.out/'qualification'/name/(key.replace('/','_')+'_minute.json')
                fm=self._minute(r,markpath)
                evidence_bindings.append(bound(markpath))
                capcells[key]['finer_dd_pp']=fm.get('retained_minute_dd_pct')
                checks['source']=checks['source'] and fm['passed']
                detail[key]=dict(funding=funding_chronology(r),minute=fm)
            evidence_receipts.extend(extra.values())
            controls=read(self.registration_data['capital_controls']['path'])
            detail['capital_sensitivity']={}
            for capital in (95000000,105000000):
                capreceipts={k:read(v['path']) for k,v in controls['capitals'][str(capital)].items()}
                for v in controls['capitals'][str(capital)].values():
                    if bound(v['path'])['sha256']!=v['sha256']:raise ValueError('Changed capital control receipt')
                    evidence_bindings.append(v)
                evidence_receipts.extend(capreceipts.values());baseline={k:account(v) for k,v in capreceipts.items()}
                candidate={k:capcells[k+'/'+str(capital)] for k in self.control}
                if all(c['native_valid'] for c in candidate.values()):
                    detail['capital_sensitivity'][str(capital)]=dict(statistics=compare(baseline,candidate),control=baseline,
                        delta_return_pp={k:candidate[k]['return_pp']-baseline[k]['return_pp'] for k in baseline},
                        scope='Paired within capital; not independent market samples')
            # Perform every registered removal on the actual changed population.
            removals={}
            for component in ('veto','cap'):
                for p in ('is','oos'):identifier=self._qualification_profile(name,component,p)
                removal=self._main(identifier,progress)
                removals[component]=dict(native_valid=removal is not None,
                    delta_return_pp={k:cells[k]['return_pp']-removal[k]['return_pp'] for k in cells} if removal else None,
                    statistics=compare(removal,cells) if removal else None)
                evidence_receipts.extend(self.receipts(identifier).values())
            cfg=self.arms[name]['config'];votes=cfg['entry_plan_routes'][0]['min_quality_votes'];hold=cfg['research_opening_failure_min_hold_bars']
            neighbours=set()
            if votes!=6:neighbours.add(f'v6_h{hold}')
            if hold:neighbours.update((f'v{votes}_h0',f'v{votes}_h'+str(8 if hold==6 else 6)))
            neighbour_data={n:self._main(n,progress) for n in sorted(neighbours)}
            checks['ablation']=all(v is not None for v in neighbour_data.values())
            checks['neighbourhood']=checks['ablation']
            component_signatures={}
            for component,n in ([('votes',f'v6_h{hold}')] if votes!=6 else [])+([('hold',f'v{votes}_h0')] if hold else []):
                parent=neighbour_data[n]
                reached={k:native_reach(self.receipts(name)[k]['folder'],self.receipts(n)[k]['folder']) for k in cells} if parent else {}
                changed=any(r['changed_action_records'] and r['changed_physical_fills'] for r in reached.values())
                if component=='hold':changed=changed and any(r['opening_failure_exit_decisions'] for r in reached.values())
                contribution=scalar(cells)['R']-scalar(parent)['R'] if parent else None
                component_signatures[component]=dict(native_reach=reached,positive_funded_contribution=contribution is not None and contribution>0,contribution_pp=contribution)
                checks['mechanism']=checks['mechanism'] and changed and contribution is not None and contribution>0
            for n in neighbours:evidence_receipts.extend(self.receipts(n).values())
            checks['retained_components']=all(v['native_valid'] and v['statistics']['5']['unadjusted_CI95_pp'][1]>=0 for v in removals.values()) and checks['ablation']
            base_delay=self.baseline_delay(progress)['receipts'];test_delay=self._batch(name,delay=5,progress=progress)
            evidence_receipts.extend([*base_delay.values(),*test_delay.values()]);evidence_bindings.append(bound(self.out/'baseline_delay.seal.json'))
            delay_valid=all(r.get('passed') for r in [*base_delay.values(),*test_delay.values()])
            if delay_valid:
                bd={k.rsplit('/',1)[0]:account(v) for k,v in base_delay.items()};td={k.rsplit('/',1)[0]:account(v) for k,v in test_delay.items()}
                detail['delay']=dict(statistics=compare(bd,td),cells=td,control=bd)
                delay_risk=True
                for key,r in test_delay.items():
                    markpath=self.out/'qualification'/name/('delay_'+key.replace('/','_')+'_minute.json')
                    fm=self._minute(r,markpath);evidence_bindings.append(bound(markpath))
                    td[key.rsplit('/',1)[0]]['finer_dd_pp']=fm.get('retained_minute_dd_pct')
                    delay_risk=delay_risk and fm['passed'] and max(account(r)['mtm_dd_pp'],fm['retained_minute_dd_pct'])<=15
                delay_gain=scalar(td)['R']-scalar(bd)['R']
                checks['delay']=self.baseline_delay(progress)['qualification_pass'] and delay_risk and delay_gain>=0 and min(td[k]['return_pp']-bd[k]['return_pp'] for k in bd)>=-1
                detail['delay']['aggregate_gain_pp']=delay_gain
            detail['removals']=removals;detail['neighbours']=neighbour_data
            detail['component_signatures']=component_signatures
        qualification=qualify(self.control,cells,capcells,stats,checks,diagnostic=not self.arms[name]['selectable'])
        all_native_valid=all(r.get('passed') for r in evidence_receipts)
        result=dict(native_valid=all_native_valid,accepted_control=False,qualification=qualification,
            statistics=stats,cells=cells,capital_cells=capcells,mechanism=reach,funded_decomposition=components,
            qualification_detail=detail,qualification_checks=checks,receipt=str(path),
            coverage_gaps=['Global funded oracles','All-rejected standalone labels','Matched-entry placebo','Untouched holdout'],
            uncertainty_scope='Dependent scenarios/capitals are not independent market samples; earlier adaptive search remains.')
        if all_native_valid:result['score']=score
        else:result['primary_valid_ensemble_metrics']={k:v for k,v in score.items() if k!='S'}
        seal(path,result,evidence_receipts,evidence_bindings);self._evaluations[name]=result;return result

    def validate_selected(self,name):
        self.check_frozen();value=self.evaluate(name);authenticate(value['receipt']);selectable_score(value)
        for receipt in self.receipts(name).values():
            verify_acceptance(receipt);audit=read(Path(receipt['folder'])/'execution_audit.json')['audit_receipt']
            if (audit['audit_version'],audit['metric_version'],audit['execution_version'])!=('execution-ledger-v16','session-equity-trade-groups-v5','timestamp-phases-v6'):
                raise ValueError('Unexpected native execution contract')
            funding_chronology(receipt)
        return value

    def phase_evidence(self,phase,name):
        return dict(phase=phase,current_policy=name,registration=str(self.registration),
            trials={k:dict(receipt=v.get('receipt'),native_valid=v['native_valid'],qualification=v.get('qualification')) for k,v in self._evaluations.items()},
            reach=self.reach,coverage='Native funded gate/management contrasts; global signal-value/capture measurements remain incomplete.',
            score_contract='C16 unchanged',max_workers=2,holdout='Unavailable: all retained history already consumed',
            production_enabled=False,stopped_studies=['exp0031','exp0039/C15'])

    def final_freeze(self,name):
        target=self.out/'final_freeze.json'
        if target.exists():
            if read(target)['selected']!=name:raise ValueError('No post-freeze replacement')
            return
        value=self.validate_selected(name)
        write(target,dict(selected=name,configuration=self.arms[name]['config'],evaluation=bound(value['receipt']),
            contract=bound(ROOT/'freeze/manifest.json'),holdout_accesses=0,untouched_holdout_available=False,
            no_replacement_after_final_freeze=True))

    def close(self):pass

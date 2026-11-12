"""Apply predeclared preservation gates, then stress the small native frontier."""
from scripts import olr_targeted_risk_study_v1 as t
from scripts import olr_targeted_risk_results_v1 as r
n,s,b=t.n,t.s,t.b


def rank(row):
    a,z=row['windows']['train'],row['windows']['oos']
    return (z['return_pct']/z['drawdown_pct'],a['return_pct'],z['trades'])


def qualify():
    data=r.collect()
    assert all(len(row['windows'])==2 for row in data['candidates'])
    candidates={c['id']:c for c in n.read(t.ROOT/'candidate_plan.json')['candidates']}
    eligible=[row for row in data['candidates'] if row.get('nominal_qualified')]
    # Three objective leaders may coincide. No new parameter or signal is searched.
    finalists=[]
    if eligible:
        leaders=(max(eligible,key=rank),max(eligible,key=lambda row:(row['windows']['train']['return_pct'],rank(row))),
            max(eligible,key=lambda row:(min(row['windows']['train']['trades']/177,row['windows']['oos']['trades']/47),
                row['windows']['train']['trades']/177+row['windows']['oos']['trades']/47,rank(row))))
        finalists=list({row['id']:row for row in leaders}.values())
    else:
        finalists=[row for row in data['candidates'] if row['label']=='auction_priority']
    selected_ids=[row['id'] for row in finalists]
    proof=t.ROOT/'stress_selection.json'
    registration={'utc':s.utc(),'consumed_results_sha256':n.sha(t.OUT/'results.json'),
        'finalists':selected_ids,'reason':'Return/risk, IS preservation and balanced frequency leaders; mechanical-priority fallback if no risk profile meets nominal gates.',
        'adaptive_development':True,'no_new_parameter_values':True}
    if proof.exists():assert n.read(proof)['finalists']==selected_ids
    else:n.write(proof,registration)
    control=next(c for c in candidates.values() if c['label']=='guard_control')
    for c in [control,*[candidates[cid] for cid in selected_ids]]:
        for w in ('train','oos'):b.evaluate(c,w,'combined')
    data=r.collect()
    qualified=[row for row in data['candidates'] if row.get('nominal_qualified') and row.get('stress_qualified')]
    if qualified:
        selected=max(qualified,key=rank);status='qualified_research_successor'
    else:
        priority=next(row for row in data['candidates'] if row['label']=='auction_priority')
        if not priority.get('stress'):
            for w in ('train','oos'):b.evaluate(candidates[priority['id']],w,'combined')
            data=r.collect();priority=next(row for row in data['candidates'] if row['label']=='auction_priority')
        mechanical=all(v for k,v in priority['gates'].items() if k!='oos_drawdown_target') and priority.get('stress_qualified')
        selected=priority if mechanical else next(row for row in data['candidates'] if row['label']=='guard_control')
        status='mechanical_improvement_only' if mechanical else 'retain_guard_no_qualified_upgrade'
    for c in [control,candidates[selected['id']]]:
        for w in ('train','oos'):b.evaluate(c,w,'baseline',True)
    data=r.collect()
    alternatives=[]
    for row in data['candidates']:
        if row.get('nominal_qualified') and row.get('stress_qualified'):
            profile={'candidate_id':row['id'],'label':row['label'],'mutations':candidates[row['id']]['mutations'],
                'history_policy':'literal_carry','promotion':False,
                'nominal':{w:{k:row['windows'][w][k] for k in ('return_pct','drawdown_pct','trades')} for w in ('train','oos')},
                'combined_stress':{w:{k:row['stress'][w][k] for k in ('return_pct','drawdown_pct','trades')} for w in ('train','oos')}}
            n.write(t.OUT/('policy_'+row['label']+'.json'),profile)
            alternatives.append(profile)
    n.write(t.OUT/'qualified_alternatives.json',{'policies':alternatives,'scope':'Only the stress-checked leaders that pass every nominal and stress gate; not a global optimum.'})
    selection={'created_utc':s.utc(),'status':status,'selected_id':selected['id'],'selected_label':selected['label'],
        'mutations':candidates[selected['id']]['mutations'],'history_policy':'literal_carry',
        'promotion':False,'independent_validation':False,
        'source_generation_hashes':n.read(t.ROOT/'preparation_complete.json')['sources'],
        'results_sha256':n.sha(t.OUT/'results.json'),'registration_sha256':n.sha(t.ROOT/'registration.json'),
        'stress_selection_sha256':n.sha(proof),
        'scope':'Research successor using exact original scoring, completed five-minute execution and candidate/scenario-specific actual IS terminal market history. Account starts flat at 100m KRW in each scoring window.'}
    n.write(t.OUT/'selected_research_policy.json',selection)
    print({'selection':selected['label'],'status':status},flush=True)


if __name__=='__main__':qualify()

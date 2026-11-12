"""Research-only rescue on otherwise empty afternoons; all downstream gates run."""
from dataclasses import replace
from math import isfinite

def eligible(ctx,reasons):
    values=(ctx.lagged_flow_5d,ctx.intraday_sector_ret,ctx.intraday_sector_breadth,ctx.intraday_sector_effective_count)
    return (reasons==['lagged_sector_flow_below_floor'] and all(isfinite(x) for x in values)
        and values[0]>0 and values[1]>0 and values[2]>=.5 and values[3]>=3)

def select(snapshot,contexts,config):
    from .research import afternoon_selection_from_contexts, _afternoon_reject_reasons, final_candidate_config_fingerprint
    base_cfg=replace(config,afternoon_empty_sector_flow_rescue=False)
    base=afternoon_selection_from_contexts(snapshot,contexts,base_cfg)
    candidates=[]
    if not base.candidates:
        candidates=[c for c in snapshot.candidates if c.symbol in contexts and eligible(contexts[c.symbol],_afternoon_reject_reasons(contexts[c.symbol],base_cfg))]
    if candidates:
        # Only these prequalified names receive the single gate exception.
        # All score bands and later native entry conditions remain mandatory.
        rescued=afternoon_selection_from_contexts(replace(snapshot,candidates=tuple(candidates)),contexts,
            replace(base_cfg,afternoon_min_sector_flow=-1e100))
        selected=tuple(replace(c,metadata={**c.metadata,'empty_sector_flow_rescue_v4':True,
            'rescue_original_reasons':base.metadata['afternoon_rejected_symbols'][c.symbol]}) for c in rescued.candidates)
        result=replace(rescued,candidates=selected)
    else:result=base
    fingerprint=final_candidate_config_fingerprint(config)
    meta={**result.metadata,'candidate_config_hash':fingerprint,'final_candidate_config_hash':fingerprint,
        'empty_sector_flow_rescue_v4':{'baseline_empty':not bool(base.candidates),
            'eligible_symbols':[c.symbol for c in candidates],
            'rescued_symbols':[c.symbol for c in result.candidates] if not base.candidates else [],
            'baseline_rejected':base.metadata.get('afternoon_rejected_symbols',{}),
            'rule':'sole lagged-sector-flow rejection; positive stock flow/current sector return; breadth>=.5; effective count>=3; downstream score and six-bar signal gates unchanged'}}
    return replace(result,metadata=meta)

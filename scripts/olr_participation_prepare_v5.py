"""Clone immutable v4 generation and add default-off bounded entry pilot."""
from scripts import olr_participation_study_v5 as t
from scripts import olr_repair_prepare_v4 as parent
n,s,b=t.n,t.s,t.b
once=parent.once


def patch(name,v):
    if name=='strategy_olr/config.py':
        v=once(v,'    afternoon_empty_sector_flow_rescue: bool = False\n',
            '    afternoon_empty_sector_flow_rescue: bool = False\n    research_empty_participation_rescue: bool = False\n')
        v=once(v,'        import math\n','        import math\n        if type(self.research_empty_participation_rescue) is not bool:\n            raise ValueError("Participation rescue must be boolean")\n')
        return once(v,'    "olr.afternoon.empty_sector_flow_rescue":',
            '    "olr.research.empty_participation_rescue": "research_empty_participation_rescue",\n    "olr.afternoon.empty_sector_flow_rescue":')
    if name=='strategy_olr/research.py':
        v=once(v,'    "afternoon_empty_sector_flow_rescue",\n','    "afternoon_empty_sector_flow_rescue",\n    "research_empty_participation_rescue",\n')
        return once(v,"    if config.afternoon_empty_sector_flow_rescue:\n", "    if config.research_empty_participation_rescue:\n        payload['research_empty_participation_rescue']=True\n    if config.afternoon_empty_sector_flow_rescue:\n")
    if name=='strategy_olr/core/logic.py':
        marker='    metadata["reserved_notional"] = int(qty) * estimated_price * (1.0 + entry_cost_buffer)\n'
        return once(v,marker,"    if candidate.metadata.get('empty_participation_rescue_v5'):\n        metadata['empty_participation_rescue_v5']=True\n"+marker)
    if name=='strategy_olr/core/allocation.py':
        v=once(v,'    unrestricted_account = account\n',"    from strategy_olr.participation_rescue_v5 import constrain\n    proposals, account = constrain(proposals, account, state, bars, portfolio)\n    unrestricted_account = account\n")
        marker='            action = by_id[decision.intent_id]\n            cap, exit_cap, evidence = forecast(state, by_symbol[action.symbol], config)\n'
        return once(v,marker,"            action = by_id[decision.intent_id]\n            if action.metadata.get('empty_participation_rescue_v5'):\n                continue  # The small pilot cannot acquire recycled top-ups.\n            cap, exit_cap, evidence = forecast(state, by_symbol[action.symbol], config)\n")
    raise ValueError(name)


def configure():
    parent.t=t; parent.patch=patch


def prepare():
    configure(); parent.prepare()


def parity(*args,**kwargs):
    configure(); return parent.parity(*args,**kwargs)


if __name__=='__main__': prepare()

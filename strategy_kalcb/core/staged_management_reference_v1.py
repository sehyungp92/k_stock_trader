"""Explicit research R reference, separate from actual weighted purchase cost.

This is a management hypothesis, not an accounting correction or live default.
Cash PnL, funding, entry_price and monetary cost guards remain actual-cost based.
Only R-valued management/observed excursions use the declared initial reference.
"""
from dataclasses import replace
from math import isfinite

META='staged_management_entry_reference'


def reference_r(position, price, original):
    anchor=position.metadata.get(META)
    if anchor is None:return original(position,price)
    if not isfinite(float(anchor)) or float(anchor)<=0:raise ValueError('Invalid declared management reference')
    if position.risk_per_share<=0:return 0.
    return (float(price)-float(anchor))/position.risk_per_share


def install_reference():
    from . import staged_commitment_v1 as staged
    from .state import KALCBPositionState
    if getattr(staged,'_reference_installed',False):raise ValueError('Duplicate reference installation')
    staged._reference_installed=True
    old_r=KALCBPositionState.unrealized_r
    old_completion=staged.completion_action
    def unrealized_r(position,price):return reference_r(position,price,old_r)
    def completion(state,symbol,clock,config,portfolio):
        action=old_completion(state,symbol,clock,config,portfolio)
        if action is None or not state.meta.get('staged_management_reference_enabled',False):return action
        position=state.symbol_state(symbol).position
        if not position or position.qty_open!=position.qty_entry:raise ValueError('Reference requires intact initial commitment')
        if META in position.metadata:raise ValueError('Reference already assigned')
        # Freeze only when a completion is actually intended; no future fill or
        # price is observed and an uncompleted initial position stays original.
        anchor=float(position.entry_price)
        if not isfinite(anchor) or anchor<=0:raise ValueError('Invalid initial native fill basis')
        position.metadata[META]=anchor
        position.metadata['staged_management_reference_at']=clock.isoformat()
        position.metadata['staged_management_reference_scope']='R_management_only_actual_VWAP_cost_unchanged'
        return replace(action,metadata={**action.metadata,META:anchor,
            'staged_management_reference_at':clock.isoformat(),
            'staged_management_reference_scope':'R_management_only_actual_VWAP_cost_unchanged'})
    KALCBPositionState.unrealized_r=unrealized_r
    staged.completion_action=completion

"""Native checkpoint evidence must not depend on unordered-container layout."""
import json
import os
import subprocess
import sys

from strategy_olr.core.state import OLRState
from strategy_olr.core.serializers import snapshot_state
from backtests.auto.shared.cache_keys import stable_signature


def state(items):
    return OLRState(meta={'funded_entry_contexts_v1':[{'account':{'occupied':frozenset(items)}}]})


def test_snapshot_nested_immutable_membership_is_strict_json():
    original=state([('OLR','035900'),('OLR','009150')])
    value=json.loads(json.dumps(snapshot_state(original)))
    assert value['meta']['funded_entry_contexts_v1'][0]['account']['occupied']==[['OLR','009150'],['OLR','035900']]
    assert isinstance(original.meta['funded_entry_contexts_v1'][0]['account']['occupied'],frozenset)
    assert json.loads(json.dumps(snapshot_state(state([]))))['meta']['funded_entry_contexts_v1'][0]['account']['occupied']==[]


def test_snapshot_hash_survives_pickle_and_process_hash_seed():
    code='''import json,pickle
from strategy_olr.core.state import OLRState
from strategy_olr.core.serializers import snapshot_state
from backtests.auto.shared.cache_keys import stable_signature
x=OLRState(meta={'funded_entry_contexts_v1':[{'account':{'occupied':frozenset([('OLR','035900'),('OLR','009150')])}}]})
y=pickle.loads(pickle.dumps(x))
print(json.dumps([stable_signature(snapshot_state(x)),stable_signature(snapshot_state(y))]))
'''
    hashes=[]
    for seed in (0,26,90210):
        env=os.environ.copy();env.update(PYTHONHASHSEED=str(seed),PYTHONDONTWRITEBYTECODE='1')
        result=subprocess.run([sys.executable,'-B','-c',code],env=env,capture_output=True,text=True,check=True)
        before,after=json.loads(result.stdout.strip().splitlines()[-1])
        assert before==after, 'Equivalent state changed across pickle at hash seed '+str(seed)
        hashes.append(before)
    assert len(set(hashes))==1


def test_membership_and_quantity_changes_remain_in_checkpoint_signature():
    a=state([('OLR','035900'),('OLR','009150')]);b=state([('OLR','035900')])
    assert stable_signature(snapshot_state(a))!=stable_signature(snapshot_state(b))
    a.meta['current_quantity']=19;b=state([('OLR','035900'),('OLR','009150')]);b.meta['current_quantity']=18
    assert stable_signature(snapshot_state(a))!=stable_signature(snapshot_state(b))

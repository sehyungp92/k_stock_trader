"""Finalize draft isolated engine bytes before any validation or release."""
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import olr_native_acceleration as a

window=sys.argv[1]
path=Path('C:/Users/sehyu/.codex/research/olr_native_r19')/window/'inputs/generation.json'
info=a.read(path)
v=Path('E:/CodexResearch/k_stock_trader/olr_round1_comprehensive_20261009/full_oos_20260401_20260710/continuous_history_v10/revisions/revision_19_native_overheads')
archive=v/(window+'_draft_generation.json')
assert not archive.exists()
a.write(archive,info)
engine=Path(info['engine']); original=Path(info['original_engine'])
for name in a.PATCHED:
    assert a.sha(engine/name)==info['changed_files'][name]['after']
    assert a.sha(original/name)==info['changed_files'][name]['before']
    text=a.optimized_source(name,(original/name).read_text(encoding='utf-8'))
    compile(text,name,'exec')
    (engine/name).write_text(text,encoding='utf-8',newline='\n')
    info['changed_files'][name]['after']=a.sha(engine/name)
    info['files'][name]=a.sha(engine/name)
info['source_generation_hash']=subprocess.check_output([sys.executable,'-B','-c',
    'from backtests.auto.shared.cache_keys import source_generation_hash;print(source_generation_hash())'],cwd=engine,text=True).strip()
info['builder_sha256']=a.sha(a.__file__)
a.write(path,info)
print(window,info['source_generation_hash'],flush=True)

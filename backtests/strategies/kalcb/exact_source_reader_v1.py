"""Fresh all-byte source digests through read-only mappings; NEVER a hash cache.

Every invocation re-enumerates source membership, verifies file identity/stat,
hashes every mapped byte afresh and checks the expected complete source hash.
No manifest substitutes for executing bytes. A mutation fails closed.
"""
import hashlib
import json
import mmap
import os
from pathlib import Path

DIRECTORIES=('backtests','strategy_common','strategy_kalcb','strategy_olr','strategy_pcim',
             'deployment','instrumentation','oms','oms_client','kis_core','scripts')


def inventory(root):
    paths=[]
    for directory in DIRECTORIES:
        base=os.path.join(root,directory)
        for folder,dirs,files in os.walk(base):
            for name in files:
                if name.lower().endswith('.py'):paths.append(os.path.join(folder,name))
    data=os.path.join(root,'kis_core','data')
    if os.path.isdir(data):
        paths.extend(e.path for e in os.scandir(data) if e.is_file() and e.name.lower().endswith('.yaml'))
    return sorted(paths,key=lambda p:tuple(os.path.normcase(p).split(os.sep)))


def fingerprint(stat):
    # Python3.12 Windows path-stat ctime is birthtime while handle-fstat ctime
    # can be metadata-change time. Compare actual identity, extent and mtime;
    # freshly hashing every byte also detects restored-mtime same-size edits.
    return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)


def locked_read(path):
    if os.name!='nt':return open(path,'rb')
    import ctypes,msvcrt
    from ctypes import wintypes
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    create=api.CreateFileW
    create.argtypes=(wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE)
    create.restype=wintypes.HANDLE
    # GENERIC_READ, FILE_SHARE_READ only, OPEN_EXISTING. Existing readers may
    # coexist, but the kernel denies writes, truncation, deletion/replacement
    # for the lifetime of this handle. No attributes or file bytes are changed.
    handle=create(path,0x80000000,1,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    try:return os.fdopen(msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY),'rb')
    except BaseException:
        api.CloseHandle.argtypes=(wintypes.HANDLE,);api.CloseHandle(handle);raise


class ExactSourceReader:
    def __init__(self,root,expected):
        self.root=str(Path(root).resolve());self.expected=expected;self.paths=inventory(self.root);self.rows=[];self.calls=0
        try:
            for p in self.paths:
                f=locked_read(p);stat=os.fstat(f.fileno())
                mapped=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) if stat.st_size else None
                self.rows.append((p,os.path.relpath(p,self.root).replace('\\','/'),f,mapped,fingerprint(stat)))
            self()
        except BaseException:self.close();raise

    def __call__(self):
        if inventory(self.root)!=self.paths:raise ValueError('Executing source membership changed')
        rows=[]
        for path,relative,file,mapped,original in self.rows:
            if os.name!='nt' and (fingerprint(os.stat(path))!=original or fingerprint(os.fstat(file.fileno()))!=original):
                raise ValueError('Executing source identity/stat changed')
            sha=hashlib.sha256(mapped if mapped is not None else b'').hexdigest()
            if os.name!='nt' and fingerprint(os.stat(path))!=original:raise ValueError('Executing source changed while hashing')
            rows.append({'path':relative,'sha256':sha})
        actual=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if actual!=self.expected:raise ValueError('Fresh executing source bytes changed')
        self.calls+=1;return actual

    def close(self):
        for path,relative,file,mapped,original in self.rows:
            if mapped is not None:mapped.close()
            file.close()
        self.rows=[]

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

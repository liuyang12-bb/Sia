"""Read-only CPython3.10/Win64 stack sampler for THIS frozen worker only.

Uses QUERY_INFORMATION|VM_READ. No memory writes, injection, debugger attach,
thread suspension, signals, source changes or additional dependencies. ABI
offsets match the installed Python310 include headers and are self-tested.
Only project frames and whitelisted scalar proof locals are exported.
"""
import ctypes as C,struct,json,sys,time,hashlib
from ctypes import wintypes as W
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_core80_full_lex_repair_20261031_run5';TREE=ROOT/'cc-joint-core80-full-lex-repair-v8'
K=C.WinDLL('kernel32',use_last_error=True);P=C.WinDLL('psapi',use_last_error=True)
K.OpenProcess.argtypes=[W.DWORD,W.BOOL,W.DWORD];K.OpenProcess.restype=W.HANDLE
K.ReadProcessMemory.argtypes=[W.HANDLE,C.c_void_p,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t)];K.ReadProcessMemory.restype=W.BOOL
K.CloseHandle.argtypes=[W.HANDLE]
P.EnumProcessModulesEx.argtypes=[W.HANDLE,C.c_void_p,W.DWORD,C.POINTER(W.DWORD),W.DWORD];P.EnumProcessModulesEx.restype=W.BOOL
P.GetModuleFileNameExW.argtypes=[W.HANDLE,C.c_void_p,W.LPWSTR,W.DWORD];P.GetModuleFileNameExW.restype=W.DWORD

class Reader:
 def __init__(self,pid):
  self.handle=K.OpenProcess(0x0400|0x0010,False,pid)
  if not self.handle:raise C.WinError(C.get_last_error())
 def close(self):K.CloseHandle(self.handle)
 def read(self,address,size):
  if not address or size<0 or size>2**20:raise ValueError('Bounded invalid memory read')
  b=C.create_string_buffer(size);n=C.c_size_t()
  if not K.ReadProcessMemory(self.handle,C.c_void_p(address),b,size,C.byref(n)) or n.value!=size:raise OSError('Racy or unreadable address')
  return b.raw
 def u64(self,p):return struct.unpack('<Q',self.read(p,8))[0]
 def i64(self,p):return struct.unpack('<q',self.read(p,8))[0]
 def i32(self,p):return struct.unpack('<i',self.read(p,4))[0]
 def cstr(self,p,maximum=256):
  s=bytearray()
  for i in range(maximum):
   v=self.read(p+i,1)
   if v==b'\0':break
   s+=v
  return s.decode('utf8',errors='replace')
 def kind(self,p):return self.cstr(self.u64(self.u64(p+8)+24))
 def string(self,p):
  if self.kind(p)!='str':raise ValueError('Expected unicode')
  n=self.i64(p+16);state=struct.unpack('<I',self.read(p+32,4))[0];kind=(state>>2)&7;compact=(state>>5)&1;ascii_=(state>>6)&1
  if not 0<=n<=8192 or kind not in (1,2,4):raise ValueError('Unicode shape')
  data=p+(48 if ascii_ else 72) if compact else self.u64(p+72)
  return self.read(data,n*kind).decode({1:'latin1',2:'utf-16-le',4:'utf-32-le'}[kind]) if n else ''
 def scalar(self,p):
  if not p:return None
  k=self.kind(p)
  if k in ('int','bool'):
   n=self.i64(p+16)
   if abs(n)>4:return '<large int>'
   v=sum(struct.unpack('<I',self.read(p+24+4*i,4))[0]<<(30*i) for i in range(abs(n)))
   return (-v if n<0 else v) if k=='int' else bool(v)
  if k=='str':return self.string(p)
  if k=='float':return struct.unpack('<d',self.read(p+16,8))[0]
  if k=='NoneType':return None
  return '<'+k+'>'
 def dictionary(self,p):
  if self.kind(p)!='dict':return {}
  keys=self.u64(p+32);values=self.u64(p+40);size=self.i64(keys+8);entries=self.i64(keys+32)
  if size<1 or size>2**16 or size&(size-1) or not 0<=entries<=4096:raise ValueError('Dictionary layout')
  width=1 if size<=0xff else 2 if size<=0xffff else 4
  table=keys+40+size*width;out={}
  for i in range(entries):
   key=self.u64(table+24*i+8);value=self.u64(values+8*i) if values else self.u64(table+24*i+16)
   if key and value:
    label=self.string(key)
    if label in ('budget','incumbent_integer_cost','valid_improved_integer_cost','gcd','lex_stage','retained_prior_constraint_groups','status','message','seconds','variables','original_variables','proof','primary_global_upper','original_primary_lower_guard','proved','objective'):out[label]=self.scalar(value)
  return out
 def module(self):
  handles=(C.c_void_p*4096)();needed=W.DWORD()
  if not P.EnumProcessModulesEx(self.handle,handles,C.sizeof(handles),C.byref(needed),3):raise C.WinError(C.get_last_error())
  for base in list(handles)[:needed.value//C.sizeof(C.c_void_p)]:
   buf=C.create_unicode_buffer(32768);P.GetModuleFileNameExW(self.handle,base,buf,len(buf))
   if Path(buf.value).name.lower()=='python310.dll':return base,Path(buf.value)
  raise ValueError('Requires matching Python310.dll')

def exported_offset(file,name):
 b=file.read_bytes();pe=struct.unpack_from('<I',b,0x3c)[0];assert b[pe:pe+4]==b'PE\0\0';coff=pe+4;ns=struct.unpack_from('<H',b,coff+2)[0];osize=struct.unpack_from('<H',b,coff+16)[0];opt=coff+20;assert struct.unpack_from('<H',b,opt)[0]==0x20b
 sections=opt+osize
 def offset(rva):
  for i in range(ns):
   a=sections+40*i;vs,va,rs,rp=struct.unpack_from('<IIII',b,a+8)
   if va<=rva<va+max(vs,rs):return rp+rva-va
  raise ValueError('RVA outside sections')
 ex=offset(struct.unpack_from('<I',b,opt+112)[0]);count=struct.unpack_from('<I',b,ex+24)[0];functions,names,ordinals=[offset(v) for v in struct.unpack_from('<III',b,ex+28)]
 for i in range(count):
  p=offset(struct.unpack_from('<I',b,names+4*i)[0]);s=b[p:b.index(b'\0',p)].decode('ascii')
  if s==name:
   ordinal=struct.unpack_from('<H',b,ordinals+2*i)[0];return struct.unpack_from('<I',b,functions+4*ordinal)[0]
 raise ValueError('Export absent: '+name)

def frame_info(r,p):
 code=r.u64(p+32);filename=r.string(r.u64(code+104));name=r.string(r.u64(code+112));out=dict(function=name,file=filename,first_line=r.i32(code+40),instruction_offset=r.i32(p+96))
 if 'epsilon_lex_repair.py' in filename or 'numeric_repair.py' in filename or 'primary_band_presolve.py' in filename:
  n=r.i32(code+28);names=r.u64(code+72);assert 0<=n<=1024 and n<=r.i64(names+16)
  values={}
  for i in range(n):
   label=r.string(r.u64(names+24+8*i));value=r.u64(p+352+8*i)
   if label in ('value','budget','g','n','L','U','cutoff','bound','upper','record') and value:
    values[label]=r.dictionary(value) if label=='record' else r.scalar(value)
  out['proof_locals']=values
 return out

def selftest():
 r=Reader(psutil.Process().pid)
 try:
  sample={'budget':8185,'lex_stage':'诊断','proved':True};assert r.dictionary(id(sample))==sample
  f=sys._getframe();v=frame_info(r,id(f));assert v['function']=='selftest' and v['file']==__file__
  base,dll=r.module();runtime=base+exported_offset(dll,'_PyRuntime');api=C.pythonapi.PyInterpreterState_Head;api.restype=C.c_void_p;assert r.u64(runtime+32)==api()
  return dict(status='SELF_ABI_PASS',read_only_access_hex='0x410',python_dll=str(dll))
 finally:r.close()

def main():
 test=selftest()
 if '--self-test' in sys.argv:print(json.dumps(test,ensure_ascii=False));return
 ledger=[json.loads(x) for x in (OUT/'execution_manifest.jsonl').read_text(encoding='utf8').splitlines() if x];pid=[x['pid'] for x in ledger if x['event']=='CELL_PROCESS'][-1];wrapper=psutil.Process(pid);children=wrapper.children(recursive=True);workers=[p for p in [wrapper,*children] if any(Path(a).resolve()==(TREE/'run_core80.py').resolve() for a in p.cmdline() if a.endswith('.py'))];assert workers
 process=max(workers,key=lambda p:p.memory_info().rss);assert process.memory_info().rss>32*2**20
 manifest=json.loads((OUT/'preregistration.json').read_text(encoding='utf8'));assert hashlib.sha256((TREE/'epsilon_lex_repair.py').read_bytes()).hexdigest().upper()==manifest['source_hashes']['epsilon_lex_repair.py']
 r=Reader(process.pid)
 try:
  base,dll=r.module();runtime=base+exported_offset(dll,'_PyRuntime');interp=r.u64(runtime+32);threads=[];seen=set()
  for _ in range(16):
   if not interp or interp in seen:break
   seen.add(interp);thread=r.u64(interp+8)
   for _ in range(128):
    if not thread:break
    frame=r.u64(thread+24);frames=[];fs=set()
    for _ in range(128):
     if not frame or frame in fs:break
     fs.add(frame);info=frame_info(r,frame)
     if str(TREE).lower() in info['file'].lower() or info['file'].endswith('_milp.py'):frames.append(info)
     frame=r.u64(frame+24)
    if frames:threads.append(dict(thread_address=hex(thread),frames=frames))
    thread=r.u64(thread+8)
   interp=r.u64(interp)
  result=dict(unix=time.time(),pid=process.pid,method='READ_PROCESS_MEMORY_ONLY_NO_INJECTION_NO_SUSPEND',abi=test,threads=threads)
 finally:r.close()
 d=OUT/'readonly_diagnostics';d.mkdir(exist_ok=True);path=d/('stack_'+str(time.time_ns())+'.json')
 with path.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
 print(json.dumps(dict(path=str(path),**result),ensure_ascii=False))
if __name__=='__main__':main()

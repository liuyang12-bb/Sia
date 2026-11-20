import json,sys,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[2];sys.path.insert(0,str(ROOT/'sia-joint-full160/sia-simulator'))
from joint_prototype.r0_liveness import R0Oracle,APPLICATION_TEMPLATES
from joint_prototype.joint_profile import CPU_COORDS
manifest=json.loads((P.parent/'sia_r0_full160_comparison/attempt2/preregistration.json').read_text())
o=R0Oracle(manifest['registry'],manifest['models'],{a:a for a in APPLICATION_TEMPLATES},'medium');native=[]
for a,t in APPLICATION_TEMPLATES.items():
 q=o.template(t);slopes=np.diff(q.matrix,axis=0)/np.diff(CPU_COORDS)[:,None]
 native.append(dict(application=a,template=t,sha=q.source_sha256,min=float(q.matrix.min()),max=float(q.matrix.max()),cpu_negative_increments=int((np.diff(q.matrix,axis=0)<-1e-9).sum()),cpu_increasing_marginal_slopes=int((np.diff(slopes,axis=0)>1e-9).sum())))
points=json.loads((P/'feasible_response_points.json').read_text(encoding='utf-8'))
candidates=[x for x in points['cifar10/dgx-ext:1:4'] if x['cpu']*4+4<=24 and x['ram']*4+20.83*4<=500]
best=max(candidates,key=lambda x:x['rho'])
with (P/'native_support.json').open('x',encoding='utf-8') as f:json.dump(dict(native_templates=native,complement_witness=dict(bert_gpus=4,bert_per_gpu=[1,20.83],cifar10_gpus=4,cifar10_per_gpu=[best['cpu'],best['ram']],cifar10_rho=best['rho'],total_cpu=best['cpu']*4+4,total_ram=best['ram']*4+20.83*4)),f,indent=2)
hashes={str(x.relative_to(P)):hashlib.sha256(x.read_bytes()).hexdigest().upper() for x in P.rglob('*') if x.is_file() and '__pycache__' not in x.parts}
with (P/'artifact_hashes.json').open('x',encoding='utf-8') as f:json.dump(hashes,f,indent=2)
print('SEALED',len(hashes),'diagnostic files')

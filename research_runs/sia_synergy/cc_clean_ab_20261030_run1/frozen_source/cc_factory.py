"""The four existing experimental methods, no new scheduling semantics."""
import hashlib,importlib.metadata,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SIM=ROOT/'sia-simulator'
METHODS=('SIA_P','SIA_ECT_AUX','JOINT_CC_P','JOINT_CC','JOINT_CC_TAIL')
RESPONSES=('neutral','medium')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest().upper()

def verify():
    m=json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8'))
    for name,h in m['files'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('SOURCE_OR_INPUT_CHANGED: '+name)
    pins={line.split('==',1)[0].lower():line.split('==',1)[1]
          for line in (ROOT/'dependencies.lock.txt').read_text(encoding='utf-8').splitlines() if '==' in line}
    for name in ['numpy','scipy','pandas','cvxpy','cylp','psutil']:
        if importlib.metadata.version(name)!=pins[name]:raise RuntimeError('DEPENDENCY_VERSION_CHANGED: '+name)
    return m

def construct(method,response,workload_path=None):
    if method not in METHODS or response not in RESPONSES:raise ValueError('Unsupported comparison method/response')
    sys.path.insert(0,str(SIM))
    import pandas as pd
    from joint_prototype.full160 import make_cluster
    from joint_prototype.small_mechanism import ProbeSia
    from joint_prototype.physical_domain import restrict_single_node_domain
    from joint_prototype.completion_time import install_aux
    from joint_prototype.completion_continuity import install
    cfg=json.loads((ROOT/'inputs/config_symmetric.json').read_text(encoding='utf-8'))
    c=make_cluster(cfg,pd.read_csv(workload_path or ROOT/'inputs/workload.csv'),ROOT/'inputs/registry.json',ROOT/'inputs/models',
        'sia_p' if method.startswith('SIA') else 'joint_o',response,'main',True)
    c.policy.node_capacities={t:dict(cpu_cores=12.,ram_gb=250.) for t in ['aws','dgx-ext']}
    restrict_single_node_domain(c.policy)
    c.policy.aux_cpu_points=tuple(sorted(set(c.policy.aux_cpu_points)|{8.}))
    if method=='SIA_P':c.policy.__class__=ProbeSia
    elif method=='SIA_ECT_AUX':install_aux(c)
    else:install(c,proportional=method=='JOINT_CC_P')
    if method=='JOINT_CC_TAIL':
        from joint_prototype.tail_recovery import install as install_tail
        install_tail(c)
    return c

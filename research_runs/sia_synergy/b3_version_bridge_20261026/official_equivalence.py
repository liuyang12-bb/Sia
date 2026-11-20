import ast,hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
def functions(path):
    return {n.name:ast.dump(n,include_attributes=False) for c in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(c,ast.ClassDef) and c.name=='SiaPolicy' for n in c.body if isinstance(n,ast.FunctionDef)}
def main():
    a=functions(ROOT/'official-artifacts/sia-simulator/sia.py');b=functions(ROOT/'sia-joint-version-bridge/sia-simulator/sia.py')
    checks={name:dict(equal=a[name]==b[name],official_ast_sha256=hashlib.sha256(a[name].encode()).hexdigest(),bridge_ast_sha256=hashlib.sha256(b[name].encode()).hexdigest()) for name in a}
    # Windows compatibility/verbosity may exist outside the scientific AST;
    # report every function, rather than claim whole-file identity by name.
    with (OUT/'official_sia_ast_equivalence.json').open('x') as f:json.dump(checks,f,indent=2)
    print('OFFICIAL_SIA_AST',sum(v['equal'] for v in checks.values()),len(checks),[k for k,v in checks.items() if not v['equal']])
if __name__=='__main__':main()

"""Read-only inventory and dependency plan; no simulation or deletion."""
import ast,collections,hashlib,json,os,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'sia-joint-cc-sustained20/sia-simulator'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest().upper()
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def command(args):
    p=subprocess.run(args,cwd=ROOT,capture_output=True,encoding='utf-8',errors='replace')
    return dict(command=args,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
def main():
    folders=[]
    for d in sorted(ROOT.glob('sia-*')):
        if not d.is_dir():continue
        size=0;count=0;links=[]
        for current,dirs,names in os.walk(d,followlinks=False):
            for n in names:
                p=Path(current)/n
                if p.is_symlink():links.append(str(p));continue
                size+=p.stat().st_size;count+=1
        git=(d/'.git').exists()
        folders.append(dict(name=d.name,absolute_path=str(d.resolve()),bytes=size,files=count,symlinks=links,
            git=git,git_directory=(d/'.git').is_dir(),
            head=command(['git','-C',str(d),'rev-parse','HEAD']) if git else None,
            branch=command(['git','-C',str(d),'branch','--show-current']) if git else None,
            status=command(['git','-C',str(d),'status','--short','--branch','--untracked-files=all']) if git else None))
    modules={};trees={};allpy=[]
    for p in SOURCE.rglob('*.py'):
        if '__pycache__' in p.parts:continue
        rel=p.relative_to(SOURCE);allpy.append(rel.as_posix())
        mod='.'.join(rel.with_suffix('').parts)
        if rel.name=='__init__.py':mod=mod.removesuffix('.__init__')
        modules[mod]=rel.as_posix()
        if rel.parts[0]=='policies':modules[rel.stem]=rel.as_posix()
        if rel.parts[0]=='tests':modules[rel.stem]=rel.as_posix()
        trees[rel.as_posix()]=ast.parse(p.read_text(encoding='utf-8-sig'))
    def resolve(mod):
        # Include a local module/package and its parent initializer.
        names=[]
        while mod:
            if mod in modules:names.append(modules[mod])
            mod=mod.rpartition('.')[0]
        return names
    edges={}
    for rel,tree in trees.items():
        package=list(Path(rel).with_suffix('').parts[:-1]);deps=set()
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):
                for a in n.names:deps.update(resolve(a.name))
            elif isinstance(n,ast.ImportFrom):
                prefix='.'.join(package[:len(package)-n.level+1]) if n.level else ''
                target='.'.join(v for v in [prefix,n.module] if v)
                deps.update(resolve(target))
                for a in n.names:deps.update(resolve(target+'.'+a.name))
            elif isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='import_module':
                if n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str):deps.update(resolve(n.args[0].value))
        edges[rel]=sorted(deps)
    runtime_roots=['joint_prototype/completion_continuity.py','joint_prototype/completion_time.py',
        'joint_prototype/full160.py','joint_prototype/physical_domain.py','joint_prototype/small_mechanism.py',
        'run_stage2c_cell.py','check_stage1_legacy_regression.py']
    def closure(roots):
        seen=set();todo=list(roots)
        while todo:
            r=todo.pop()
            if r in seen:continue
            seen.add(r);todo.extend(edges.get(r,[]))
        return seen
    runtime=closure(runtime_roots)
    tests={p for p in allpy if p.startswith('tests/')}
    keep=closure(runtime|tests)
    # Gavel is eagerly imported by the inherited Cluster. Retain its complete
    # policy package, even unused implementations, rather than edit core code.
    keep.update(p for p in allpy if p.startswith(('policies/','utils/')))
    data=[p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob('*') if p.is_file() and
        (p.relative_to(SOURCE).parts[0] in ['traces','tests'] or
         (p.relative_to(SOURCE).parts[0]=='joint_prototype' and p.suffix=='.json')) and p.suffix not in ['.py','.pyc']]
    save(OUT/'inventory.json',dict(created_unix=time.time(),root=str(ROOT),source=str(SOURCE),folders=folders,
        roots=runtime_roots,runtime_python=sorted(runtime),regression_python=sorted(keep-runtime),
        keep_python=sorted(keep),exclude_python=sorted(set(allpy)-keep),keep_data=sorted(data),edges=edges,
        repo_worktrees={r:command(['git','-C',str(ROOT/r),'worktree','list','--porcelain']) for r in ['official-artifacts','sia-joint-experiments']},
        source_hashes={p:sha(SOURCE/p) for p in sorted(keep|set(data))}))
    print('INVENTORY',len(folders),'trees',round(sum(r['bytes'] for r in folders)/2**30,3),'GiB',
        'runtime_python',len(runtime),'regression+support',len(keep-runtime),'excluded_python',len(set(allpy)-keep),flush=True)
    for d in folders:print(d['name'],round(d['bytes']/2**20,1),'MiB','git' if d['git'] else 'snapshot')

if __name__=='__main__':main()

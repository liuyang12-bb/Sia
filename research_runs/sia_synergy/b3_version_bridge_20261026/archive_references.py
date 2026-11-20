import hashlib,json,zipfile
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
def main():
    p=json.loads((OUT/'preregistration.json').read_text(encoding='utf-8'))
    total=0
    with zipfile.ZipFile(OUT/'reference_artifacts.zip','x',zipfile.ZIP_STORED) as z:
        for name,h in p['baseline_hashes'].items():
            f=Path(name);assert hashlib.sha256(f.read_bytes()).hexdigest().upper()==h
            z.write(f,f.relative_to(ROOT));total+=f.stat().st_size
    print('ARCHIVED_REFERENCE_FILES',len(p['baseline_hashes']),'BYTES',total)
if __name__=='__main__':main()

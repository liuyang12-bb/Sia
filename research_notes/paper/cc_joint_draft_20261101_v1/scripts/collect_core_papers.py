"""Fetch identified open-access papers; no experiments or source modifications."""
from pathlib import Path
import hashlib, json, shutil, urllib.request
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
papers = [
 dict(key='sia2023', title='Sia: Heterogeneity-aware, goodput-optimized ML-cluster scheduling', authors='Suhas Jayaram Subramanya; Daiyaan Arfeen; Shouxu Lin; Aurick Qiao; Zhihao Jia; Gregory R. Ganger', year=2023, venue='SOSP', pages='642--657', doi='10.1145/3600006.3613175', url='https://suhasjs.github.io/publication/sia-sosp23', pdf_url='https://suhasjs.github.io/files/sia-sosp23.pdf', local=PROJECT/'official-artifacts/sia-sosp23.pdf'),
 dict(key='synergy2022', title='Looking Beyond GPUs for DNN Scheduling on Multi-Tenant Clusters', authors='Jayashree Mohan; Amar Phanishayee; Janardhan Kulkarni; Vijay Chidambaram', year=2022, venue='OSDI', pages='579--596', url='https://www.usenix.org/conference/osdi22/presentation/mohan', pdf_url='https://www.usenix.org/system/files/osdi22-mohan.pdf', local=PROJECT/'research_sources/Synergy/osdi22-mohan.pdf'),
 dict(key='pollux2021', title='Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning', authors='Aurick Qiao; Sang Keun Choe; Suhas Jayaram Subramanya; Willie Neiswanger; Qirong Ho; Hao Zhang; Gregory R. Ganger; Eric P. Xing', year=2021, venue='OSDI', pages='1--18', url='https://www.usenix.org/conference/osdi21/presentation/qiao', pdf_url='https://www.usenix.org/system/files/osdi21-qiao.pdf'),
 dict(key='gavel2020', title='Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads', authors='Deepak Narayanan; Keshav Santhanam; Fiodar Kazhamiaka; Amar Phanishayee; Matei Zaharia', year=2020, venue='OSDI', pages='481--498', url='https://www.usenix.org/conference/osdi20/presentation/narayanan-deepak', pdf_url='https://www.usenix.org/system/files/osdi20-narayanan_deepak.pdf'),
]
out = ROOT/'literature/papers'
out.mkdir(exist_ok=True)
manifest=[]
for p in papers:
    dest=out/(p['key']+'.pdf')
    if not dest.exists():
        if p.get('local'):
            shutil.copyfile(p['local'], dest)
        else:
            req=urllib.request.Request(p['pdf_url'],headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req,timeout=40) as response:
                data=response.read(20_000_001)
            if len(data)>20_000_000 or not data.startswith(b'%PDF'): raise ValueError('Unexpected download')
            dest.write_bytes(data)
    reader=PdfReader(dest)
    text='\n\n'.join(f'[PDF page {i+1}]\n'+(page.extract_text() or '') for i,page in enumerate(reader.pages))
    (out/(p['key']+'.txt')).write_text(text,encoding='utf-8')
    record={k:str(v) if isinstance(v,Path) else v for k,v in p.items()}
    record.update(file=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),bytes=dest.stat().st_size,pdf_pages=len(reader.pages),verification='publisher/author source checked 2026-11-01')
    manifest.append(record)
    print(p['key'],len(reader.pages),dest.stat().st_size)
(ROOT/'literature/core_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

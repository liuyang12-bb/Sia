"""Save public primary PDFs identified in the bibliography. Never bypass access controls."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request, json, hashlib
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
papers=json.loads((ROOT/'literature/all_sources.json').read_text(encoding='utf-8'))
def fetch(p):
    target=ROOT/'literature/papers'/(p['key']+'.pdf')
    rec={'key':p['key'],'url':p.get('pdf_url'),'title':p['title']}
    if not p.get('pdf_url'):
        return dict(rec,status='LINK_ONLY_NO_OPEN_FULLTEXT',primary_url=p['url'])
    try:
        if not target.exists():
            req=urllib.request.Request(p['pdf_url'],headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req,timeout=60) as response:data=response.read(20_000_001)
            if len(data)>20_000_000 or not data.startswith(b'%PDF'):raise ValueError('Not a bounded PDF')
            target.write_bytes(data)
        data=target.read_bytes()
        return dict(rec,status='SAVED',path=str(target),sha256=hashlib.sha256(data).hexdigest().upper(),bytes=len(data),pages=len(PdfReader(target).pages))
    except Exception as e:return dict(rec,status='DOWNLOAD_UNAVAILABLE',error=str(e),primary_url=p['url'])
with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(fetch,papers))
(ROOT/'literature/download_receipt.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
for r in rows:print(r['key'],r['status'],r.get('bytes',''))

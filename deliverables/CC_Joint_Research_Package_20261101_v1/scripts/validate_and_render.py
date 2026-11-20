from pathlib import Path
import csv,hashlib,json,re,zipfile
import pymupdf as fitz
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'evidence/paper_results.json').read_text(encoding='utf-8'))
assert len(data['cells'])==42 and all(x['aggregates_from_completion_arrival_match'] and x['exclusive_time_sum_matches_jct'] for x in data['checks'])
csv_jobs=list(csv.DictReader((ROOT/'evidence/all_jobs.csv').open(encoding='utf-8-sig',newline='')))
assert len(csv_jobs)==1020
for row in csv_jobs:
    assert row['completed']=='True'
    jct=float(row['completion'])-float(row['arrival'])
    assert abs(jct-float(row['jct']))<1e-7
    assert abs(jct-sum(float(row[x]) for x in ['wait','training','checkpoint']))<1e-6
for name in ['01_ALGORITHM_DESIGN_REPORT','02_EXPERIMENT_REPORT']:
    md=(ROOT/(name+'.md')).read_text(encoding='utf-8')
    for path in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',md):assert (ROOT/path).is_file()
    doc=fitz.open(ROOT/(name+'.pdf'))
    out=ROOT/'evidence/pdf_qa'/name;out.mkdir(parents=True,exist_ok=True)
    stats=[];pages=[]
    for i,p in enumerate(doc):
        dest=out/f'page-{i+1:02}.png'
        p.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(dest);pages.append(dest)
        excess=[]
        for block in p.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    if span['bbox'][0]<20 or span['bbox'][2]>p.rect.width-18:excess.append(span['text'])
        assert not excess,(name,i,excess)
        stats.append({'page':i+1,'text_chars':len(p.get_text()),'margin_exceptions':excess})
    for i in range(0,len(pages),2):
        ims=[Image.open(p).convert('RGB') for p in pages[i:i+2]]
        sheet=Image.new('RGB',(sum(im.width for im in ims),max(im.height for im in ims)+25),'#dddddd')
        left=0
        for im in ims:sheet.paste(im,(left,25));left+=im.width
        ImageDraw.Draw(sheet).text((10,5),name+f' pages {i+1}-{min(i+2,len(pages))}',fill='black')
        sheet.save(out/f'pair-{i//2+1:02}.png')
    (out/'checks.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    print(name,len(doc),'pages; no margin exceptions')
zip_path=ROOT/'03_CC_JOINT_CODE_AND_STRUCTURE.zip'
with zipfile.ZipFile(zip_path) as z:
    assert z.testzip() is None
    hashes=json.loads(z.read('provenance/source_hashes.json'))
    assert len(hashes)==329
    for path,h in hashes.items():assert hashlib.sha256(z.read('workspace/cc-joint-core80-full-lex-repair-v8/'+path)).hexdigest().upper()==h.upper()
result={'status':'PASS','cells':42,'job_rows':1020,'source_files_verified_in_zip':329,'source_code_changes':False,'new_simulations':0,'zip_sha256':hashlib.sha256(zip_path.read_bytes()).hexdigest().upper(),'note':'PDF visual review is recorded after manual inspection of rendered pages.'}
(ROOT/'evidence/validation_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS: event-derived metrics, CSVs, images, PDF bounds and source ZIP.')

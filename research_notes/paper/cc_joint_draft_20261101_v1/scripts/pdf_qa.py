from pathlib import Path
from PIL import Image,ImageDraw
import fitz,json
ROOT=Path(__file__).resolve().parents[1]
r=ROOT/'evidence/pdf_qa_final';r.mkdir(exist_ok=True)
doc=fitz.open(ROOT/'CC_JOINT_PAPER_DRAFT.pdf')
stats=[]
for i,p in enumerate(doc):
    p.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(r/f'page-{i+1:02}.png')
    blocks=p.get_text('dict')['blocks']
    bounds=[]
    for b in blocks:
        for line in b.get('lines',[]):
            for s in line['spans']:
                if s['bbox'][0]<20 or s['bbox'][2]>p.rect.width-18:bounds.append(s['text'])
    stats.append({'page':i+1,'text_chars':len(p.get_text()),'out_of_page_text':bounds})
pages=[r/f'page-{i+1:02}.png' for i in range(len(doc))]
for i in range(0,len(pages),2):
    images=[Image.open(p) for p in pages[i:i+2]]
    sheet=Image.new('RGB',(sum(im.width for im in images),max(im.height for im in images)+25),'#dddddd')
    x=0
    for im in images:sheet.paste(im,(x,25));x+=im.width
    ImageDraw.Draw(sheet).text((10,5),f'Pages {i+1} / {min(i+2,len(pages))}',fill='black')
    sheet.save(r/f'pair-{i//2+1:02}.png')
(r/'layout_checks.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
print(len(pages),'pages; margin exceptions:',sum(len(x['out_of_page_text']) for x in stats))

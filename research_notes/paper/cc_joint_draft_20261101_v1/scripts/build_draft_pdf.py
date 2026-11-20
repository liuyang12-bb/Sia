"""Render the Markdown manuscript to a reviewed Chinese PDF, without new experiments.

Run using bundled Codex Python (reportlab/pypdf); source Markdown remains editable.
"""
from pathlib import Path
import re, html, json
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether, CondPageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
pdfmetrics.registerFont(TTFont('Song',r'C:\Windows\Fonts\simsun.ttc',subfontIndex=0))
pdfmetrics.registerFont(TTFont('Hei',r'C:\Windows\Fonts\simhei.ttf'))
pdfmetrics.registerFontFamily('Song',normal='Song',bold='Hei',italic='Song',boldItalic='Hei')
W,H=A4;LEFT=48;RIGHT=48;WIDTH=W-LEFT-RIGHT
base=dict(fontName='Song',fontSize=10.2,leading=16.1,wordWrap='CJK',spaceAfter=6,splitLongWords=True)
styles={
 'body':ParagraphStyle('body',**base),
 'title':ParagraphStyle('title',fontName='Hei',fontSize=20,leading=29,spaceAfter=14,alignment=TA_CENTER,wordWrap='CJK'),
 'h1':ParagraphStyle('h1',fontName='Hei',fontSize=14,leading=20,spaceBefore=14,spaceAfter=8,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='Hei',fontSize=11.8,leading=18,spaceBefore=9,spaceAfter=5,keepWithNext=True),
 'caption':ParagraphStyle('caption',fontName='Song',fontSize=8.8,leading=13.2,wordWrap='CJK',spaceAfter=8),
 'small':ParagraphStyle('small',fontName='Song',fontSize=8.2,leading=12,wordWrap='CJK',spaceAfter=4,splitLongWords=True),
 'table':ParagraphStyle('table',fontName='Song',fontSize=8.3,leading=12,wordWrap='CJK',splitLongWords=True),
 'code':ParagraphStyle('code',fontName='Song',fontSize=9,leading=14,wordWrap='CJK',spaceAfter=3),
}
def inline(s):
    s=s.replace('\u2212','-').replace('\u2013','-')
    s=html.escape(s,quote=False)
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:f'<a href="{html.escape(m[2],quote=True)}" color="#175C7C">{m[1]}</a>',s)
    s=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'`([^`]+)`',r'\1',s)
    s=re.sub(r'\b([GCMFwBQHacm])_([njkt])\b',r'\1<sub>\2</sub>',s)
    return s
def para(s,kind='body'):return Paragraph(inline(s),styles[kind])
def figure(path,maxheight=430,scale=1,equation=False):
    iw,ih=ImageReader(str(path)).getSize();w=min(WIDTH,iw*(72/240 if equation else .5))*scale;h=w*ih/iw
    if h>maxheight:w*=maxheight/h;h=maxheight
    return Image(str(path),width=w,height=h,hAlign='CENTER')
def table(lines):
    rows=[[v.strip() for v in line.strip().strip('|').split('|')] for line in lines]
    rows=[r for r in rows if not all(re.fullmatch(r':?-+:?',v) for v in r)]
    n=len(rows[0]);widths=[WIDTH/n]*n
    if n==4:widths=[WIDTH*.15,WIDTH*.24,WIDTH*.27,WIDTH*.34]
    if n==6:widths=[WIDTH*.17]+[WIDTH*.166]*5
    data=[[Paragraph(('<b>'+inline(v)+'</b>') if i==0 else inline(v),styles['table']) for v in row] for i,row in enumerate(rows)]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='CENTER')
    t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E9EEF2')),('LINEABOVE',(0,0),(-1,0),.8,colors.HexColor('#334450')),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#334450')),('LINEBELOW',(0,-1),(-1,-1),.7,colors.HexColor('#334450')),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    return t

lines=(ROOT/'CC_JOINT_PAPER_DRAFT.md').read_text(encoding='utf-8').splitlines()
story=[];i=0;eq=0;inrefs=False
while i<len(lines):
    line=lines[i].strip()
    if not line:i+=1;continue
    if line.startswith('<!--'):i+=1;continue
    if line=='$$':
        eq+=1;i+=1
        while i<len(lines) and lines[i].strip()!='$$':i+=1
        story.extend([Spacer(1,3),figure(ROOT/f'figures/equations/eq{eq:02}.png',80,equation=True),Spacer(1,7)]);i+=1;continue
    if line.startswith('```'):
        code=[];i+=1
        while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
        story.append(CondPageBreak(190));story.append(KeepTogether([para(x,'code') for x in code]));i+=1;continue
    if line.startswith('|'):
        group=[]
        while i<len(lines) and lines[i].strip().startswith('|'):group.append(lines[i]);i+=1
        story.extend([Spacer(1,4),table(group),Spacer(1,8)]);continue
    image=re.match(r'!\[(.*?)\]\((.*?)\)',line)
    if image:
        path=ROOT/image[2];cap=para(image[1],'caption');group=[figure(path),Spacer(1,4),cap]
        story.append(KeepTogether(group));i+=1;continue
    if line.startswith('# '):story.append(para(line[2:],'title'));i+=1;continue
    if line.startswith('## '):
        inrefs='参考文献' in line
        story.append(para(line[3:],'h1'));i+=1;continue
    if line.startswith('### '):story.append(para(line[4:],'h2'));i+=1;continue
    kind='small' if inrefs else ('caption' if re.match(r'图\d+[：:]',line) else 'body')
    story.append(para(line,kind));i+=1

def footer(canvas,doc):
    canvas.saveState();canvas.setFont('Hei',8);canvas.setFillColor(colors.HexColor('#555555'))
    canvas.drawString(LEFT,H-26,'CC-Joint 研究论文初稿 v0.1')
    canvas.setFont('Helvetica',7);canvas.drawString(LEFT,24,'Synthetic cross-hardware transfer - not real Saturn performance')
    canvas.drawRightString(W-RIGHT,24,str(doc.page));canvas.restoreState()

pdf=ROOT/'CC_JOINT_PAPER_DRAFT.pdf'
doc=SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=LEFT,rightMargin=RIGHT,topMargin=43,bottomMargin=42,title='CC-Joint 面向异构GPU集群的有序启动与持续服务多资源联合调度',author='Research manuscript draft')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
r=PdfReader(pdf)
(ROOT/'evidence/pdf_build.json').write_text(json.dumps({'pdf':str(pdf),'pages':len(r.pages),'equations':eq,'source':'CC_JOINT_PAPER_DRAFT.md','note':'CJK embedded fonts; visual QA required.'},ensure_ascii=False,indent=2),encoding='utf-8')
print('Built',pdf,'pages',len(r.pages))

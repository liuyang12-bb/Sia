from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
core=json.loads((ROOT/'literature/core_sources.json').read_text(encoding='utf-8'))
supp=json.loads((ROOT/'literature/supplemental_sources.json').read_text(encoding='utf-8'))['papers']
coremap={p['key']:p for p in core}
papers=[coremap[k] for k in ['sia2023','pollux2021','synergy2022','gavel2020']]+supp
refs=[];bib=[]
for n,p in enumerate(papers,1):
    p['reference_number']=n
    authors=p['authors'].split('; ') if isinstance(p['authors'],str) else p['authors']
    names=', '.join(authors[:3])+(', et al.' if len(authors)>3 else '')
    label='[J]' if 'VLDB' in p['venue'] else '[C]'
    volume=f", {p['volume']}({p['number']})" if p.get('volume') else ''
    pages=p['pages'].replace('--','–')
    refs.append(f"[{n}] {names}. {p['title']}{label}. {p['venue']}, {p['year']}{volume}: {pages}. [出版方/作者来源]({p['url']})"+(f". DOI: {p['doi']}." if p.get('doi') else '.'))
    kind='article' if label=='[J]' else 'inproceedings'
    fields={'author':' and '.join(authors),'title':p['title'],'year':str(p['year']),'journal' if kind=='article' else 'booktitle':p['venue'],'pages':p['pages'],'url':p['url']}
    if p.get('doi'):fields['doi']=p['doi']
    if p.get('volume'): fields.update(volume=p['volume'],number=p['number'])
    bib.append('@'+kind+'{'+p['key']+',\n'+',\n'.join('  '+k+' = {'+v+'}' for k,v in fields.items())+'\n}')
(ROOT/'literature/references.bib').write_text('\n\n'.join(bib)+'\n',encoding='utf-8')
(ROOT/'literature/REFERENCES.md').write_text('# 核验参考文献\n\n'+'\n\n'.join(refs)+'\n',encoding='utf-8')
(ROOT/'literature/all_sources.json').write_text(json.dumps(papers,ensure_ascii=False,indent=2),encoding='utf-8')
draft=ROOT/'CC_JOINT_PAPER_DRAFT.md';text=draft.read_text(encoding='utf-8')
if '<!-- REFERENCES_INSERT -->' in text:draft.write_text(text.replace('<!-- REFERENCES_INSERT -->','\n\n'.join(refs)),encoding='utf-8')
print('Prepared',len(papers),'verified bibliography entries')

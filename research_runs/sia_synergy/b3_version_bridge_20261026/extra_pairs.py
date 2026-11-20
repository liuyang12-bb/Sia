import json
from pathlib import Path
from analyze import compare
OUT=Path(__file__).resolve().parent
def main():
    data=json.loads((OUT/'comparison_summary.json').read_text(encoding='utf-8'))['results'];out=[]
    for ds in ['original8','original20','holdout20']:
        for response in ['neutral','medium']:
            d={s['method']:s for s in data if s['dataset']==ds and s['response']==response}
            out.append(dict(dataset=ds,response=response,**compare(d['sia_p'],d['joint_g'])))
        for m in ['sia_p','sia_aux','b3_free','joint_g']:
            d={s['response']:s for s in data if s['dataset']==ds and s['method']==m}
            out.append(dict(dataset=ds,response='NEUTRAL_TO_MEDIUM',**compare(d['neutral'],d['medium'])))
    with (OUT/'additional_pairs.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,ensure_ascii=False)
    print('ADDITIONAL_PAIRS',len(out))
if __name__=='__main__':main()

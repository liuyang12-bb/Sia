"""Resume post-analysis after preserved descriptor path failure; never rerun cells."""
from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research_runs/sia_synergy/cc_cifar_bert12_20261029_run1'
v=json.loads((OUT/'validation.json').read_text());assert v['cells']==6 and v['completed_cells']==6
text=(OUT/'describe_response.py').read_text().replace('ROOT=Path(__file__).resolve().parents[1]','ROOT=Path(__file__).resolve().parents[3]')
target=OUT/'describe_response_v2.py';target.write_text(text,encoding='utf8')
with (OUT/'response_v2_console.log').open('x',encoding='utf8') as f:r=subprocess.run([sys.executable,'-X','utf8','-B',str(target)],stdout=f,stderr=subprocess.STDOUT)
assert r.returncode==0
old=(ROOT/'maintenance/finish_cifar_bert12_20261029.py').read_text()
head=old.split('def main():',1)[0];tail=old.split(' response=json.loads',1)[1]
exec(compile(head+'def main():\n response=json.loads'+tail,str(ROOT/'maintenance/finish_cifar_bert12_20261029.py'),'exec'),{'__file__':str(ROOT/'maintenance/finish_cifar_bert12_20261029.py'),'__name__':'__main__'})

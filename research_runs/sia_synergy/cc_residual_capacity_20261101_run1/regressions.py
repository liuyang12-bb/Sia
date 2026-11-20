import json,subprocess,sys
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[2];T=R/'cc-joint-residual-capacity-v1'
states={}
for name in ('official-artifacts','cc-joint-residual-capacity-v1'):
 command=[sys.executable,'-X','utf8','-B',str(T/'sia-simulator/check_stage1_legacy_regression.py'),'--probe',str(R/name/'sia-simulator')]
 p=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (O/('sia_probe_'+name+'.log')).write_bytes(p.stdout);assert p.returncode==0
 states[name]=json.loads(next(l.split('=',1)[1] for l in p.stdout.decode('utf8').splitlines() if l.startswith('PROBE_JSON=')))
assert len(states['official-artifacts'])==3 and states['official-artifacts']==states['cc-joint-residual-capacity-v1']
(O/'original_sia_regression.json').write_text(json.dumps(dict(exact=True,rounds=3,states=states),indent=2))
for method in ('FO','FULL'):
 # Probe full unchanged installation, candidate opt-in disabled.
 script=(O/'probe_full.py').read_text().replace("build('FULL'",f"build('{method}'")
 probe=O/('probe_'+method+'.py');probe.write_text(script)
 results={}
 for tree in ('cc-joint-uncertainty-v4','cc-joint-residual-capacity-v1'):
  p=subprocess.run([sys.executable,'-X','utf8','-B',str(probe),tree],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  (O/(tree+'_'+method+'_regression.log')).write_bytes(p.stdout);assert p.returncode==0
  results[tree]=json.loads(next(l.split('=',1)[1] for l in p.stdout.decode('utf8').splitlines() if l.startswith('FULL_PROBE=')))
 assert results['cc-joint-uncertainty-v4']==results['cc-joint-residual-capacity-v1']
 (O/(method+'_regression.json')).write_text(json.dumps(dict(exact=True,rounds=3),indent=2))
print('SIA/FO/FULL three-round exact regressions PASS')

"""Wait for the one fixed technical probe, then serial acceptance and launch."""
import json,time,sys
from pathlib import Path
from accept_and_launch_core80_repair_v3 import main
P=Path(__file__).resolve().parents[1]/'research_runs/sia_synergy/cc_core80_repair_20261031_run3/tests/real80_FULL_12_primal_dual/result.json'
stop=time.time()+3600
while not P.exists():
 if time.time()>=stop:raise RuntimeError('TECHNICAL_PROBE_WALL_GATE_NO_RESULT')
 time.sleep(2)
r=json.loads(P.read_text());assert r['status']=='PASS',r
raise SystemExit(main())

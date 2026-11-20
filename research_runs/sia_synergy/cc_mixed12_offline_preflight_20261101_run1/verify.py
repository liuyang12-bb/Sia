"""Independent offline receipts and arithmetic checks, no simulator imports."""
import ast,bisect,collections,csv,hashlib,json,math,unittest
from pathlib import Path
O=Path(__file__).resolve().parent;ROOT=O.parents[2];TREE=ROOT/'cc-joint-recovery-equivalent-v1'
def read(n):return json.loads((O/n).read_text(encoding='utf8'))
P=read('preregistration.json');CFG=read('inputs/config.json');COV=read('coverage.json');REC=read('reconciled_preflight.json')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def matrix(name):
    tree=ast.parse((TREE/'inputs/models'/name).read_text(encoding='utf8'))
    for n in ast.walk(tree):
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='tput' for t in n.targets):
            return ast.literal_eval(n.value.args[0])
    raise ValueError(name)
def response(m,c,r):
    cx=(1,2,3,4,5,6,9,12,24);rx=(20.83,62.5,125,187.5,250)
    c=max(cx[0],min(cx[-1],c));r=max(rx[0],min(rx[-1],r))
    i=min(max(bisect.bisect_right(cx,c)-1,0),len(cx)-2)
    j=min(max(bisect.bisect_right(rx,r)-1,0),len(rx)-2)
    a=(c-cx[i])/(cx[i+1]-cx[i]);b=(r-rx[j])/(rx[j+1]-rx[j])
    return (1-a)*((1-b)*m[i][j]+b*m[i][j+1])+a*((1-b)*m[i+1][j]+b*m[i+1][j+1])
class OfflineChecks(unittest.TestCase):
    def test_source_and_inputs_unchanged(self):
        self.assertEqual(len(P['source_hashes']),344)
        for name,h in P['source_hashes'].items():self.assertEqual(digest(TREE/name),h,name)
        for name,h in P['input_hashes'].items():self.assertEqual(digest(name),h,name)
    def test_source_workload_and_controlled_arrivals(self):
        with (ROOT/'official-artifacts/sia-simulator/workloads/saturn/workload-1.csv').open(encoding='utf8') as f:src=list(csv.DictReader(f))
        with (O/'inputs/workload.csv').open(encoding='utf8') as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),12);self.assertEqual(len(src),160)
        for i,r in enumerate(rows):
            orig=src[int(r['original_source_row_index'])]
            for k,v in orig.items():
                if k!='time':self.assertEqual(r[k],v)
            self.assertEqual(float(r['time']),0 if i<6 else 600)
            self.assertEqual(float(r['original_arrival']),float(orig['time']))
    def test_all_job_config_pairs_and_prop_feasible(self):
        self.assertEqual(len(COV['rows']),72)
        self.assertEqual(len({(x['job'],x['config']) for x in COV['rows']}),72)
        for x in COV['rows']:
            typ,_,g=x['config'].split(':');g=int(g);q=x['proportional'];caps=CFG['cluster']['capacity_scenarios']['main'][typ]
            self.assertGreater(x['columns'],0)
            self.assertLessEqual(g,CFG['cluster']['ngpus_per_node'][typ])
            self.assertLessEqual(g*q['cpu_per_gpu'],caps['cpu_cores']+1e-8)
            self.assertLessEqual(g*q['ram_gb_per_gpu'],caps['ram_gb']+1e-8)
    def test_rho_from_source_matrix_independent(self):
        templates={app:matrix(name) for app,name in CFG['response']['application_template_rule'].items() if app in ('cifar10','deepspeech2','bert')}
        for x in COV['rows']:
            q=x['proportional'];m=templates[x['application']]
            num=response(m,q['requested_source_cpu_coordinate'],q['requested_source_ram_coordinate'])
            den=response(m,q['requested_baseline_source_cpu_coordinate'],q['requested_baseline_source_ram_coordinate'])
            self.assertAlmostEqual(q['rho'],math.sqrt(num/den),places=11)
        for state in ('cold','reference_E0'):
            for method in ('FO','FULL'):
                for name,q in read(f'oneshot_v2_{state}_{method}.json')['selected'].items():
                    app=next(v['application'] for v in P['workload'] if v['name']==name)
                    num=q['source_coordinates']['requested'];den=q['baseline_source_coordinates']['requested']
                    self.assertAlmostEqual(q['rho'],math.sqrt(response(templates[app],num['cpu'],num['ram_gb'])/response(templates[app],den['cpu'],den['ram_gb'])),places=11)
    def test_static_and_one_shot_capacity_independent(self):
        selections=[]
        for state in ('E0','E25'):
            for label in ('PROPORTIONAL_GPU_PROPOSAL','FO_AUX_OPTIMUM','FULL_RESOURCE_OPTIMUM'):
                r=read(state+'_'+label+'.json');self.assertEqual(r['status'],0);self.assertEqual(r['gap'],0.)
                selections.append(r['selected'])
        for state in ('cold','reference_E0'):
            for method in ('FO','FULL'):selections.append(REC['one_shot'][state][method]['selected'])
        for selected in selections:
            sums=collections.defaultdict(lambda:[0.,0.,0.]);seen=set()
            for x in selected:
                self.assertNotIn(x['job'],seen);seen.add(x['job'])
                placement=x['placement'].items() if isinstance(x['placement'],dict) else x['placement']
                count=0
                for node,g in placement:
                    count+=g;sums[node][0]+=g;sums[node][1]+=g*x['cpu_per_gpu'];sums[node][2]+=g*x['ram_per_gpu']
                self.assertEqual(count,int(x['config'].split(':')[-1]))
            for node,v in sums.items():
                t=node.split(':')[0];cap=CFG['cluster']['capacity_scenarios']['main'][t]
                for used,limit in zip(v,(CFG['cluster']['ngpus_per_node'][t],cap['cpu_cores'],cap['ram_gb'])):self.assertLessEqual(used,limit+1e-7)
    def test_static_fo_gpu_freeze(self):
        for state in ('E0','E25'):
            a=read(state+'_PROPORTIONAL_GPU_PROPOSAL.json')['selected'];b=read(state+'_FO_AUX_OPTIMUM.json')['selected']
            self.assertEqual({x['job']:x['config'] for x in a},{x['job']:x['config'] for x in b})
    def test_no_training_actual_calls_and_common_contract(self):
        for state in ('cold','reference_E0'):
            a=read(f'oneshot_v2_{state}_FO.json');b=read(f'oneshot_v2_{state}_FULL.json')
            for r in (a,b):self.assertEqual(r['training_steps'],0);self.assertEqual(r['time_advanced'],0);self.assertTrue(r['valid'])
            for k in ('response_exponent','registry_sha256','reference','scoring','startup_budget_scope','cold_start','service'):
                self.assertEqual(a['contracts'][k],b['contracts'][k],k)
            self.assertEqual(a['declared_work'],b['declared_work'])
    def test_donor_loss_not_omitted(self):
        for state in ('E0','E25'):
            for w in read('conditional_opportunities.json')[state]['windows'].values():
                self.assertAlmostEqual(w['net'],w['receivers']-w['donor_losses'],places=12)
                self.assertAlmostEqual(w['net'],sum(w['per_job_net_fraction'].values()),places=12)
        for w in REC['one_shot']['reference_E0']['held_bundle_conditional_windows'].values():
            self.assertAlmostEqual(w['net'],w['receivers']-w['donor_loss'],places=12)
            self.assertGreater(w['donor_loss'],0)

if __name__=='__main__':unittest.main(verbosity=2)

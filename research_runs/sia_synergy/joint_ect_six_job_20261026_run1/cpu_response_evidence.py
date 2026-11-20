"""All six jobs and legal GPU configs: fixed-RAM CPU response, no training."""
import csv,json,sys
from experiment import OUT,load,save

def main():
    load();sys.path.insert(0,str(OUT/'source'))
    from joint_prototype.r0_liveness import R0Oracle
    from joint_prototype.joint_profile import JobSpec,GpuConfig
    work=list(csv.DictReader((OUT/'inputs/workload.csv').open(encoding='utf-8')))
    oracle=R0Oracle(OUT/'inputs/registry.json',OUT/'inputs/models',{r['name']:r['application'] for r in work},'medium')
    result=[]
    for job in work:
      for cid in ['aws:1:1','aws:1:2','dgx-ext:1:1','dgx-ext:1:2']:
        row=oracle.reference(job['name'],cid);typ,nn,ng=cid.split(':');g=int(ng)
        cfg=GpuConfig(cid,typ,g,int(nn),1.,row['phi_c'],row['phi_m'])
        spec=JobSpec(job['name'],row['template'],(cfg,))
        baseline=oracle.query(spec,cfg,6.,125.,(6.,125.))
        for cpu in [1.,2.,3.,4.,6.,8.,9.,12.,24.]:
          q=oracle.query(spec,cfg,cpu,125.,(6.,125.))
          result.append(dict(job=job['name'],app=job['application'],config=cid,template=row['template'],
            phi_c=row['phi_c'],phi_m=row['phi_m'],cpu_per_gpu=cpu,ram_per_gpu=125.,
            baseline_cpu=6.,baseline_rho=baseline.rho,rho=q.rho,relative_speed=q.rho/baseline.rho-1,
            node_feasible=cpu*g<=12,source_clipped=q.source_coordinate_clipped,
            reference_clipped=q.baseline_coordinate_clipped))
    with (OUT/'cpu_response_evidence.csv').open('x',encoding='utf-8',newline='') as stream:
      writer=csv.DictWriter(stream,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
    save(OUT/'cpu_response_evidence.json',dict(training_run=False,records=result,
      note='Response ratio only, no current GPU performance injected. Includes infeasible and clipped points explicitly.'))
    print('CPU_RESPONSE_RECORDS',len(result))

if __name__=='__main__':main()

"""Offline profile arithmetic only; never calls Job.step or any scheduler.

Reference: declared-batch, fixed one-GPU execution, faster of the two types.
This policy-independent composition screen is NOT an online information source.
All-type alternatives are retained so the composition is not called invariant.
Pair model uses whole-job average rates with explicit startup and a release at
the first completion. It is an analytical opportunity, not simulated JCT proof.
"""
import csv,math
from experiment import OUT,save,sha

def reference_gate(oracle,jobs):
    from applications import APPLICATIONS
    from simulator_config import SLOWDOWNS
    from joint_prototype.joint_profile import JobSpec,GpuConfig
    from joint_prototype.r0_liveness import APPLICATION_TEMPLATES
    raw={};sources=[];responses={}
    for j in jobs:
      appname=j['application'];name=j['name'];batch=int(j['batch_size'])
      for typ in ['aws','dgx-ext']:
       app=APPLICATIONS[typ][appname]
       for g in [1,2]:
        # Same declared target batch, original accumulation/throughput formula,
        # epoch gradient gain and original work. No scheduler or execution loop.
        throughput=app.get_throughput_with_accum((g,),batch);seconds=0.;epochs=[]
        for ep in range(app.max_epochs):
          sq,var=app.get_grad_stats(batch,ep);scale=batch/app.init_batch_size
          gain=(var+sq)/(var/scale+sq)
          rate=gain*throughput/batch/SLOWDOWNS[appname]
          work=app.get_progress(ep+1)-app.get_progress(ep)
          dt=work/rate;seconds+=dt
          epochs.append(dict(epoch=ep,work=work,raw_rate=rate,seconds=dt))
        cid=f'{typ}:1:{g}';raw[name,cid]=seconds
        ref=oracle.reference(name,cid);cfg=GpuConfig(cid,typ,g,1,1.,ref['phi_c'],ref['phi_m'])
        spec=JobSpec(name,APPLICATION_TEMPLATES[appname],(cfg,))
        for cpu in [4.,6.,8.,12.]:
          if cpu*g>12:continue
          q=oracle.query(spec,cfg,cpu,125.,(6.,125.));responses[name,cid,cpu]=q
        sources.append(dict(job=name,application=appname,config=cid,batch=batch,
            raw_training_seconds=seconds,total_progress=app.get_progress(app.max_epochs),
            epochs=epochs,reference_not_runtime_batch_adaptation=True))
    jobsref=[];shares={}
    for j in jobs:
      times={typ:raw[j['name'],f'{typ}:1:1']/responses[j['name'],f'{typ}:1:1',6.].rho for typ in ['aws','dgx-ext']}
      jobsref.append(dict(job=j['name'],application=j['application'],one_gpu_type_seconds=times,
          fastest_one_gpu_seconds=min(times.values())))
    for mode in ['aws','dgx-ext','fastest_one_gpu']:
      totals={a:sum((r['fastest_one_gpu_seconds'] if mode=='fastest_one_gpu' else r['one_gpu_type_seconds'][mode])
          for r in jobsref if r['application']==a) for a in ['cifar10','deepspeech2']}
      shares[mode]=dict(seconds=totals,cifar_sensitive_share=totals['cifar10']/sum(totals.values()))
    ja=next(j['name'] for j in jobs if j['application']=='cifar10')
    jb=next(j['name'] for j in jobs if j['application']=='deepspeech2')
    witnesses=[]
    for typ in ['aws','dgx-ext']:
      cid=f'{typ}:1:1';a0=raw[ja,cid];b0=raw[jb,cid]
      def q(j,c):return responses[j,cid,float(c)]
      supported=all(not q(j,c).source_coordinate_clipped and not q(j,c).baseline_coordinate_clipped
          for j,c in [(ja,6),(ja,8),(jb,6),(jb,4),(jb,12)])
      def pair(ca,cb):
        ta=50+a0/q(ja,ca).rho;tb=25+b0/q(jb,cb).rho
        assert ta<tb,'Pair phase formula requires CIFAR to finish first'
        # DS starts at25s; consumes that fraction before CF completes. Then
        # auxiliary-only change has zero extra delay in the inherited model.
        fraction=(ta-25)*q(jb,cb).rho/b0
        finish_b=ta+(1-fraction)*b0/q(jb,12).rho
        return dict(cifar_completion=ta,ds_completion=finish_b,mean=(ta+finish_b)/2,
          overlap_seconds=ta-50,donor_fraction_complete_before_release=fraction)
      base=pair(6,6);alt=pair(8,4)
      # Compare serialized DS2 then CF2 to a split GPU plan; refuse clipped
      # numerator/denominator in this primary witness.
      c2=f'{typ}:1:2'
      q2a=responses[ja,c2,6.];q2b=responses[jb,c2,6.]
      serial_b=25+raw[jb,c2]/q2b.rho
      serial_a=serial_b+50+raw[ja,c2]/q2a.rho
      # DS changes from1 to2 GPUs after CF completion, charging real25s.
      split_a=alt['cifar_completion'];f=alt['donor_fraction_complete_before_release']
      split_b=split_a+25+(1-f)*raw[jb,c2]/q2b.rho
      two_supported=all(not q.source_coordinate_clipped and not q.baseline_coordinate_clipped for q in [q2a,q2b])
      witnesses.append(dict(gpu_type=typ,jobs=[ja,jb],fully_supported=supported,
        proportional=base,exchange=alt,cpu=[8,4],ram=[125,125],
        receiver_saving=base['cifar_completion']-alt['cifar_completion'],
        donor_delay=alt['ds_completion']-base['ds_completion'],
        mean_saving=base['mean']-alt['mean'],
        gpu_coupling=dict(supported=supported and two_supported,
          serial_DS2_then_CF2_mean=(serial_a+serial_b)/2,
          split_CF1_DS1_then_DS2_mean=(split_a+split_b)/2,
          transition_delay=25,mean_saving=(serial_a+serial_b-split_a-split_b)/2)))
    share=shares['fastest_one_gpu']['cifar_sensitive_share']
    gates=dict(reference_time_share_40_to_60=.4<=share<=.6,
      supported_net_completion_exchange=any(w['fully_supported'] and w['mean_saving']>0 for w in witnesses),
      supported_gpu_coupling=any(w['gpu_coupling']['supported'] and w['gpu_coupling']['mean_saving']>0 for w in witnesses),
      exchange_longer_than_max_startup=all(w['exchange']['overlap_seconds']>50 for w in witnesses))
    save(OUT/'offline_reference_detail.json',dict(training_run=False,source_fingerprint='unchanged',
      sources=sources,jobs_reference=jobsref,shares=shares,witnesses=witnesses,gates=gates,
      definition='Declared-batch epochwise offline Sia formula; composition uses faster one-GPU proportional type, uniformly for every job. Alternatives retained. Pair uses average rates, not a global guarantee.',
      excludes='No future observations passed to actual policy; runtime batch adaptation unchanged.'))
    return dict(valid=all(gates.values()),gates=gates,shares=shares,witnesses=witnesses,
      detail_sha256=sha(OUT/'offline_reference_detail.json'))

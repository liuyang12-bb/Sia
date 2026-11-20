"""Unmodified official Job.step reproduces zero-second completion rounding."""
import copy,importlib.util,unittest
from pathlib import Path
from applications import APPLICATIONS
from job import Job
from joint_prototype.execution_audit import audit_execution

ROOT=Path(__file__).resolve().parents[3]
def configured(cls):
    apps={t:APPLICATIONS[t]['cifar10'] for t in ['aws','dgx-ext','rtx']}
    j=cls('cifar-quantized-boundary',apps,0,target_num_replicas=4,target_batch_size=4096)
    j.migrate('rtx',[4]);j.current_time=660;j.rescale_time=50
    a=apps['rtx'];bsz=sum(j.placement)*j.atomic_bsz*(j.accum_steps+1)
    j.epoch=a.get_completion_epoch(bsz);j.epoch_duration=[0]*j.epoch
    gs,gv=a.get_grad_stats(bsz,j.epoch);scale=bsz/a.init_batch_size
    gain=(gv+gs)/(gv/scale+gs);step,sync=a.get_throughput((4,),j.atomic_bsz)
    raw=gain/(step+(step-sync)*j.accum_steps)*j.calibration_factor
    j.progress=a.get_progress(j.epoch+1)-.25*raw
    j.enable_joint_accounting=True
    return j
def event():
    j=configured(Job);before=j.progress;j.step(60)
    return dict(interval_start=660,interval_end=720,arrival_time=0,observed_active_seconds=50,
        waiting_seconds=0.,checkpoint_delay_consumed=50.,training_seconds=0.,progress_delta=j.progress-before,
        rho=1.,completion_time=j.completion_time,training_segments=j.joint_step_segments),j

class CheckpointRoundingTests(unittest.TestCase):
    def test_official_step_matches_quantized_completion_after_checkpoint(self):
        module=importlib.util.spec_from_file_location('untouched_official_job_rounding',ROOT/'official-artifacts/sia-simulator/job.py')
        official=importlib.util.module_from_spec(module);module.loader.exec_module(official)
        old=configured(official.Job);initial=old.progress;old.step(60)
        e,j=event()
        self.assertEqual((old.completion_time,old.progress,old.epoch,old.execution_time),
                         (j.completion_time,j.progress,j.epoch,j.execution_time))
        self.assertEqual(old.completion_time,710);self.assertEqual(old.execution_time,0)
        self.assertGreater(old.progress,initial)
        self.assertTrue(audit_execution(e)['valid'])

    def test_segment_cannot_advance_inside_checkpoint(self):
        e,_=event();e['training_segments'][0]['start_time']=709
        self.assertFalse(audit_execution(e)['valid'])

    def test_full_requested_checkpoint_cannot_hide_rounding_progress(self):
        e,_=event();e.update(checkpoint_delay_consumed=60.,observed_active_seconds=60.,completion_time=720)
        e['training_segments'][0]['start_time']=720
        self.assertFalse(audit_execution(e)['valid'])

    def test_quantization_residual_must_stay_within_half_second_rate(self):
        e,_=event();v=e['training_segments'][0];v['progress_delta']=v['effective_rate'];v['rounding_progress_residual']=v['progress_delta'];e['progress_delta']=v['progress_delta']
        self.assertFalse(audit_execution(e)['valid'])

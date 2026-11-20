"""Opt-in resource exchange and fixed-window net-progress scoring.

Does not change L2 admission, service clocks, execution rho, or startup physics.
Window scoring REPLACES the historic restart proxy, never stacks with it.
Relaxing the per-round proportional progress guarantee is an explicit policy
choice, not a claim of preserving Synergy's per-job non-degradation guarantee.
"""
import math
from dataclasses import asdict, replace
import simulator_config as sc
from sia import ZERO_ALLOC_GAIN
from .bundle_adaptation import BundleJoint, install as install_bundle
from .full160 import FullCluster


def fair_utility(rate, p=-.5, lambda_n=1.1):
    if not math.isfinite(rate) or rate < 0 or p != -.5 or lambda_n != 1.1:
        raise ValueError('INVALID_OR_UNREGISTERED_FAIRNESS_INPUT')
    powered=min(round(max(rate,ZERO_ALLOC_GAIN)**p,2),sc.sia_goodput_clip_val)
    return lambda_n-powered


def window_utility(rate, delay, horizon=300., p=-.5, lambda_n=1.1):
    if not math.isfinite(delay) or delay < 0 or horizon != 300.:
        raise ValueError('INVALID_OR_UNREGISTERED_WINDOW')
    return fair_utility(rate*max(horizon-delay,0.)/horizon,p,lambda_n)


def configure_scheduler(scheduler, allow_exchange):
    scheduler.enforce_floor=not allow_exchange
    return scheduler


class ExchangePolicy(BundleJoint):
    def _scheduler(self,nodes,specs):
        return configure_scheduler(super()._scheduler(nodes,specs),self.allow_exchange)

    def factor_columns(self,jobs,specs,s):
        original,proposals=super().factor_columns(jobs,specs,s)
        by={j.job_id:j for j in specs}
        scored=[];records=[]
        for column in original:
            job=by[column.job_id]
            same=(job.current_config_id==column.config.config_id and
                  job.current_placement==column.placement.node_gpus)
            if job.job_id not in self.remaining_delay:
                raise RuntimeError('MISSING_CURRENT_CHECKPOINT_DELAY:'+job.job_id)
            remaining=float(self.remaining_delay[job.job_id])
            if not math.isfinite(remaining) or remaining<0:
                raise RuntimeError('INVALID_CURRENT_CHECKPOINT_DELAY:'+job.job_id)
            # Same GPU geometry includes CPU/RAM-only changes: no added delay.
            delay=remaining if same else self.liveness._delay(job.job_id,column)
            rate=column.query.effective_goodput  # already contains rho exactly once
            if not math.isfinite(rate) or rate<=0:
                raise RuntimeError('NONPOSITIVE_CANDIDATE_RATE:'+job.job_id)
            value=(window_utility(rate,delay,p=self.p_fairness,lambda_n=self.lambda_n)
                   if self.net_progress else column.scheduling_utility)
            scored.append(replace(column,scheduling_utility=value))
            records.append(dict(job=job.job_id,column_id=column.column_id,
                same_gpu_geometry=same,current_delay=remaining,candidate_delay=delay,
                normalized_rate_before_delay=rate,net_normalized_rate=rate*max(300.-delay,0.)/300.,
                previous_proxy_utility=column.scheduling_utility,utility=value))
        self.exchange_columns=scored
        self.exchange_scores=records
        return scored,proposals

    def optimize(self,*args):
        result=super().optimize(*args)
        self.last_decision['shared_contract'].update(
            policy_version='JOINT_EXCHANGE_NETPROGRESS_V1',allow_exchange=self.allow_exchange,
            enforce_proportional_progress_floor=not self.allow_exchange,
            net_progress=self.net_progress,horizon_seconds=300.,
            restart_scoring='ACTUAL_WINDOW_DELAY_ONLY' if self.net_progress else 'HISTORIC_PROXY',
            admission_unchanged=True)
        # EvidenceMixin's historical probe is diagnostic only. Replace its
        # candidate list/contract with THIS policy's actual scored candidates.
        self.last_decision['probe']['joint_columns']=[asdict(c) for c in self.exchange_columns]
        self.last_decision['probe']['contract'].update(
            progress_floor=not self.allow_exchange,net_progress=self.net_progress)
        self.last_decision['net_progress_scores']=self.exchange_scores
        return result


class ExchangeCluster(FullCluster):
    def optimize_policy(self):
        self.policy.remaining_delay={j.name:float(j.rescale_time) for j in self.jobs}
        return super().optimize_policy()


def install(cluster,allow_exchange=True,net_progress=True):
    install_bundle(cluster)
    cluster.__class__=ExchangeCluster
    cluster.policy.__class__=ExchangePolicy
    cluster.policy.allow_exchange=bool(allow_exchange)
    cluster.policy.net_progress=bool(net_progress)
    cluster.policy.remaining_delay={}
    return cluster

import numpy as np

from sia import SiaPolicy, ZERO_ALLOC_GAIN


def get_las_attained_service(job_info):
    """Return productive GPU-seconds, excluding queueing and restart waste."""
    if not hasattr(job_info, "used_gpu_seconds"):
        raise AttributeError(
            "sia_las requires JobInfo.used_gpu_seconds from "
            "Cluster.get_sia_multi_job_info()"
        )
    return float(job_info.used_gpu_seconds)


def is_high_priority(attained_service, threshold_gpu_seconds):
    return attained_service < threshold_gpu_seconds


def apply_las_priority_factor(
        goodput_matrix, high_priority_rows, priority_factor):
    """Scale valid Goodput entries while preserving Sia's zero-allocation floor.

    This is a Sia/LAS composition, not a formula from Tiresias. Multiplication
    happens before Sia's p transform, so factor > 1 improves preference both
    for p < 0 (minimization) and p > 0 (maximization).
    """
    if priority_factor <= 0:
        raise ValueError("las_priority_factor must be greater than zero")
    if priority_factor == 1.0 or goodput_matrix.size == 0:
        return goodput_matrix

    result = np.copy(goodput_matrix)
    for row_idx, high_priority in enumerate(high_priority_rows):
        if not high_priority:
            continue
        valid = result[row_idx, :] > ZERO_ALLOC_GAIN
        result[row_idx, valid] *= priority_factor
    return result


class SiaLASPolicy(SiaPolicy):
    """Sia with a two-class, attained-service-derived Goodput preference.

    Tiresias motivates using productive GPU-seconds as attained service. This
    weighted-Goodput adaptation does not reproduce Tiresias's MLFQ scheduler,
    FIFO-within-queue ordering, skip behavior, or starvation promotion.
    """

    def __init__(self, *args, las_threshold_gpu_seconds=3600,
                 las_priority_factor=1.0, **kwargs):
        if las_threshold_gpu_seconds < 0:
            raise ValueError(
                "las_threshold_gpu_seconds must be non-negative"
            )
        if las_priority_factor <= 0:
            raise ValueError("las_priority_factor must be greater than zero")
        super().__init__(*args, **kwargs)
        self.las_threshold_gpu_seconds = float(
            las_threshold_gpu_seconds
        )
        self.las_priority_factor = float(las_priority_factor)
        self._las_high_priority = {}
        print(
            "Sia-LAS params: "
            f"threshold_gpu_seconds={self.las_threshold_gpu_seconds}, "
            f"priority_factor={self.las_priority_factor}"
        )

    def optimize_mip(self, jobs, nodes, prev_allocations):
        self._las_high_priority = {
            jobname: is_high_priority(
                get_las_attained_service(job),
                self.las_threshold_gpu_seconds,
            )
            for jobname, job in jobs.items()
        }
        return super().optimize_mip(jobs, nodes, prev_allocations)

    def _apply_objective_goodput_priorities(
            self, adaptive_jobnames, adaptive_speedup_mat,
            rigid_jobnames, rigid_speedup_mat):
        adaptive_rows = [
            self._las_high_priority[name] for name in adaptive_jobnames
        ]
        rigid_rows = [
            self._las_high_priority[name] for name in rigid_jobnames
        ]
        adaptive_speedup_mat = apply_las_priority_factor(
            adaptive_speedup_mat,
            adaptive_rows,
            self.las_priority_factor,
        )
        rigid_speedup_mat = apply_las_priority_factor(
            rigid_speedup_mat,
            rigid_rows,
            self.las_priority_factor,
        )
        return adaptive_speedup_mat, rigid_speedup_mat

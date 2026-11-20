"""Independent execution invariants using raw rates captured BEFORE rho."""
import math


def audit_execution(event):
    errors = []
    segments = event["training_segments"]
    def close(a, b):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8)
    if not close(sum(s["seconds"] for s in segments), event["training_seconds"]):
        errors.append("segment time differs from execution_time delta")
    if not close(sum(s["progress_delta"] for s in segments), event["progress_delta"]):
        errors.append("segment progress differs from Job progress delta")
    for s in segments:
        raw, rho, rate, dt = s["raw_rate_before_rho"], s["rho"], s["effective_rate"], s["seconds"]
        if not all(math.isfinite(v) for v in (raw, rho, rate, dt)) or min(raw, rho, rate, dt) < 0:
            errors.append("non-finite/negative rate or time")
        if not close(rho, event["rho"]) or not close(rate, raw*rho):
            errors.append("rho not applied exactly once to directly recorded raw rate")
        residual = s["progress_delta"] - raw*rho*dt
        if s["epoch_after"] == s["epoch_before"]:
            if not close(residual, 0):
                errors.append("non-boundary progress differs from rate integral")
        else:
            # Original Job.step rounds boundary duration to the nearest second.
            if s["epoch_after"] != s["epoch_before"] + 1 or abs(residual) > 0.5*rate + 1e-7:
                errors.append("epoch-boundary residual exceeds Sia rounding bound")
        if not close(residual, s["rounding_progress_residual"]):
            errors.append("incorrect recorded rounding residual")
    observed = event["observed_active_seconds"]
    if not close(event["waiting_seconds"] + event["checkpoint_delay_consumed"] + event["training_seconds"], observed):
        errors.append("non-exclusive observed time")
    if event["checkpoint_delay_consumed"] >= observed > 0 and event["progress_delta"] != 0:
        errors.append("progress during full checkpoint interval")
    if event["completion_time"] is not None:
        for s in segments:
            if s["start_time"] + s["seconds"] > event["completion_time"] + 1e-8:
                errors.append("training after completion")
    return {"valid": not errors, "violations": errors, "raw_evidence": "captured_before_rho_not_effective_divided_by_rho"}

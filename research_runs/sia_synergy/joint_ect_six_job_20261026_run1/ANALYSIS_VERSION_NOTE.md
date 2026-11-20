# Independent auditor field correction

Before the first cell result was read, code inspection found that the original
preregistered auditor expected per-segment progress_before/progress_after fields.
The actual unchanged Job.step schema stores progress_delta and a bounded rounding
residual. Preserve audit.py and its preregistered hash. Use audit_v2.py to check
progress_delta = effective_rate * seconds + rounding_progress_residual, together
with the existing independent execution audit's segment-residual bounds.

This is an analysis-schema correction only. No source, policy, input, response,
solver, capacity, observation window or running experiment script is changed.
Archive and hash both auditors. Original execution evidence is not rewritten.

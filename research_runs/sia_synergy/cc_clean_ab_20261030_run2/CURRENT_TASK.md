# Current authorized task: clean A/B ablations and C offline precheck

This new source snapshot starts byte-identical to accepted cc-joint-ample-lex-repair (289 files). Historical directories, official sources and old algorithms are immutable. New adapters only; no change to historical score/grid/response/capacity or workload. No 104/160-job runs.

A: one frozen20-job cohort, Neutral, proportional, 2x2 ordered admission x continuous service + external actual Sia. Shared Sia fair score and original cold-start utility (not new completion-score bootstrap), shared eligibility for every previously started job whether held or requeued, lifetime delay budget and actual locks. Quantized300 productive-second expiry has non-destructive resume-budget safety. No individual starvation guarantee for optional admission.
B: existing COMPACT and SPREAD12-job cohorts, Medium, 2x2 serial GPU choice x proportional/optimized auxiliary quotas + external Sia. Shared reciprocal ETA score, FIFO/continuity and eligibility. Serial proposal is a common-controller proportional-column MIP, NOT original Sia. It freezes selected jobs and GPU config; placement remains optimizable in auxiliary stage. Both passes use same utility, resource budgets and locks. A change in placement remains part of optimization, not isolated separately.
C: offline full app/config/quota enumeration for A/B, CPU/RAM numerator and denominator bounds separately; sensitivity classifications and non-flat donor opportunities. E-legacy remains unchanged. No new boundary response enters execution.

Before formal runs: red/green contracts, complete old tests, bounded original Sia regression, offline precheck and preregistration hashes. All formal cells60s/6000rounds, separate processes,2GiB memory/5GiB disk gate. No result-driven tuning. Censoring preserved, JCT means only complete pairs. Failed attempts preserved. Resume entry maintenance/run_clean_ab_20261030.py (to be installed/frozen after tests).

Status: FROZEN15_RUNNING. Preregistration CC26F23F89DB6D134999E7B5D5AAA428CC513A9BB62D3A96E59C3ECD8FAAD8C1. Formal source=frozen_source/. Only monitor existing batch; do not start duplicates. Runner maintenance/run_clean_ab_20261030.py batch validates completed prefix hashes before resume.

Successor preparation: run1 interrupted during A00 for pre-B numeric-control fix, not performance. New working tree cc-joint-clean-ab-r2. Do not resume run1.

RUN2_FROZEN15_RUNNING. Source=frozen_source/, runner=maintenance/run_clean_ab_r2_20261030.py. Do not resume run1 or start duplicate batch.

STOPPED_SHARED_ELIGIBILITY_REVIEW. Do not resume; successor cc_clean_ab_20261030_run3.

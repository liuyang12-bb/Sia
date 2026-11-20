# Current task: completion-oriented joint scheduling

Status: COMPLETE_ENGINEERING_VALID_PERFORMANCE_GATE_NOT_PASSED.
Completed code/math review, two design alternatives, one opt-in minimal implementation,
red/green and legacy tests, offline coverage/opportunity audit, and controlled 2-job training.
No full160/104-cell resumption, parameter search, or large cohort runs.

Baseline actual source: joint_exchange_realization_20261026_run1/source (119 Python files,
including opt-in grid8), checked against sia-joint-service-horizon actual 118 Python files.
Baseline bare HEAD 2d5f9dbfcb7e7bfc2bfd9dc194b1838674a227c2 is NOT the full actual source.
New isolated source: D:/科研启航/Sia/sia-joint-completion-time/sia-simulator.
All old source trees, source snapshots, reports and results are protected.

Recovery: read REPORT.md, DESIGN.md, DATA_GATE.md, freeze.json and delivery_manifest.json.
No full-completion performance cells were started: the data/prediction gate did not pass.
Do not create or resume them automatically. Historical full160/104 runs remain paused.

Result: 188/188 tests; exact bounded official Sia regression. Fixed-GPU CPU exchange is
supported and receiver training improved, but model-rate errors reverse the tiny predicted
net frozen-rate ETC improvement. Two preselected actual online warm states show zero extra
GPU-choice gain, and CIFAR DGX2 has a reference-denominator coverage gap. New JCT is N/A.
Stop here. No parameter, capacity, mapping, clipping or observation-window adjustment.
Source diff: only completion_time.py and test_completion_time.py added; legacy files intact.
Final archive includes full actual source (not bare HEAD), inputs, scripts and dependency lock.

SYNTHETIC CROSS-HARDWARE TRANSFER — NOT REAL SATURN PERFORMANCE.

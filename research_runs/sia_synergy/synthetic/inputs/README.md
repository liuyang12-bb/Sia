# Synthetic experiment inputs

`saturn_workload_20.csv` is the first 20 arrivals of the unmodified official
`sia-simulator/workloads/saturn/workload-1.csv`.  `saturn_workload_3.csv` is its
first three arrivals and is used only for manual behavior checks.

Only names, arrival times, applications, requested GPU replicas, and batch sizes
come from Saturn.  Synthetic sensitivity is assigned from the terminal job index
modulo three and is not a measured property of these applications.

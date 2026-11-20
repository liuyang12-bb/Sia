# Process evidence scope

Each exec tool session can expose a restricted process inventory. A separate
exec invocation returned NoSuchProcess for PID19476 while the active supervisor
session61215 was still waiting for that registered FULL worker. Consequently,
`prior_processes_stopped.json` must not be interpreted as a global host-process
scan. It records only what that separate invocation could observe.

Run1/session86954 and run2/session14630 were explicitly interrupted through
their original unified-exec sessions; both returned completed exit1. Their
execution/console/raw files then stopped growing. Run3 is the only active
supervisor session. Do not restart a batch based solely on an externally
invisible PID; use the original session exit plus artifact activity and hashes.

The watchdog uses its own subprocess handles and psutil from inside the same
parent process/session; this separate-inventory caveat is not a reason to
disable its resource or wall-time safeguards.

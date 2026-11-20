param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$SimulatorDirectory,
    [Parameter(Mandatory = $true)][string]$Workload,
    [Parameter(Mandatory = $true)][ValidateSet("sia", "sia_las")][string]$Policy,
    [Parameter(Mandatory = $true)][string]$Output,
    [Parameter(Mandatory = $true)][string]$StdoutLog,
    [Parameter(Mandatory = $true)][string]$Metadata,
    [double]$LasThresholdGpuSeconds = 3600,
    [double]$LasPriorityFactor = 1.0
)

$ErrorActionPreference = "Stop"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

$commandArgs = @(
    "multi_simulator.py",
    $Workload,
    "--policy=$Policy",
    "--policy_p_val=-0.5",
    "--mip_lambda_a=0.01",
    "--mip_lambda_n=1.1",
    "--interval=60",
    "--project_throughputs",
    "--share_max_replicas",
    "--output=$Output"
)
if ($Policy -eq "sia_las") {
    $commandArgs += "--las_threshold_gpu_seconds=$LasThresholdGpuSeconds"
    $commandArgs += "--las_priority_factor=$LasPriorityFactor"
}

$started = Get-Date
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
Push-Location $SimulatorDirectory
try {
    # Windows PowerShell surfaces native stderr (including harmless Python
    # warnings) as ErrorRecord objects. Keep capturing it without terminating
    # the simulator; the native process exit code remains authoritative.
    $ErrorActionPreference = "Continue"
    & $Python @commandArgs *> $StdoutLog
    $exitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = "Stop"
    Pop-Location
    $stopwatch.Stop()
}

$record = [ordered]@{
    command = @($Python) + $commandArgs
    working_directory = (Resolve-Path $SimulatorDirectory).Path
    start_time = $started.ToString("o")
    end_time = (Get-Date).ToString("o")
    elapsed_seconds = $stopwatch.Elapsed.TotalSeconds
    exit_code = $exitCode
    stdout_stderr_log = $StdoutLog
    output = $Output
}
$record | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $Metadata
exit $exitCode

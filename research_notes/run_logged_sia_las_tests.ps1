param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$Workspace,
    [Parameter(Mandatory = $true)][string]$StdoutLog,
    [Parameter(Mandatory = $true)][string]$Metadata
)

$ErrorActionPreference = "Stop"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

$commandArgs = @(
    "-m", "unittest", "discover",
    "-s", "sia-simulator/tests",
    "-p", "test_sia_las.py",
    "-v"
)
$started = Get-Date
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
Push-Location $Workspace
try {
    $ErrorActionPreference = "Continue"
    & $Python @commandArgs *> $StdoutLog
    $exitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = "Stop"
    Pop-Location
    $stopwatch.Stop()
}

[ordered]@{
    command = @($Python) + $commandArgs
    working_directory = (Resolve-Path $Workspace).Path
    start_time = $started.ToString("o")
    end_time = (Get-Date).ToString("o")
    elapsed_seconds = $stopwatch.Elapsed.TotalSeconds
    exit_code = $exitCode
    stdout_stderr_log = $StdoutLog
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $Metadata
exit $exitCode

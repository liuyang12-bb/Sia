param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $taskRoot '..\..')).Path
$cleanupPlan = Get-Content -LiteralPath (Join-Path $taskRoot 'relocation_plan.json') -Raw -Encoding utf8 | ConvertFrom-Json
$validation = Get-Content -LiteralPath (Join-Path $taskRoot 'relocation_validation.json') -Raw -Encoding utf8 | ConvertFrom-Json
if (!$validation.valid) { throw 'Archive relocation was not validated' }
$archiveRoot = (Resolve-Path -LiteralPath $cleanupPlan.archive).Path
if ([IO.Path]::GetFullPath($cleanupPlan.root) -cne $projectRoot) { throw 'Root mismatch' }
if (!$archiveRoot.StartsWith($projectRoot + '\_archive\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Archive boundary' }
foreach ($treeName in $cleanupPlan.names) {
    if ($treeName -notmatch '^sia-[a-z0-9-]+$') { throw 'Invalid target' }
    $from = (Resolve-Path -LiteralPath (Join-Path $archiveRoot $treeName)).Path
    $to = [IO.Path]::GetFullPath((Join-Path $projectRoot $treeName))
    if ([IO.Path]::GetDirectoryName($from) -cne $archiveRoot -or [IO.Path]::GetDirectoryName($to) -cne $projectRoot) { throw 'Target escaped boundary' }
    if (Test-Path -LiteralPath $to) { throw "Refusing to overwrite: $to" }
    Write-Output "RESTORE PLAN: $from -> $to"
}
if (!$Apply) { Write-Output 'DRY_RUN_ONLY'; exit 0 }
# Restore the main repo before repairing the archived linked worktrees.
# Force handles Windows Hidden metadata only; every destination was verified absent.
$mainFrom = Join-Path $archiveRoot $cleanupPlan.main_repo
$mainTo = Join-Path $projectRoot $cleanupPlan.main_repo
Move-Item -LiteralPath $mainFrom -Destination $mainTo -Force
& git -C $mainTo worktree repair
if ($LASTEXITCODE -ne 0) { throw 'Main repair failed; preserve current layout' }
foreach ($treeName in $cleanupPlan.linked_joint) {
    & git -C $mainTo worktree move (Join-Path $archiveRoot $treeName) (Join-Path $projectRoot $treeName)
    if ($LASTEXITCODE -ne 0) { throw "Joint restore failed: $treeName" }
}
foreach ($treeName in $cleanupPlan.linked_official) {
    & git -C (Join-Path $projectRoot 'official-artifacts') worktree move (Join-Path $archiveRoot $treeName) (Join-Path $projectRoot $treeName)
    if ($LASTEXITCODE -ne 0) { throw "Official-linked restore failed: $treeName" }
}
foreach ($treeName in $cleanupPlan.snapshots) {
    Move-Item -LiteralPath (Join-Path $archiveRoot $treeName) -Destination (Join-Path $projectRoot $treeName) -Force
}
Write-Output 'RESTORE_COMPLETE; current cc-joint/ was not changed'

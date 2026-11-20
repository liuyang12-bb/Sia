param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $taskRoot '..\..')).Path
$cleanupPlan = Get-Content -LiteralPath (Join-Path $taskRoot 'relocation_plan.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$backup = Get-Content -LiteralPath (Join-Path $taskRoot 'backup_manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$archiveRoot = [IO.Path]::GetFullPath($cleanupPlan.archive)
if ($projectRoot -cne [IO.Path]::GetFullPath($cleanupPlan.root)) { throw 'Project root mismatch' }
if (!$archiveRoot.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Archive outside project' }
if ((Get-FileHash -LiteralPath (Join-Path $taskRoot $backup.zip) -Algorithm SHA256).Hash -ne $backup.zip_sha256) { throw 'Backup hash mismatch' }
if ((Get-FileHash -LiteralPath (Join-Path $taskRoot 'relocation_plan.json') -Algorithm SHA256).Hash -ne $backup.plan_sha256) { throw 'Plan hash mismatch' }
if (!$backup.zip_readback_verified) { throw 'Backup was not verified' }
foreach ($treeName in $cleanupPlan.names) {
    if ($treeName -notmatch '^sia-[a-z0-9-]+$') { throw 'Invalid explicit target' }
    $oldTarget = (Resolve-Path -LiteralPath (Join-Path $projectRoot $treeName)).Path
    $newTarget = [IO.Path]::GetFullPath((Join-Path $archiveRoot $treeName))
    if ([IO.Path]::GetDirectoryName($oldTarget) -cne $projectRoot) { throw 'Old target not direct project child' }
    if ([IO.Path]::GetDirectoryName($newTarget) -cne $archiveRoot) { throw 'New target escaped archive' }
    if ((Get-Item -LiteralPath $oldTarget -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Reparse target refused' }
    if (Test-Path -LiteralPath $newTarget) { throw 'Refusing destination collision' }
    Write-Output "CHECKED: $oldTarget -> $newTarget"
}
if (!$Apply) { Write-Output 'DRY_RUN_ONLY: no directory moved'; exit 0 }
if (Test-Path -LiteralPath $archiveRoot) { throw 'Archive already exists; do not repeat this operation' }
New-Item -ItemType Directory -Path $archiveRoot | Out-Null
$journal = Join-Path $taskRoot 'relocation_journal.jsonl'
if (Test-Path -LiteralPath $journal) { throw 'Journal already exists' }
function Record-Relocation($kind, $name, $result) {
    [ordered]@{utc=(Get-Date).ToUniversalTime().ToString('o');kind=$kind;name=$name;result=$result} |
        ConvertTo-Json -Compress | Add-Content -LiteralPath $journal -Encoding utf8
}
function Move-Registered($repoName, $treeName) {
    $oldTarget = Join-Path $projectRoot $treeName
    $newTarget = Join-Path $archiveRoot $treeName
    & git -C (Join-Path $projectRoot $repoName) worktree move $oldTarget $newTarget
    if ($LASTEXITCODE -ne 0) { Record-Relocation 'git_worktree_move' $treeName 'FAILED'; throw "Git move failed: $treeName" }
    Record-Relocation 'git_worktree_move' $treeName 'SUCCESS'
}
foreach ($treeName in $cleanupPlan.linked_joint) { Move-Registered 'sia-joint-experiments' $treeName }
foreach ($treeName in $cleanupPlan.linked_official) { Move-Registered 'official-artifacts' $treeName }
foreach ($treeName in $cleanupPlan.snapshots) {
    Move-Item -LiteralPath (Join-Path $projectRoot $treeName) -Destination (Join-Path $archiveRoot $treeName)
    Record-Relocation 'snapshot_move' $treeName 'SUCCESS'
}
$mainOld = (Resolve-Path -LiteralPath (Join-Path $projectRoot $cleanupPlan.main_repo)).Path
$mainNew = [IO.Path]::GetFullPath((Join-Path $archiveRoot $cleanupPlan.main_repo))
if ([IO.Path]::GetDirectoryName($mainOld) -cne $projectRoot -or [IO.Path]::GetDirectoryName($mainNew) -cne $archiveRoot) { throw 'Main target boundary failed' }
Move-Item -LiteralPath $mainOld -Destination $mainNew
Record-Relocation 'main_repo_move' $cleanupPlan.main_repo 'SUCCESS'
& git -C $mainNew worktree repair
if ($LASTEXITCODE -ne 0) { Record-Relocation 'worktree_repair' $cleanupPlan.main_repo 'FAILED'; throw 'Repair failed; preserve all directories' }
Record-Relocation 'worktree_repair' $cleanupPlan.main_repo 'SUCCESS'
Write-Output 'RELOCATION_COMPLETE_NO_DELETION'

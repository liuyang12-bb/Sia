$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $taskRoot '..\..')).Path
$oldRoot = (Resolve-Path -LiteralPath (Join-Path $projectRoot 'sia-joint-experiments')).Path
$newRoot = (Resolve-Path -LiteralPath (Join-Path $projectRoot '_archive\legacy_code_20261027\sia-joint-experiments')).Path
if ([IO.Path]::GetDirectoryName($oldRoot) -cne $projectRoot) { throw 'Old root boundary' }
if (!$newRoot.StartsWith($projectRoot + '\_archive\legacy_code_20261027\', [StringComparison]::OrdinalIgnoreCase)) { throw 'New root boundary' }
& git -C $newRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Destination Git metadata not intact' }
$remaining = @(Get-ChildItem -LiteralPath $oldRoot -Force)
if (@($remaining | Where-Object {$_.Name -notin @('.git','adaptdl','sia-simulator')}).Count -ne 0) { throw 'Unexpected source remainder' }
$oldGit = Join-Path $oldRoot '.git'
if (@(Get-ChildItem -LiteralPath $oldGit -Force).Count -ne 0) { throw 'Old .git is not empty; stop' }
foreach ($component in @('adaptdl','sia-simulator')) {
    $from = (Resolve-Path -LiteralPath (Join-Path $oldRoot $component)).Path
    $to = [IO.Path]::GetFullPath((Join-Path $newRoot $component))
    if ([IO.Path]::GetDirectoryName($from) -cne $oldRoot -or [IO.Path]::GetDirectoryName($to) -cne $newRoot) { throw 'Child boundary' }
    if (Test-Path -LiteralPath $to) { throw 'Refusing destination collision' }
    Move-Item -LiteralPath $from -Destination $to
}
# Only verified empty leftover containers are removed. No recursive removal.
if (@(Get-ChildItem -LiteralPath $oldGit -Force).Count -ne 0) { throw 'Old .git gained content' }
Remove-Item -LiteralPath $oldGit -Force
if (@(Get-ChildItem -LiteralPath $oldRoot -Force).Count -ne 0) { throw 'Old root not empty' }
Remove-Item -LiteralPath $oldRoot
& git -C $newRoot worktree repair
if ($LASTEXITCODE -ne 0) { throw 'Worktree repair failed' }
[ordered]@{utc=(Get-Date).ToUniversalTime().ToString('o');kind='finish_main_move_and_repair';name='sia-joint-experiments';result='SUCCESS';empty_containers_removed=@($oldGit,$oldRoot)} |
    ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $taskRoot 'relocation_journal.jsonl') -Encoding utf8
Write-Output 'MAIN_RELOCATION_AND_GIT_REPAIR_COMPLETE'

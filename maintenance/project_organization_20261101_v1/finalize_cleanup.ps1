$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath('D:\科研启航\Sia').TrimEnd('\')
$taskOut = Join-Path $taskRoot 'maintenance\project_organization_20261101_v1'
if (Test-Path -LiteralPath (Join-Path $taskOut 'cleanup_receipt.json')) { throw 'Cleanup already performed.' }
$taskPlan = Get-Content -LiteralPath (Join-Path $taskOut 'archive_manifest.json') -Raw | ConvertFrom-Json
if ($taskPlan.status -ne 'VERIFIED_BEFORE_REMOVAL' -or !$taskPlan.all_files_verified) { throw 'Archive not verified.' }
$taskZip = [IO.Path]::GetFullPath($taskPlan.archive)
if ($taskZip -ne (Join-Path $taskRoot '_archive\cc_intermediate_sources_20261101_v1.zip')) { throw 'Unexpected archive.' }
if ((Get-FileHash -LiteralPath $taskZip -Algorithm SHA256).Hash -ne $taskPlan.archive_sha256) { throw 'Archive hash mismatch.' }
# Validate ALL targets and hashes before any source-directory removal.
$taskTargets = @()
foreach ($taskName in $taskPlan.trees) {
    if ($taskName -notlike 'cc-joint-*' -or $taskName -match '[\\/]') { throw 'Invalid tree name.' }
    if ($taskName -in $taskPlan.preserved_trees) { throw 'Protected tree.' }
    $taskTarget = Join-Path $taskRoot $taskName
    $taskResolved = (Resolve-Path -LiteralPath $taskTarget).Path.TrimEnd('\')
    if ($taskResolved -ne $taskTarget -or (Split-Path -Parent $taskResolved) -ne $taskRoot) { throw 'Target outside workspace.' }
    $taskItem = Get-Item -LiteralPath $taskResolved -Force
    if ($taskItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Target is a link.' }
    if (Test-Path -LiteralPath (Join-Path $taskResolved '.git')) { throw 'Target is a Git tree.' }
    $taskExpected = @($taskPlan.files | Where-Object { $_.path.StartsWith($taskName + '/') })
    $taskActual = @(Get-ChildItem -LiteralPath $taskResolved -File -Recurse -Force)
    if ($taskActual.Count -ne $taskExpected.Count) { throw 'Source file count changed.' }
    foreach ($taskFile in $taskExpected) {
        $taskFilePath = Join-Path $taskRoot $taskFile.path
        if ((Get-FileHash -LiteralPath $taskFilePath -Algorithm SHA256).Hash -ne $taskFile.sha256) { throw "Changed: $taskFilePath" }
    }
    $taskTargets += $taskResolved
}
foreach ($taskTarget in $taskTargets) {
    Remove-Item -LiteralPath $taskTarget -Recurse -Force
    @{action='remove_verified_archived_tree';path=$taskTarget;utc=(Get-Date).ToUniversalTime().ToString('o')} |
        ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $taskOut 'cleanup_journal.jsonl') -Encoding utf8
}
# Remove only bytecode whose corresponding Python source still exists in an isolated environment.
$taskCacheFiles = @()
foreach ($taskEnv in @('.venv-sia-baseline','.venv-synergy')) {
    $taskEnvRoot = (Resolve-Path -LiteralPath (Join-Path $taskRoot $taskEnv)).Path.TrimEnd('\')
    if ($taskEnvRoot -ne (Join-Path $taskRoot $taskEnv)) { throw 'Unexpected environment path.' }
    foreach ($taskCache in @(Get-ChildItem -LiteralPath $taskEnvRoot -Directory -Filter '__pycache__' -Recurse -Force)) {
        if (!$taskCache.FullName.StartsWith($taskEnvRoot + '\')) { throw 'Cache outside environment.' }
        if ($taskCache.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Linked cache.' }
        foreach ($taskPyc in @(Get-ChildItem -LiteralPath $taskCache.FullName -File -Filter '*.pyc')) {
            $taskModule = $taskPyc.Name -replace '\.(cpython|pypy)-.*\.pyc$',''
            if ($taskModule -eq $taskPyc.Name) { continue }
            $taskPy = Join-Path $taskCache.Parent.FullName ($taskModule + '.py')
            if (Test-Path -LiteralPath $taskPy -PathType Leaf) {
                $taskCacheFiles += @{path=$taskPyc.FullName;bytes=$taskPyc.Length;source=$taskPy}
                Remove-Item -LiteralPath $taskPyc.FullName -Force
            }
        }
        if (@(Get-ChildItem -LiteralPath $taskCache.FullName -Force).Count -eq 0) { Remove-Item -LiteralPath $taskCache.FullName }
    }
}
$taskCacheFiles | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskOut 'removed_regenerable_bytecode.json') -Encoding utf8
$taskCacheBytes = ($taskCacheFiles | Measure-Object -Property bytes -Sum).Sum
@{status='COMPLETED';archived_trees=$taskTargets.Count;archived_source_bytes=$taskPlan.source_bytes;
  archive_bytes=$taskPlan.archive_bytes;removed_cache_files=$taskCacheFiles.Count;
  removed_cache_bytes=$taskCacheBytes;net_content_bytes_freed=($taskPlan.source_bytes-$taskPlan.archive_bytes+$taskCacheBytes);
  archive_sha256=$taskPlan.archive_sha256;utc=(Get-Date).ToUniversalTime().ToString('o')} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskOut 'cleanup_receipt.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $taskOut 'cleanup_receipt.json')

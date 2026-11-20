param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath('D:\科研启航\Sia').TrimEnd('\')
$taskOut = Join-Path $taskRoot 'maintenance\project_organization_20261101_v1'
$taskPlan = Get-Content -LiteralPath (Join-Path $taskOut 'archive_manifest.json') -Raw | ConvertFrom-Json
$taskZip = Join-Path $taskRoot '_archive\cc_intermediate_sources_20261101_v1.zip'
if ((Get-FileHash -LiteralPath $taskZip -Algorithm SHA256).Hash -ne $taskPlan.archive_sha256) { throw 'Archive hash mismatch.' }
foreach ($taskName in $taskPlan.trees) {
    if ($taskName -match '[\\/]' -or $taskName -notlike 'cc-joint-*') { throw 'Invalid archived name.' }
    if (Test-Path -LiteralPath (Join-Path $taskRoot $taskName)) { throw "Will not overwrite existing tree: $taskName" }
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$taskRead = [IO.Compression.ZipFile]::OpenRead($taskZip)
try {
    foreach ($taskEntry in $taskRead.Entries) {
        if ($taskEntry.FullName -eq '_ARCHIVE_SOURCE_MANIFEST.json') { continue }
        $taskDest = [IO.Path]::GetFullPath((Join-Path $taskRoot $taskEntry.FullName))
        if (!$taskDest.StartsWith($taskRoot + '\')) { throw 'Zip path outside workspace.' }
        $taskTop = $taskEntry.FullName.Split('/')[0]
        if ($taskTop -notin $taskPlan.trees) { throw 'Unexpected top-level ZIP entry.' }
    }
    if (!$Execute) {
        Write-Output "DRY RUN: verified archive; would restore $($taskPlan.trees.Count) trees, $($taskPlan.files.Count) files. Use -Execute to restore."
        return
    }
    foreach ($taskEntry in $taskRead.Entries) {
        if ($taskEntry.FullName -eq '_ARCHIVE_SOURCE_MANIFEST.json') { continue }
        $taskDest = [IO.Path]::GetFullPath((Join-Path $taskRoot $taskEntry.FullName))
        if ($taskEntry.FullName.EndsWith('/')) { [IO.Directory]::CreateDirectory($taskDest) | Out-Null; continue }
        [IO.Directory]::CreateDirectory((Split-Path -Parent $taskDest)) | Out-Null
        [IO.Compression.ZipFileExtensions]::ExtractToFile($taskEntry,$taskDest,$false)
    }
} finally { $taskRead.Dispose() }
foreach ($taskFile in $taskPlan.files) {
    $taskDest = Join-Path $taskRoot $taskFile.path
    if ((Get-FileHash -LiteralPath $taskDest -Algorithm SHA256).Hash -ne $taskFile.sha256) { throw "Restored file hash mismatch: $taskDest" }
}
Write-Output 'RESTORED: every archived file matches its original SHA-256.'

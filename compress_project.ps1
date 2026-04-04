
$dirName = "ModelStream"
$sourceDir = ".\$dirName"
$excludeDirs = @("docs", "build", "bin", "lib", ".vscode")
$timestamp = Get-Date -Format "yyyy-MM-dd-HH-mm"
$outputFile = "$sourceDir\$timestamp.zip"

$tempDir = "$env:TEMP\ZipExclude_$PID"
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

Get-ChildItem -Path $sourceDir -Directory | ForEach-Object {
    $dirName = $_.Name
    if ($excludeDirs -notcontains $dirName) {
        Copy-Item -Path $_.FullName -Destination "$tempDir\$dirName" -Recurse -Force
    }
}

Get-ChildItem -Path $sourceDir -File | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination $tempDir -Force
}

if (Test-Path $outputFile) {
    Remove-Item $outputFile -Force
}

Compress-Archive -Path "$tempDir\*" -DestinationPath $outputFile -CompressionLevel Optimal

Remove-Item -Path $tempDir -Recurse -Force

Write-Host "压缩完成: $outputFile" -ForegroundColor Green

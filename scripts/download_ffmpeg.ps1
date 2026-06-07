# FFmpeg Download and Setup Script for Windows
$ProjectDir = Resolve-Path "$PSScriptRoot\.."
$TempZip = Join-Path $ProjectDir "ffmpeg.zip"
$ExtractDir = Join-Path $ProjectDir "ffmpeg_temp"

# Gyan.dev Essentials Release URL
$Url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

Write-Host "Downloading FFmpeg from $Url..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri $Url -OutFile $TempZip -ErrorAction Stop
    Write-Host "Extracting FFmpeg..." -ForegroundColor Cyan
    if (Test-Path $ExtractDir) { Remove-Item -Recurse -Force $ExtractDir }
    Expand-Archive -Path $TempZip -DestinationPath $ExtractDir -Force

    # Find ffmpeg.exe inside the extracted directory
    $FfmpegExe = Get-ChildItem -Path $ExtractDir -Recurse -File -Filter "ffmpeg.exe" | Select-Object -First 1

    if ($FfmpegExe) {
        # Copy to project root
        Copy-Item $FfmpegExe.FullName -Destination $ProjectDir -Force
        Write-Host "Installed FFmpeg to Project Root: $ProjectDir" -ForegroundColor Green

        # Copy to build Release directory if it exists
        $ReleaseDir = Join-Path $ProjectDir "build\Release"
        if (Test-Path $ReleaseDir) {
            Copy-Item $FfmpegExe.FullName -Destination $ReleaseDir -Force
            Write-Host "Installed FFmpeg to Build Release: $ReleaseDir" -ForegroundColor Green
        }
    } else {
        Write-Host "Error: ffmpeg.exe was not found in the downloaded package." -ForegroundColor Red
    }
} catch {
    Write-Host "Failed to download/install FFmpeg." -ForegroundColor Red
    Write-Host $_.Exception.Message
} finally {
    if (Test-Path $TempZip) { Remove-Item $TempZip -Force }
    if (Test-Path $ExtractDir) { Remove-Item -Recurse -Force $ExtractDir }
}

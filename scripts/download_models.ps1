# Real-ESRGAN & RIFE Model Download Script (PowerShell Native)
$ProjectDir = $PSScriptRoot
$ModelDir = Join-Path $ProjectDir "models"
$TempZip = Join-Path $ProjectDir "temp.zip"
$ExtractDir = Join-Path $ProjectDir "temp_extract"

if (!(Test-Path $ModelDir)) { New-Item -ItemType Directory -Path $ModelDir }

function Download-And-Extract {
    param($Url, $Name)
    Write-Host "Downloading $Name..." -ForegroundColor Yellow
    try {
        # Using -UseBasicParsing to avoid IE dependency if needed
        Invoke-WebRequest -Uri $Url -OutFile $TempZip -ErrorAction Stop
        Write-Host "Extracting $Name..." -ForegroundColor Cyan
        if (Test-Path $ExtractDir) { Remove-Item -Recurse -Force $ExtractDir }
        Expand-Archive -Path $TempZip -DestinationPath $ExtractDir -Force
        
        # モデルファイルをコピー (再帰検索)
        # Real-ESRGAN v0.2.5.0 の ZIP では 'models' フォルダ内に格納されている
        Get-ChildItem -Path $ExtractDir -Recurse -File -Include "*.bin","*.param" | ForEach-Object {
            Copy-Item $_.FullName -Destination $ModelDir -Force
            Write-Host "  Installed: $($_.Name)"
        }
    } catch {
        Write-Host "Failed to process $Name" -ForegroundColor Red
        Write-Host $_.Exception.Message
    } finally {
        if (Test-Path $TempZip) { Remove-Item $TempZip -Force }
    }
}

# 1. Real-ESRGAN (v0.2.5.0 portable package contains models)
Download-And-Extract "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip" "Real-ESRGAN"

# 2. RIFE (20221029 windows package contains models)
Download-And-Extract "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-windows.zip" "RIFE"

if (Test-Path $ExtractDir) { Remove-Item -Recurse -Force $ExtractDir }
Write-Host "`nAll models processed. Check $ModelDir" -ForegroundColor Green

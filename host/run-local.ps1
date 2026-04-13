# Chạy Streamlit LucenFace trên localhost (Windows).
# Chạy từ bất kỳ đâu:  .\host\run-local.ps1  hoặc cd host; .\run-local.ps1

$ErrorActionPreference = "Stop"
$HostDir = $PSScriptRoot
$RepoRoot = Split-Path -Parent $HostDir

Set-Location $RepoRoot
Write-Host "Repo: $RepoRoot"

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$VenvPip = Join-Path $RepoRoot ".venv\Scripts\pip.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Tao venv .venv ..."
    $used312 = $false
    $null = & py -3.12 -c "import sys" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Dung Python 3.12 (py -3.12) - on dinh hon 3.13/3.14 voi OpenCV/rembg."
        & py -3.12 -m venv .venv
        if ($LASTEXITCODE -eq 0) {
            $used312 = $true
        }
    }
    if (-not $used312) {
        Write-Host "Dung python trong PATH (cai Python 3.12 neu rembg/cv2 crash)."
        python -m venv .venv
    }
}

Write-Host "Cap nhat pip va cai requirements.txt (co the vai phut)..."
& $VenvPip install -U pip
& $VenvPip install -r (Join-Path $RepoRoot "requirements.txt")

$Streamlit = Join-Path $RepoRoot ".venv\Scripts\streamlit.exe"
$App = Join-Path $RepoRoot "app.py"

Write-Host ""
Write-Host "Mo trinh duyet: http://localhost:8501"
Write-Host "Dung: Ctrl+C"
Write-Host ""

& $Streamlit run $App `
    --server.address=localhost `
    --server.port=8501 `
    --browser.gatherUsageStats=false

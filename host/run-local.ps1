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
    python -m venv .venv
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

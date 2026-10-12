Set-Location $PSScriptRoot

Write-Host "AI Teaching Agent - Power Electronics" -ForegroundColor Cyan

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment..."
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Python 3.12 was not found. Install Python 3.11/3.12." -ForegroundColor Red
        exit 1
    }
}

& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host ""
Write-Host "Open http://127.0.0.1:8000" -ForegroundColor Green
& ".venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

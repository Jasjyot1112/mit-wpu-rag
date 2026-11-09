<#
.SYNOPSIS
    Automated Windows Setup and Dependency Installer for MIT-WPU RAG Engine.
.DESCRIPTION
    Checks Python 3.12, sets up virtual environment, installs requirements,
    and checks Tesseract OCR.
#>

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  MIT-WPU RAG Engine — Windows Environment Setup" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[!] Python was not found in PATH." -ForegroundColor Red
    Write-Host "    Please install Python 3.12 from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "    Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
    exit 1
}

$pyVer = python --version
Write-Host "[✓] Found Python: $pyVer" -ForegroundColor Green

# 2. Check / Create Virtual Environment
$venvPath = ".\venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "[*] Creating virtual environment at $venvPath..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "[✓] Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "[✓] Virtual environment already exists at $venvPath." -ForegroundColor Green
}

# 3. Activate and Install Requirements
Write-Host "[*] Upgrading pip and installing dependencies from requirements.txt..." -ForegroundColor Yellow
& ".\venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\venv\Scripts\python.exe" -m pip install -r requirements.txt
Write-Host "[✓] Dependencies installed." -ForegroundColor Green

# 4. Check Tesseract OCR
$tesseractPath = "C:\Program Files\Tesseract-OCR\tesseract.exe"
if (Test-Path $tesseractPath) {
    Write-Host "[✓] Tesseract OCR found at $tesseractPath" -ForegroundColor Green
} else {
    Write-Host "[!] Tesseract OCR not found at default location: $tesseractPath" -ForegroundColor Yellow
    Write-Host "    Download the Windows installer from:" -ForegroundColor Yellow
    Write-Host "    https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Yellow
    Write-Host "    Ensure English language pack is selected during install." -ForegroundColor Yellow
}

# 5. Environment File
if (-not (Test-Path ".\.env")) {
    Write-Host "[*] Copying .env.example to .env..." -ForegroundColor Yellow
    Copy-Item ".\.env.example" ".\.env"
    Write-Host "[!] Please edit .\.env and configure your GEMINI_API_KEY, PINECONE_API_KEY, and RAG_API_KEY." -ForegroundColor Cyan
} else {
    Write-Host "[✓] .env file exists." -ForegroundColor Green
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete! To start the RAG Engine:" -ForegroundColor Green
Write-Host "    .\venv\Scripts\activate" -ForegroundColor White
Write-Host "    uvicorn app.main:app --reload --port 8000" -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor Cyan

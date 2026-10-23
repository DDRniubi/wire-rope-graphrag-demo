param([string]$PythonExe = '', [switch]$WithGraphIndexing)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $root '.venv'
if (-not (Test-Path -LiteralPath (Join-Path $venv 'Scripts\python.exe'))) {
    if ($PythonExe) { & $PythonExe -m venv $venv }
    else { & py -3.12 -m venv $venv }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 environment creation failed.' }
}
$python = Join-Path $venv 'Scripts\python.exe'
& $python -m pip install -r (Join-Path $root 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Runtime dependency installation failed.' }
if ($WithGraphIndexing) {
    $indexVenv = Join-Path $root '.index-venv'
    if (-not (Test-Path -LiteralPath (Join-Path $indexVenv 'Scripts\python.exe'))) {
        if ($PythonExe) { & $PythonExe -m venv $indexVenv }
        else { & py -3.12 -m venv $indexVenv }
        if ($LASTEXITCODE -ne 0) { throw 'Index environment creation failed.' }
    }
    & (Join-Path $indexVenv 'Scripts\python.exe') -m pip install -r (Join-Path $root 'requirements-indexing.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Graph indexing dependency installation failed.' }
}
Write-Host 'Setup complete. Run run_full_pipeline.ps1 next.'

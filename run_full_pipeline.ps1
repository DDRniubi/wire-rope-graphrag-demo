param(
    [string]$InputPath = 'data\sample_images',
    [string]$OutputDir = '',
    [string]$PythonExe = '',
    [string]$Device = 'cpu',
    [ValidateSet('site', 'dbtt')][string]$ModelSet = 'site',
    [double]$Confidence = 0.25,
    [int]$ImageSize = 1280,
    [switch]$UseQwen
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
if (-not $PythonExe) { $PythonExe = Join-Path $root '.venv\Scripts\python.exe' }
elseif (-not [IO.Path]::IsPathRooted($PythonExe) -and ($PythonExe.Contains('\') -or $PythonExe.Contains('/'))) {
    $PythonExe = Join-Path $root $PythonExe
}
if ($UseQwen -and [string]::IsNullOrWhiteSpace($env:DASHSCOPE_API_KEY)) {
    $env:DASHSCOPE_API_KEY = [Environment]::GetEnvironmentVariable('DASHSCOPE_API_KEY', 'User')
}
$arguments = @((Join-Path $root 'scripts\run_pipeline.py'), '--input', $InputPath,
    '--device', $Device, '--model-set', $ModelSet, '--confidence', $Confidence, '--image-size', $ImageSize)
if ($OutputDir) { $arguments += @('--output', $OutputDir) }
if ($UseQwen) { $arguments += '--use-qwen' }
& $PythonExe @arguments
if ($LASTEXITCODE -ne 0) { throw "Pipeline failed with exit code $LASTEXITCODE" }

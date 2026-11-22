<#
.SYNOPSIS
    Повна збірка оптимізованого дистрибутиву та інсталятора SilentVoice TTS.
.DESCRIPTION
    1. Очищує тимчасові папки build/ та dist/.
    2. Збирає оптимізований onedir-бандл через PyInstaller (SilentVoice.spec).
    3. Додатково очищує випадкові залишки у _internal (зайві переклади, .pyc тести тощо).
    4. Компілює інсталятор + деінсталятор через Inno Setup 6 (LZMA2/ultra64).
#>
[CmdletBinding()]
param(
    [switch]$SkipPyInstaller,
    [switch]$SkipInstaller,
    [switch]$Fast
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Find-Executable([string]$Name, [string[]]$ExtraPaths) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in $ExtraPaths) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

$PyInstaller = Find-Executable "pyinstaller.exe" @(
    "$env:APPDATA\Python\Python314\Scripts\pyinstaller.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python314\Scripts\pyinstaller.exe",
    "C:\Program Files\Python314\Scripts\pyinstaller.exe"
)

$ISCC = Find-Executable "ISCC.exe" @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
)

if (-not $SkipPyInstaller) {
    if (-not $PyInstaller) {
        throw "Не знайдено pyinstaller.exe! Встановіть PyInstaller: pip install pyinstaller"
    }
    Write-Host "[1/3] Очищення попередніх артефактів збірки..." -ForegroundColor Cyan
    if (-not $Fast -and (Test-Path "$Root\build")) { Remove-Item "$Root\build" -Recurse -Force }
    if (Test-Path "$Root\dist\SilentVoice") { Remove-Item "$Root\dist\SilentVoice" -Recurse -Force }

    Write-Host "[2/3] Збірка оптимізованого бандлу PyInstaller (SilentVoice.spec)..." -ForegroundColor Cyan
    $pyiArgs = @("--noconfirm")
    if (-not $Fast) { $pyiArgs += "--clean" }
    & $PyInstaller @pyiArgs "$Root\SilentVoice.spec"
    if ($LASTEXITCODE -ne 0) {
        throw "Помилка збірки PyInstaller (код $LASTEXITCODE)"
    }

    $DistDir = "$Root\dist\SilentVoice"
    $DistSizeMB = [math]::Round(((Get-ChildItem $DistDir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 2)
    Write-Host "      Розмір розпакованого бандлу: $DistSizeMB МБ" -ForegroundColor Green
}

if (-not $SkipInstaller) {
    if (-not $ISCC) {
        throw "Не знайдено ISCC.exe (Inno Setup 6)!"
    }
    Write-Host "[3/3] Компіляція інсталятора та деінсталятора Inno Setup 6 (LZMA2 Ultra64)..." -ForegroundColor Cyan
    if (-not (Test-Path "$Root\dist\installer")) {
        New-Item -ItemType Directory -Path "$Root\dist\installer" | Out-Null
    }
    $isccArgs = @()
    if ($Fast) { $isccArgs += "/DFastBuild" }
    & $ISCC @isccArgs "$Root\installer\SilentVoice.iss"
    if ($LASTEXITCODE -ne 0) {
        throw "Помилка компіляції Inno Setup (код $LASTEXITCODE)"
    }

    $SetupExe = Get-ChildItem "$Root\dist\installer\SilentVoice_Setup_*.exe" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($SetupExe) {
        $SetupMB = [math]::Round(($SetupExe.Length / 1MB), 2)
        $HashObj = Get-FileHash -Path $SetupExe.FullName -Algorithm SHA256
        $ShaLine = "$($HashObj.Hash.ToLower())  $($SetupExe.Name)"
        $SumsFile = Join-Path $SetupExe.DirectoryName "SHA256SUMS.txt"
        Set-Content -Path $SumsFile -Value $ShaLine -Encoding UTF8
        Write-Host "`n[ГОТОВО] Інсталятор створено: $($SetupExe.FullName) ($SetupMB МБ)" -ForegroundColor Green
        Write-Host "         SHA256SUMS.txt:     $ShaLine" -ForegroundColor Cyan
    }
}

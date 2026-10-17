# Builds whisper.cpp with the Vulkan backend (AMD / Intel GPUs on Windows) and downloads the model.
# Idempotent: every step is skipped when already done. Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File scripts/build-whisper-vulkan.ps1
$ErrorActionPreference = 'Stop'

$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Tag = 'v1.9.5'
$Src = Join-Path $Repo 'tools\src\whisper.cpp'
$BinDir = Join-Path $Repo 'tools\bin'
$ModelDir = Join-Path $Repo 'tools\models'
$ModelName = 'ggml-large-v3-turbo.bin'
$ModelUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$ModelName"
$Exe = Join-Path $BinDir 'whisper-cli.exe'
$Model = Join-Path $ModelDir $ModelName

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" } }
function Get-VulkanSdk {
    $p = [Environment]::GetEnvironmentVariable('VULKAN_SDK', 'Machine')
    if ($p -and (Test-Path $p)) { return $p }
    if (Test-Path 'C:\VulkanSDK') {
        $d = Get-ChildItem 'C:\VulkanSDK' -Directory | Sort-Object Name -Descending | Select-Object -First 1
        if ($d) { return $d.FullName }
    }
    return $null
}
function Find-CMakeBin {
    $c = Get-Command cmake -ErrorAction SilentlyContinue
    if ($c) { return Split-Path $c.Source }
    $p = 'C:\Program Files\CMake\bin'
    if (Test-Path (Join-Path $p 'cmake.exe')) { return $p }
    return $null
}

$alreadyBuilt = (Test-Path $Exe) -and (Test-Path $Model)

if (-not $alreadyBuilt) {
    # 1. Tooling
    if (-not (Find-CMakeBin)) {
        Step 'Installing Kitware.CMake'
        winget install -e --id Kitware.CMake --accept-package-agreements --accept-source-agreements
        Check 'winget install Kitware.CMake'
    }
    $cmakeBin = Find-CMakeBin
    if (-not $cmakeBin) { throw 'cmake not found after install' }
    if (($env:Path -split ';') -notcontains $cmakeBin) { $env:Path = "$cmakeBin;$env:Path" }

    if (-not (Get-VulkanSdk)) {
        Step 'Installing KhronosGroup.VulkanSDK'
        winget install -e --id KhronosGroup.VulkanSDK --accept-package-agreements --accept-source-agreements
        Check 'winget install KhronosGroup.VulkanSDK'
    }
    $sdk = Get-VulkanSdk
    if (-not $sdk) { throw 'Vulkan SDK not found after install (expected C:\VulkanSDK\<version>)' }
    $env:VULKAN_SDK = $sdk
    Write-Host "VULKAN_SDK = $sdk"

    # 2. Source
    if (-not (Test-Path (Join-Path $Src '.git'))) {
        Step "Cloning whisper.cpp $Tag"
        New-Item -ItemType Directory -Force (Split-Path $Src) | Out-Null
        git clone --depth 1 --branch $Tag https://github.com/ggml-org/whisper.cpp $Src
        Check 'git clone'
    }

    # 3. MSVC environment
    if (-not (Test-Path $Exe)) {
        Step 'Loading VS 2022 Build Tools environment'
        $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
        $vsPath = & $vswhere -latest -products '*' -property installationPath
        Check 'vswhere'
        if (-not $vsPath) { throw 'Visual Studio Build Tools not found' }
        $devShell = Join-Path $vsPath 'Common7\Tools\Launch-VsDevShell.ps1'
        $here = Get-Location
        & $devShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
        Set-Location $here

        # 4. Build
        Step 'Configuring (GGML_VULKAN=1)'
        Push-Location $Src
        try {
            cmake -B build -DGGML_VULKAN=1 -DCMAKE_BUILD_TYPE=Release
            Check 'cmake configure'
            Step 'Building (Release)'
            cmake --build build --config Release -j
            Check 'cmake build'
        } finally { Pop-Location }

        # 5. Install binaries
        $out = Join-Path $Src 'build\bin\Release'
        if (-not (Test-Path (Join-Path $out 'whisper-cli.exe'))) { throw "whisper-cli.exe not found in $out" }
        New-Item -ItemType Directory -Force $BinDir | Out-Null
        Copy-Item (Join-Path $out '*') $BinDir -Force
    }

    # 6. Model
    if (-not (Test-Path $Model)) {
        Step "Downloading $ModelName (~1.6 GB)"
        New-Item -ItemType Directory -Force $ModelDir | Out-Null
        $part = "$Model.part"
        if (Test-Path $part) { Remove-Item $part -Force }
        curl.exe -L --fail --silent --show-error --retry 3 -o $part $ModelUrl
        Check 'model download'
        Move-Item $part $Model
    }
}

if (-not ((Test-Path $Exe) -and (Test-Path $Model))) { throw 'Build finished but outputs are missing' }
Step 'Done. Hookline auto-discovers the binary and model under tools/ - no setup needed.'
Write-Host 'Only set these environment variables if you want to override the defaults:'
Write-Host "HOOKLINE_WHISPER_CPP_BIN=$Exe"
Write-Host "HOOKLINE_WHISPER_CPP_MODEL=$Model"

# STEMForge Windows x64 one-click NETWORK installer.
# All Python runtimes, packages and FFmpeg binaries stay in this application folder.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Runtime = Join-Path $Root 'runtime'
$Downloads = Join-Path $Runtime 'downloads'
$Tools = Join-Path $Runtime 'tools'
$Cache = Join-Path $Runtime 'cache'
$Venv = Join-Path $Runtime 'venv'
$FFbin = Join-Path $Runtime 'ffmpeg'
foreach ($p in @($Runtime, $Downloads, $Tools, $Cache, (Join-Path $Root 'data\models'), (Join-Path $Root 'data\jobs'))) {
    New-Item -ItemType Directory -Force -Path $p | Out-Null
}
$Log = Join-Path $Runtime 'install.log'
Start-Transcript -Path $Log -Append | Out-Null
try {
    Write-Host ''
    Write-Host '===================================================' -ForegroundColor Cyan
    Write-Host '        STEMForge 1.0 - Local Setup' -ForegroundColor Cyan
    Write-Host '===================================================' -ForegroundColor Cyan
    Write-Host 'This installer downloads dependencies into this folder.'
    Write-Host 'It does NOT need system Python, Conda or CUDA Toolkit.'
    Write-Host 'Internet and at least 20 GB free disk space are recommended (including caches and models).'
    Write-Host ''
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'Windows x64 (64-bit) is required.' }
    $os = [Environment]::OSVersion.Platform
    if ($os -ne [PlatformID]::Win32NT) { throw 'Only Windows 10/11 x64 is supported by this installer.' }
    function Download-IfMissing([string]$url, [string]$to) {
        if ((Test-Path $to) -and ((Get-Item $to).Length -gt 10000)) {
            Write-Host "[Cache] $(Split-Path $to -Leaf)"
            return
        }
        Write-Host "[Download] $url" -ForegroundColor Yellow
        $part = "$to.part"
        if (Test-Path $part) { Remove-Item -Force $part }
        try {
            Invoke-WebRequest -Uri $url -OutFile $part -UseBasicParsing -MaximumRedirection 10
            if ((Get-Item $part).Length -lt 10000) { throw "Download returned an unexpectedly small file: $url" }
            Move-Item -Force $part $to
        } finally {
            if (Test-Path $part) { Remove-Item -Force $part }
        }
    }
    # A pinned, officially published uv binary. Executed ONLY after HTTPS download from official release.
    $Uv = Join-Path $Tools 'uv.exe'
    if (-not (Test-Path $Uv)) {
        $UvArchive = Join-Path $Downloads 'uv-0.13.3-win-x64.zip'
        Download-IfMissing 'https://github.com/astral-sh/uv/releases/download/0.13.3/uv-x86_64-pc-windows-msvc.zip' $UvArchive
        $Unpack = Join-Path $Downloads 'uv-extract'
        if (Test-Path $Unpack) { Remove-Item -Recurse -Force $Unpack }
        Expand-Archive -Path $UvArchive -DestinationPath $Unpack -Force
        $Found = Get-ChildItem $Unpack -Recurse -Filter 'uv.exe' -File | Select-Object -First 1
        if (-not $Found) { throw 'uv.exe was not found in the official release archive.' }
        Copy-Item $Found.FullName $Uv -Force
    }
    $env:UV_CACHE_DIR = $Cache
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Runtime 'python'
    $env:UV_PYTHON_BIN_DIR = Join-Path $Runtime 'python-bin'
    $env:UV_PYTHON_PREFERENCE = 'only-managed'
    $env:UV_LINK_MODE = 'copy'
    $env:PYTHONUTF8 = '1'
    if (-not $env:UV_HTTP_TIMEOUT) { $env:UV_HTTP_TIMEOUT = '600' }
    if (-not $env:UV_CONCURRENT_DOWNLOADS) { $env:UV_CONCURRENT_DOWNLOADS = '3' }
    # uv does not automatically use the Windows desktop proxy.
    if (-not $env:HTTPS_PROXY -and -not $env:ALL_PROXY) {
        $ProxySettings = Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' -ErrorAction SilentlyContinue
        if ($ProxySettings.ProxyEnable -eq 1 -and $ProxySettings.ProxyServer -and $ProxySettings.ProxyServer -notmatch '[=;]') {
            $ProxyUrl = $ProxySettings.ProxyServer
            if ($ProxyUrl -notmatch '^https?://') { $ProxyUrl = "http://$ProxyUrl" }
            $env:HTTPS_PROXY = $ProxyUrl
            if (-not $env:HTTP_PROXY) { $env:HTTP_PROXY = $ProxyUrl }
        }
    }
    Write-Host '\n[1/4] Preparing a private Python 3.11...' -ForegroundColor Cyan
    # Use the concrete runtime path: some Windows drives cannot create uv's
    # minor-version directory links, even when the runtime was unpacked correctly.
    $ManagedPython = Get-ChildItem $env:UV_PYTHON_INSTALL_DIR -Filter 'cpython-3.11.*-windows-x86_64-none' -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
    if (-not $ManagedPython) {
        & $Uv python install 3.11 --no-bin --no-registry
        $ManagedPython = Get-ChildItem $env:UV_PYTHON_INSTALL_DIR -Filter 'cpython-3.11.*-windows-x86_64-none' -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending | Select-Object -First 1
    }
    if (-not $ManagedPython) { throw 'The private Python download failed.' }
    $BasePython = Join-Path $ManagedPython.FullName 'python.exe'
    & $BasePython --version
    if ($LASTEXITCODE -ne 0) { throw 'The private Python runtime is not usable.' }
    Write-Host '\n[2/4] Creating a private environment...' -ForegroundColor Cyan
    $Python = Join-Path $Venv 'Scripts\python.exe'
    if (-not (Test-Path $Python)) {
        & $Uv venv $Venv --python $BasePython
        if ($LASTEXITCODE -ne 0) { throw 'Could not initialize private Python environment.' }
    }
    Write-Host '\n[3/4] Downloading inference libraries...' -ForegroundColor Cyan
    $NV = (Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue)
    if (-not $NV) {
        $Candidates = @(Join-Path $env:windir 'System32\nvidia-smi.exe'; Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe')
        foreach ($path in $Candidates) { if (Test-Path $path) { $NV = $path; break } }
    }
    $Choice = 'cpu'
    if ($NV) { $Choice = 'gpu' }
    Write-Host "Detected preference: $($Choice.ToUpper())" -ForegroundColor Green
    $LibMarker = Join-Path $Runtime 'installed_mode.txt'
    $VersionMarker = Join-Path $Runtime 'installed_version.txt'
    $HasCurrentVersion = (Test-Path $VersionMarker) -and ((Get-Content $VersionMarker -Raw).Trim() -eq '0.47.0')
    $Installed = (Test-Path $LibMarker) -and $HasCurrentVersion
    if ($Installed) {
        & $Python -c "import sys,torch,audio_separator,importlib.metadata as m; sys.exit(0 if m.version('audio-separator') == '0.47.0' and (sys.argv[1] != 'gpu' or (torch.version.cuda == '12.6' and m.version('onnxruntime-gpu') == '1.20.2')) else 1)" $Choice
        $Installed = $LASTEXITCODE -eq 0
    }
    if (-not $Installed) {
        $Spec = "audio-separator[$Choice]==0.47.0"
        # CUDA 12.6 supports this application's RTX 4060 / 566.xx driver target.
        $TorchBackend = if ($Choice -eq 'gpu') { 'cu126' } else { 'cpu' }
        $ExtraSpecs = @('audioread')
        # Newer ONNX GPU wheels require CUDA 12.8/13; match our CUDA 12.6 runtime.
        if ($Choice -eq 'gpu') { $ExtraSpecs += 'onnxruntime-gpu==1.20.2' }
        & $Uv pip install --python $Python --torch-backend $TorchBackend $Spec @ExtraSpecs
        if ($LASTEXITCODE -ne 0) {
            if ($Choice -eq 'gpu') {
                Write-Host 'GPU dependency installation failed, attempting CPU packages...' -ForegroundColor Yellow
                $Choice = 'cpu'
                & $Uv pip install --python $Python --torch-backend cpu 'audio-separator[cpu]==0.47.0' audioread
                if ($LASTEXITCODE -ne 0) { throw 'Could not install CPU libraries either. See runtime\install.log.' }
            } else {
                throw 'Could not install audio-separator dependencies. See runtime\install.log.'
            }
        }
        & $Uv pip install --python $Python 'imageio-ffmpeg>=0.5,<1'
        if ($LASTEXITCODE -ne 0) { Write-Host 'Optional bundled-FFmpeg fallback was not installed.' -ForegroundColor Yellow }
        Set-Content -Encoding UTF8 -Path $LibMarker -Value $Choice
        Set-Content -Encoding UTF8 -Path $VersionMarker -Value '0.47.0'
    } else {
        Write-Host '[Cache] Python packages already installed.' -ForegroundColor Green
    }
    Write-Host '\n[4/4] Preparing FFmpeg and FFprobe...' -ForegroundColor Cyan
    New-Item -Force -ItemType Directory -Path $FFbin | Out-Null
    if (-not (Test-Path (Join-Path $FFbin 'ffmpeg.exe')) -or -not (Test-Path (Join-Path $FFbin 'ffprobe.exe'))) {
        try {
            $FFarchive = Join-Path $Downloads 'ffmpeg-release-essentials.zip'
            Download-IfMissing 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' $FFarchive
            $Unpack = Join-Path $Downloads 'ffmpeg-extract'
            if (Test-Path $Unpack) { Remove-Item -Recurse -Force $Unpack }
            Expand-Archive -Path $FFarchive -DestinationPath $Unpack -Force
            foreach ($binary in @('ffmpeg.exe', 'ffprobe.exe')) {
                $Found = Get-ChildItem $Unpack -Recurse -Filter $binary -File | Select-Object -First 1
                if (-not $Found) { throw "$binary was missing from FFmpeg archive." }
                Copy-Item $Found.FullName (Join-Path $FFbin $binary) -Force
            }
            Remove-Item -Recurse -Force $Unpack
        } catch {
            Write-Host "FFmpeg archive issue: $($_.Exception.Message)" -ForegroundColor Yellow
            Write-Host 'Attempting bundled ffmpeg.exe fallback (ffprobe is still recommended)...' -ForegroundColor Yellow
            & $Python -c "import imageio_ffmpeg,shutil,sys;shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(),sys.argv[1])" (Join-Path $FFbin 'ffmpeg.exe')
            if ($LASTEXITCODE -ne 0) { throw 'Unable to install FFmpeg. Please check network access.' }
        }
    }
    # Validate imports and basic GPU discovery. GPU driver may be missing even if libraries were installed.
    $env:PATH = "$FFbin;$env:PATH"
    & $Python -c "import torch, audio_separator; print('Torch:', torch.__version__); print('CUDA ready:', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw 'Inference engine import test failed. See previous messages.' }
    $Desktop = [Environment]::GetFolderPath('Desktop')
    if ($Desktop -and (Test-Path $Desktop)) {
        try {
            $WS = New-Object -ComObject WScript.Shell
            $Link = $WS.CreateShortcut((Join-Path $Desktop 'STEMForge.lnk'))
            $Link.TargetPath = Join-Path $Root 'start.cmd'
            $Link.WorkingDirectory = $Root
            $Link.Description = 'STEMForge — Local Audio Stem Separator'
            $Link.Save()
            Write-Host '[OK] Desktop shortcut created.' -ForegroundColor Green
        } catch {
            Write-Host 'Unable to create a shortcut; start.cmd still works.' -ForegroundColor Yellow
        }
    }
    Write-Host '\n[SUCCESS] STEMForge is ready. No system Python installation needed.' -ForegroundColor Green
    Write-Host 'On first use, selected model weights are downloaded into data\models.'
    exit 0
} catch {
    Write-Host "\n[FAIL] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'Hint: Keep this package in a short, writable folder, e.g. D:\STEMForge.'
    Write-Host 'Check proxy, firewall and Windows Defender if downloads are blocked.'
    exit 1
} finally {
    Stop-Transcript | Out-Null
}

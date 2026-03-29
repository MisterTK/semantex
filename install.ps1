# semantex installer for Windows — downloads pre-built binary from GitHub Releases
# Usage (run in PowerShell as administrator or normal user):
#   irm https://raw.githubusercontent.com/MisterTK/semantex/main/install.ps1 | iex
#
# Options (set before running):
#   $env:SEMANTEX_VERSION = "v0.1.2"   # pin a specific version
#   $env:SEMANTEX_NO_TELEMETRY = "1"   # opt out of anonymous usage stats

$ErrorActionPreference = "Stop"

$Repo = "MisterTK/semantex"

function Info($label, $msg) { Write-Host "  " -NoNewline; Write-Host $label -ForegroundColor Blue -NoNewline; Write-Host " $msg" }
function Err($msg) { Write-Host "  error: $msg" -ForegroundColor Red; exit 1 }

# Detect architecture
$arch = if ([System.Environment]::Is64BitOperatingSystem) {
    $cpu = (Get-WmiObject Win32_Processor).Architecture
    # 12 = ARM64, 9 = x86_64
    if ($cpu -eq 12) { "aarch64" } else { "x86_64" }
} else {
    Err "32-bit Windows is not supported."
}
$target = "$arch-pc-windows-msvc"
Info "Platform" $target

# Determine version
if ($env:SEMANTEX_VERSION) {
    $version = $env:SEMANTEX_VERSION
} else {
    Info "Fetching" "latest release..."
    try {
        $rel = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest"
        $version = $rel.tag_name
    } catch {
        Err "Could not fetch latest release. Set `$env:SEMANTEX_VERSION = 'v0.1.0' and retry."
    }
    if (-not $version) { Err "Could not determine latest version." }
}
Info "Version" $version

# Download archive
$archive = "semantex-$version-$target.zip"
$url = "https://github.com/$Repo/releases/download/$version/$archive"
$tmp = Join-Path $env:TEMP "semantex-install-$(Get-Random)"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

$archivePath = Join-Path $tmp $archive
Info "Downloading" $url
try {
    Invoke-WebRequest -Uri $url -OutFile $archivePath -UseBasicParsing
} catch {
    Err "Download failed: $_`nCheck https://github.com/$Repo/releases for available builds."
}

# Verify checksum if available
$checksumUrl = "$url.sha256"
try {
    $checksumFile = Join-Path $tmp "$archive.sha256"
    Invoke-WebRequest -Uri $checksumUrl -OutFile $checksumFile -UseBasicParsing -ErrorAction Stop
    $expected = (Get-Content $checksumFile -Raw).Split(" ")[0].Trim().ToLower()
    $actual = (Get-FileHash $archivePath -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $expected) {
        Err "Checksum mismatch!`n  expected: $expected`n  got:      $actual"
    }
    Info "Checksum" "verified"
} catch {
    # Checksum not available — proceed without verification
}

# Extract
Expand-Archive -Path $archivePath -DestinationPath $tmp -Force
$extractDir = Join-Path $tmp "semantex-$version-$target"

# Choose install directory
# Prefer user-local install to avoid UAC: %LOCALAPPDATA%\semantex\bin
$installDir = Join-Path $env:LOCALAPPDATA "semantex\bin"
New-Item -ItemType Directory -Force -Path $installDir | Out-Null

$binSrc = Join-Path $extractDir "semantex.exe"
$binDst = Join-Path $installDir "semantex.exe"

Info "Installing" $binDst
Copy-Item $binSrc $binDst -Force

# Copy ONNX Runtime DLL alongside the binary (needed for model inference)
$dlls = Get-ChildItem (Join-Path $extractDir "onnxruntime*.dll") -ErrorAction SilentlyContinue
foreach ($dll in $dlls) {
    Copy-Item $dll.FullName (Join-Path $installDir $dll.Name) -Force
    Info "Copied" $dll.Name
}

# Telemetry: report install event (respects SEMANTEX_NO_TELEMETRY and DO_NOT_TRACK)
$noTelemetry = $env:SEMANTEX_NO_TELEMETRY -or $env:DO_NOT_TRACK -eq "1" -or $env:CI
if (-not $noTelemetry) {
    $posthogKey = "phc_UEenKOEhH6eTI11OwQgo5qxOumaPRHiBSgnqXBy5o6V"
    if ($posthogKey -ne "") {
        try {
            $machineId = ""
            $idFile = Join-Path $env:USERPROFILE ".semantex\telemetry_id"
            if (Test-Path $idFile) {
                $machineId = (Get-Content $idFile -Raw).Trim()
            }
            if (-not $machineId) {
                $machineId = [System.Guid]::NewGuid().ToString()
                New-Item -ItemType Directory -Force -Path (Split-Path $idFile) | Out-Null
                Set-Content $idFile $machineId
            }
            $payload = @{
                api_key = $posthogKey
                event = "command_run"
                distinct_id = $machineId
                properties = @{
                    command = "install"
                    version = $version
                    os = "windows"
                    arch = $arch
                    "`$lib" = "semantex"
                }
            } | ConvertTo-Json -Compress
            Invoke-RestMethod -Uri "https://app.posthog.com/capture/" `
                -Method Post -Body $payload `
                -ContentType "application/json" `
                -TimeoutSec 3 -ErrorAction SilentlyContinue | Out-Null
        } catch { }
    }
}

# Add install directory to user PATH if not already present
$userPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
if ($userPath -notlike "*$installDir*") {
    [System.Environment]::SetEnvironmentVariable(
        "PATH", "$installDir;$userPath", "User"
    )
    Info "Updated" "PATH (restart your terminal to apply)"
}
$env:PATH = "$installDir;$env:PATH"

# Clean up temp files
Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue

Write-Host ""
Info "Done!" "semantex $version is ready"
Write-Host ""
try { & semantex --version } catch { }
Write-Host ""
Write-Host "  Next: install into your AI coding tool:"
Write-Host "    semantex install-claude-code   # Claude Code"
Write-Host "    semantex install-codex         # Codex CLI"
Write-Host "    semantex install-open-code     # OpenCode"
Write-Host ""
Write-Host "  Disable telemetry anytime:"
Write-Host '    $env:SEMANTEX_NO_TELEMETRY = "1"   # current session'
Write-Host '    [System.Environment]::SetEnvironmentVariable("SEMANTEX_NO_TELEMETRY","1","User")  # permanent'
Write-Host ""

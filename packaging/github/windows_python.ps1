# Used before dependency setup; discovery never installs or replaces Python.
function Test-SessionPrepPython {
    param([string]$Candidate, [string]$Target)
    try {
        & $Candidate -c "import os,sys; sys.path.insert(0,os.environ['SOURCE_DIR']); from build_conf import check_windows_python; check_windows_python(sys.argv[1])" $Target | Out-Host
        return $LASTEXITCODE -eq 0
    } catch {
        Write-Host "Unable to validate Python at ${Candidate}: $($_.Exception.Message)"
        return $false
    }
}

function Find-SessionPrepPython {
    param([ValidateSet('win-x64', 'win-arm64')][string]$Target)

    if (-not [string]::IsNullOrWhiteSpace($env:CI_PYTHON)) {
        try {
            $candidate = if (Test-Path -LiteralPath $env:CI_PYTHON -PathType Leaf) {
                (Get-Item -LiteralPath $env:CI_PYTHON).FullName
            } else {
                (Get-Command -Name $env:CI_PYTHON -CommandType Application -ErrorAction Stop).Source
            }
        } catch {
            throw "CI_PYTHON does not identify an executable: $env:CI_PYTHON"
        }
        if (Test-SessionPrepPython -Candidate $candidate -Target $Target) { return $candidate }
        throw "CI_PYTHON must select native Python 3.13 for ${Target}: $env:CI_PYTHON"
    }

    $candidates = @(Get-Command python.exe -CommandType Application -All -ErrorAction SilentlyContinue |
        Where-Object { $_.Source -notlike '*\Microsoft\WindowsApps\*' } |
        ForEach-Object { $_.Source })
    $folder = if ($Target -eq 'win-arm64') { 'Python313-arm64' } else { 'Python313' }
    # CPython full-installer defaults, including ARM64 installed alongside x64.
    foreach ($root in @($env:ProgramFiles, $env:ProgramW6432)) {
        if (-not [string]::IsNullOrWhiteSpace($root)) {
            $candidates += Join-Path $root "$folder\python.exe"
        }
    }
    if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        $candidates += Join-Path $env:LOCALAPPDATA "Programs\Python\$folder\python.exe"
    }
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if ((Test-Path -LiteralPath $candidate -PathType Leaf) -and
            (Test-SessionPrepPython -Candidate $candidate -Target $Target)) {
            return $candidate
        }
    }
    throw "No native Python 3.13 for $Target was found on PATH or in the standard installer directories. Set CI_PYTHON to the matching python.exe; restart the runner after changing its environment."
}

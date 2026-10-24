<#
One-step setup on Windows. From this folder, in PowerShell:

    .\setup.ps1          create .venv, install packages, register the "Capstone (.venv)" notebook kernel
    .\setup.ps1 -Data    the same, then fetch the automatically available source files

If Windows blocks the script, run instead:
    powershell -ExecutionPolicy Bypass -File .\setup.ps1
#>
param([switch]$Data)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE). See the messages above." } }

Step "1/4  Find Python 3.10 or newer"
$base = $null
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $py) {
    $base = @($py)
    $found = & $py -c "import sys; print('%d.%d' % sys.version_info[:2])"
    if ($LASTEXITCODE -ne 0 -or [version]$found -lt [version]"3.10") {
        throw "Existing .venv uses an unsupported Python. Move it aside, install Python 3.11 or newer, then rerun setup."
    }
} else {
    foreach ($candidate in @(@("py", "-3.12"), @("py", "-3.11"), @("py", "-3"), @("python"))) {
        try {
            $exe, $options = $candidate[0], @($candidate | Select-Object -Skip 1)
            $found = & $exe @options -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and [version]$found -ge [version]"3.10") { $base = $candidate; break }
        } catch { }
    }
}
if (-not $base) {
    throw "Python 3.10 or newer was not found. Install Python 3.11 from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then run this again."
}
Write-Host "Using '$($base -join ' ')' (Python $found)"

Step "2/4  Create the project environment (.venv)"
if (Test-Path -LiteralPath $py) {
    Write-Host ".venv already exists"
} else {
    & $base[0] @($base | Select-Object -Skip 1) -m venv .venv
    Check "Creating .venv"
}

Step "3/4  Install packages (a few minutes the first time)"
& $py -m pip install --upgrade pip --quiet
Check "Updating pip"
& $py -m pip install -r requirements.txt --quiet
Check "Installing requirements.txt"

Step "4/4  Register the notebook kernel 'Capstone (.venv)'"
& $py -m ipykernel install --user --name capstone-week4 --display-name "Capstone (.venv)"
Check "Registering the kernel"
& $py -c "import cv2, numpy, pandas, sklearn; print('Packages OK: OpenCV', cv2.__version__, '| scikit-learn', sklearn.__version__)"
Check "Checking the packages"

if ($Data) {
    Step "Download automatically available data (Pump, Haul Truck, CORA thermal)"
    & $py Models\pump.py intake;                    Check "Pump data download"
    & $py Models\haul_truck.py intake --sensor-only; Check "Haul Truck data download"
    & $py Models\belt_conveyor.py intake;           Check "Belt Conveyor data download"
}

Step "Done"
Write-Host @"
Next:
  1. Open this folder in VS Code (File > Open Folder). Install the Python and Jupyter extensions if VS Code asks.
  2. Open Notebook\Mining_Equipment_Condition_Monitoring.ipynb.
     Check the top right says 'Capstone (.venv)' (click it to choose it if not), then click Run All.
  3. Run All shows saved results. To rerun one equipment, set ACTIVE_EQUIPMENT near the top.

Without the data the working notebook still shows the saved results.
To fetch the automatically available source files: .\setup.ps1 -Data
CORA vibration and other publisher-gated files still require the manual intake
steps in Dataset_Manifest\manifest.md before every strict experiment can rerun.
"@

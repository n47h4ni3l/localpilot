@echo off
setlocal
echo Installing LocalPilot. Windows may ask for administrator approval.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install-localpilot.ps1" %*
if errorlevel 1 (
    echo.
    echo Installation did not finish. Review the message above and rerun this installer.
    pause
    exit /b 1
)
exit /b 0

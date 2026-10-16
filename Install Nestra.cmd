@echo off
setlocal
title Install Nestra ALPHA
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-nestra-setup.ps1"
if errorlevel 1 (
  echo.
  echo Setup stopped. The message above explains what needs attention.
  pause
  exit /b 1
)
echo.
echo Setup finished. Review the installer message for any restart requirement.
pause

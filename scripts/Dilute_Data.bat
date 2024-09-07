@echo off
setlocal enabledelayedexpansion

:: Check if the correct number of arguments is provided
if "%~1"=="" (
    echo Usage: %0 directory_path percentage
    exit /b 1
)
if "%~2"=="" (
    echo Usage: %0 directory_path percentage
    exit /b 1
)

:: Get the directory path and percentage from arguments
set "dirPath=%~1"
set "percent=%~2"

:: Check if the directory exists
if not exist "%dirPath%" (
    echo Directory not found.
    exit /b 1
)

:: Validate the percentage
if "%percent%" lss 0 (
    echo Invalid percentage. Must be between 0 and 100.
    exit /b 1
)
if "%percent%" gtr 100 (
    echo Invalid percentage. Must be between 0 and 100.
    exit /b 1
)

:: Create a PowerShell command to perform the file deletion
set "psCommand=Get-ChildItem -Path '%dirPath%' | Where-Object { !($_.PSIsContainer) } | Get-Random -Count ( [math]::Round((%percent% / 100) * (Get-ChildItem -Path '%dirPath%' | Where-Object { !($_.PSIsContainer) } | Measure-Object).Count) ) | Remove-Item"

:: Run the PowerShell command
powershell -command "%psCommand%"

echo Random deletion of files complete.
endlocal

@echo off
setlocal enabledelayedexpansion


:: validate Inputs 

if "%~1" =="" (echo ERROR: No IP address provided.
    echo Usage: %~nx0 ^<IP_ADDRESS^> ^<PORT^>
    exit /b 1)

if "%~2"=="" (
    echo ERROR: No port provided.
    echo Usage: %~nx0 ^<IP_ADDRESS^> ^<PORT^>
    exit /b 1
)

set TARGET_IP=%~1
set TARGET_PORT=%~2


:: 1. Find the current GUI session ID
:: We look for the "Console" session or the active RDP-Tcp session
for /f "tokens=3,4" %%a in ('query session ^| findstr /i ">"') do (
    set SESSION_ID=%%a
)

:: Fallback: If 'query session' doesn't show an active marker, 
:: we grab it from explorer.exe via tasklist
if "%SESSION_ID%"=="" (
    for /f "tokens=4" %%a in ('tasklist /FI "IMAGENAME eq explorer.exe" /NH') do (
        set SESSION_ID=%%a
    )
)

:: 2. Execute PsExec with the dynamic Session ID
:: Using 'start' so we can capture the PID if needed, but PsExec usually handles the hook.

"C:\Users\tablet_touch\PsExec.exe" -s -i %SESSION_ID% "C:\Users\tablet_touch\touch_app.exe" %TARGET_IP% %TARGET_PORT%
::"C:\Users\AOLabs\Desktop\PsExec.exe" -s -i %SESSION_ID% "C:\Users\AOLabs\Desktop\touch_app.exe" 192.168.0.150 5005

:: 3. Handle Exit
:EXIT_ROUTINE
echo Interrupt received or process finished. 
taskkill /IM touch_app.exe /F >nul 2>&1
exit
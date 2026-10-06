@echo off
set "APP_NAME=touch_app.exe"

echo Searching for %APP_NAME%...

:: Check if the process exists
tasklist /FI "IMAGENAME eq %APP_NAME%" 2>NUL | find /I /N "%APP_NAME%">NUL
if "%ERRORLEVEL%"=="0" (
    echo Process found. Terminating %APP_NAME%...
    :: /F forces the process to stop, /T kills any child processes
    taskkill /F /IM "%APP_NAME%" /T
    echo Done.
) else (
    echo %APP_NAME% is not currently running.
)

pause
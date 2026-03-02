@echo off
echo Installation du demarrage automatique de MeshWiki...

set "STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT=%STARTUP_FOLDER%\MeshWiki.lnk"
set "TARGET=%~dp0run.bat"
set "WORKDIR=%~dp0"

powershell -NoProfile -Command ^
  "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath = '%TARGET%'; $s.WorkingDirectory = '%WORKDIR%'; $s.WindowStyle = 7; $s.Description = 'MeshWiki - Wikipedia offline via Meshtastic'; $s.Save()"

if exist "%SHORTCUT%" (
    echo Raccourci cree avec succes :
    echo   %SHORTCUT%
    echo MeshWiki demarrera automatiquement a l'ouverture de session.
) else (
    echo ERREUR : Le raccourci n'a pas pu etre cree.
    exit /b 1
)

pause

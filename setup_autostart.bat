@echo off
echo Setting up Telegram bot to auto-start...

set "STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "BAT_PATH=%~dp0start_bot.bat"

echo Creating startup shortcut...
powershell -Command "& {$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%STARTUP_FOLDER%\TelegramTradingBot.lnk'); $Shortcut.TargetPath = '%BAT_PATH%'; $Shortcut.Save()}"

echo.
echo Done! Telegram bot will now start automatically when Windows boots.
echo To remove: Delete "TelegramTradingBot.lnk" from your Startup folder
pause
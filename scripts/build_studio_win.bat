@echo off
rem Build the desktop app (WPF, .NET 8) into <project>\..\JunqiStudio  (needs the .NET 8+ SDK)
setlocal
set "ROOT=%~dp0.."
set DOTNET_CLI_TELEMETRY_OPTOUT=1
dotnet publish "%ROOT%\desktop\JunqiStudio\JunqiStudio.csproj" -c Release -o "%ROOT%\..\JunqiStudio" --nologo || exit /b 1
echo built: %ROOT%\..\JunqiStudio\JunqiStudio.exe

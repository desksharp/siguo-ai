@echo off
rem Build the C++ engine with MSVC (Windows): junqi.dll for Python ctypes, test_engine.exe, bench.exe
rem Usage: scripts\build_win.bat   (needs Visual Studio 2019/2022 or Build Tools with the C++ toolset)
setlocal
set "ROOT=%~dp0.."
set "OUT=%ROOT%\build"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS=%%i"
if not defined VS (echo MSVC C++ toolset not found & exit /b 1)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not exist "%OUT%\obj" mkdir "%OUT%\obj"
set FLAGS=/nologo /std:c++17 /O2 /DNDEBUG /EHsc /W3 /MT /utf-8 /arch:AVX2 /I"%ROOT%\engine\include" /Fo"%OUT%\obj\\"
set SRC="%ROOT%\engine\src\board.cpp" "%ROOT%\engine\src\rules.cpp" "%ROOT%\engine\src\game.cpp" "%ROOT%\engine\src\encode.cpp" "%ROOT%\engine\src\agents.cpp" "%ROOT%\engine\src\selfplay.cpp"
cl %FLAGS% /LD %SRC% "%ROOT%\engine\src\capi.cpp" /Fe:"%OUT%\junqi.dll" || exit /b 1
cl %FLAGS% %SRC% "%ROOT%\engine\tests\test_engine.cpp" /Fe:"%OUT%\test_engine.exe" || exit /b 1
cl %FLAGS% %SRC% "%ROOT%\engine\tests\bench.cpp" /Fe:"%OUT%\bench.exe" || exit /b 1
if not exist "%ROOT%\junqi\_lib" mkdir "%ROOT%\junqi\_lib"
copy /y "%OUT%\junqi.dll" "%ROOT%\junqi\_lib\" >nul || (echo junqi\_lib\junqi.dll is in use: stop training / the game server first & exit /b 1)
echo built: %OUT%\junqi.dll test_engine.exe bench.exe

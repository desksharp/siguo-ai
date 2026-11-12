@echo off
rem Full self-check on Windows: build -> C++ tests -> rule regression -> Python tests (incl. leak test)
rem Python: %PYTHON% if set, else .venv\Scripts\python.exe, else python on PATH
setlocal
set "ROOT=%~dp0.."
if not defined PYTHON (if exist "%ROOT%\.venv\Scripts\python.exe" (set "PYTHON=%ROOT%\.venv\Scripts\python.exe") else (set "PYTHON=python"))
call "%ROOT%\scripts\build_win.bat" || exit /b 1
"%ROOT%\build\test_engine.exe" || exit /b 1
"%PYTHON%" "%ROOT%\tests\test_regression.py" || exit /b 1
"%PYTHON%" "%ROOT%\tests\test_python.py" || exit /b 1

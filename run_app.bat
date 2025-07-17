@echo off
REM
call venv\Scripts\activate.bat

REM
start cmd /k python app.py

REM
timeout /t 5 /nobreak > NUL

REM
python startup.py
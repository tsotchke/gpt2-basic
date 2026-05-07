@echo off
set DOSDRV=A:
set DOSDIR=A:\FREEDOS
set PATH=A:\FREEDOS\BIN;C:\FBC;C:\FBC\BIN\DOS
set DIRCMD=/OGN /Y
set COPYCMD=/-Y

echo GPT2-BASIC QEMU 486 parity vector run
echo.
C:
cd \
if not exist GPT2.EXE goto missing_exe
if not exist MODEL\GPT2VEC.TXT goto missing_vectors
if exist VECTOR.LOG del VECTOR.LOG
echo Running C:\GPT2.EXE --vectors against MODEL\GPT2VEC.TXT...
GPT2.EXE --vectors > VECTOR.LOG
type VECTOR.LOG
goto done

:missing_exe
echo Missing C:\GPT2.EXE. Compile GPT2SRC\MAIN.BAS first.
goto done

:missing_vectors
echo Missing C:\MODEL\GPT2VEC.TXT. Run scripts\export_gpt2_basic_vectors.py on the host.

:done
echo.
echo Powering off QEMU.
A:\FREEDOS\BIN\FDAPM.COM POWEROFF

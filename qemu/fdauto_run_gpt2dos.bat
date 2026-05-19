@echo off
set DOSDRV=A:
set DOSDIR=A:\FREEDOS
set PATH=A:\FREEDOS\BIN;C:\FBC;C:\FBC\BIN\DOS
set DIRCMD=/OGN /Y
set COPYCMD=/-Y

echo GPT2-BASIC QEMU 486 run test
echo.
C:
cd \
if exist RUN.LOG del RUN.LOG
if exist GPT2DOS.EXE del GPT2DOS.EXE
echo Compiling GPT2SRC\GPT2DOS.BAS with DOS FreeBASIC...
C:\FBC\FBC.EXE GPT2SRC\GPT2DOS.BAS -x GPT2DOS.EXE > RUN.LOG
if errorlevel 1 goto compile_failed
echo Running GPT2DOS.EXE...
GPT2DOS.EXE >> RUN.LOG
type RUN.LOG
goto done

:compile_failed
echo COMPILE_FAILED
type RUN.LOG

:done
echo.
echo Powering off QEMU.
A:\FREEDOS\BIN\FDAPM.COM POWEROFF

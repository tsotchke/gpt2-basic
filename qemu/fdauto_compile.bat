@echo off
set DOSDRV=A:
set DOSDIR=A:\FREEDOS
set PATH=A:\FREEDOS\BIN;C:\FBC;C:\FBC\BIN\DOS
set DIRCMD=/OGN /Y
set COPYCMD=/-Y

echo GPT2-BASIC QEMU 486 compile test
echo.
C:
cd \
if exist COMPILE.LOG del COMPILE.LOG
if exist GPT2.EXE del GPT2.EXE
echo Compiling GPT2SRC\MAIN.BAS with DOS FreeBASIC...
C:\FBC\FBC.EXE GPT2SRC\MAIN.BAS -x GPT2.EXE > COMPILE.LOG
if errorlevel 1 goto compile_failed
echo COMPILE_OK
echo COMPILE_OK >> COMPILE.LOG
dir GPT2.EXE
dir GPT2.EXE >> COMPILE.LOG
goto done

:compile_failed
echo COMPILE_FAILED
echo COMPILE_FAILED >> COMPILE.LOG
type COMPILE.LOG

:done
echo.
echo Powering off QEMU.
A:\FREEDOS\BIN\FDAPM.COM POWEROFF

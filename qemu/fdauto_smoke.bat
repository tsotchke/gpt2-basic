@echo off
set DOSDRV=A:
set DOSDIR=A:\FREEDOS
set PATH=A:\FREEDOS\BIN;C:\FBC;C:\FBC\BIN\DOS
set DIRCMD=/OGN /Y
set COPYCMD=/-Y

echo GPT2-BASIC QEMU 486 smoke test
echo.
C:
cd \
if exist SMOKE.LOG del SMOKE.LOG
echo Compiling HELLO.BAS with DOS FreeBASIC...
C:\FBC\FBC.EXE HELLO.BAS > SMOKE.LOG
if errorlevel 1 goto compile_failed
echo Running HELLO.EXE...
HELLO.EXE >> SMOKE.LOG
type SMOKE.LOG
echo.
echo SMOKE_OK
goto done

:compile_failed
echo FBC compile failed.
type SMOKE.LOG

:done
echo.
echo Powering off QEMU.
A:\FREEDOS\BIN\FDAPM.COM POWEROFF

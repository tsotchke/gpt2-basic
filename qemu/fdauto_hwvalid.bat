@echo off
set DOSDRV=A:
set DOSDIR=A:\FREEDOS
set PATH=A:\FREEDOS\BIN;C:\FBC;C:\FBC\BIN\DOS
set DIRCMD=/OGN /Y
set COPYCMD=/-Y

echo GPT2-BASIC QEMU 486 hardware-capture rehearsal
echo.
C:
cd \GPT2
if not exist HWVALID.BAT goto missing_capture
call HWVALID.BAT
if exist HWVALID.LOG type HWVALID.LOG
goto done

:missing_capture
echo Missing C:\GPT2\HWVALID.BAT.

:done
echo.
echo Powering off QEMU.
A:\FREEDOS\BIN\FDAPM.COM POWEROFF

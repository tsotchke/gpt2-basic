@echo off
set DOSDRV=A:
set DOSDIR=A:\FREEDOS
set PATH=A:\FREEDOS\BIN;C:\FBC;C:\FBC\BIN\DOS
set DIRCMD=/OGN /Y
set COPYCMD=/-Y

echo GPT2-BASIC QEMU 486 config-open diagnostic
echo.
C:
cd \
echo n_embd=16>CONFIG.TXT
echo n_head=2>>CONFIG.TXT
echo n_layer=1>>CONFIG.TXT
echo n_positions=16>>CONFIG.TXT
echo vocab_size=512>>CONFIG.TXT
dir CONFIG.TXT
type CONFIG.TXT
C:\FBC\FBC.EXE OPENCFG.BAS -x OPENCFG.EXE
OPENCFG.EXE
echo.
echo Powering off QEMU.
A:\FREEDOS\BIN\FDAPM.COM POWEROFF

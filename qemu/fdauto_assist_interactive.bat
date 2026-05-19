@echo off
set DOSDRV=A:
set DOSDIR=A:\FREEDOS
set PATH=A:\FREEDOS\BIN;C:\FBC;C:\FBC\BIN\DOS
set DIRCMD=/OGN /Y
set COPYCMD=/-Y

echo GPT2-BASIC QEMU 486 interactive assistant
echo.
C:
cd \
if not exist GPT2SRC\ASSIST.BAS goto missing_source
if exist MODEL\GPT2CFG.TXT goto have_cfg
if exist MODEL\TINYCFG.TXT goto have_cfg
goto missing_model
:have_cfg
if exist MODEL\GPT2FX.BIN goto have_fx
if exist MODEL\TINYFX.BIN goto have_fx
goto missing_model
:have_fx
if exist MODEL\GPT2EXP.BIN goto have_exp
if exist MODEL\TINYEXP.BIN goto have_exp
goto missing_model
:have_exp
if exist ASSISTC.LOG del ASSISTC.LOG
echo Compiling interactive ASSIST.EXE pack shell...
fbc -x ASSIST.EXE GPT2SRC\ASSIST.BAS > ASSISTC.LOG
if errorlevel 1 goto assist_compile_failed
echo ASSIST_COMPILE_OK
echo.
echo Starting ASSIST.EXE. Use /about, /packs, /pack CHAT, /pack DOSHELP, /pack OFFICE, /u, /d, /h, /quit.
echo.
ASSIST.EXE
echo.
echo ASSIST.EXE exited. You are back at the DOS prompt.
goto done

:assist_compile_failed
echo ASSIST_COMPILE_FAILED
type ASSISTC.LOG
goto done

:missing_source
echo Missing GPT2SRC\ASSIST.BAS.
goto done

:missing_model
echo Missing fixed-point GPT2-BASIC model files in C:\MODEL.
echo Run scripts\train_gpt2_basic.py on the host first.

:done
echo.
echo Interactive demo remains open. Type EXIT or close QEMU when finished.

@echo off
set DOSDRV=A:
set DOSDIR=A:\FREEDOS
set PATH=A:\FREEDOS\BIN;C:\FBC;C:\FBC\BIN\DOS
set DIRCMD=/OGN /Y
set COPYCMD=/-Y

echo GPT2-BASIC QEMU 486 VGA visual trace suite
echo.
C:
cd \
if not exist GPT2.EXE goto missing_exe
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
if exist VISUAL.LOG del VISUAL.LOG
if exist TRACE.LOG del TRACE.LOG
if exist VISUALC.LOG del VISUALC.LOG
echo Compiling optional VISUAL.EXE trace visualizer...
fbc -x VISUAL.EXE GPT2SRC\VISUAL.BAS > VISUALC.LOG
if errorlevel 1 goto visual_compile_failed
echo Running C:\GPT2.EXE --trace with fixed-point GPT2-BASIC model...
GPT2.EXE --trace > TRACE.LOG
echo Running C:\VISUAL.EXE TRACE.LOG...
VISUAL.EXE TRACE.LOG VISUAL.LOG
type VISUAL.LOG
goto done

:visual_compile_failed
echo VISUAL_COMPILE_FAILED
type VISUALC.LOG
goto done

:missing_exe
echo Missing C:\GPT2.EXE. Compile GPT2SRC\MAIN.BAS first.
goto done

:missing_model
echo Missing fixed-point GPT2-BASIC model files in C:\MODEL.
echo Run scripts\train_gpt2_basic.py on the host first.

:done
echo.
echo Powering off QEMU.
A:\FREEDOS\BIN\FDAPM.COM POWEROFF

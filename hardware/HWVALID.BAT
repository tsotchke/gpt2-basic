@echo off
echo GPT2-BASIC physical hardware validation
echo.
if exist HWVALID.LOG del HWVALID.LOG
if exist QUAL.LOG del QUAL.LOG
if exist PERF.LOG del PERF.LOG
if exist ASSIST.LOG del ASSIST.LOG
if exist ASTRESS.LOG del ASTRESS.LOG
if exist ARECALL.LOG del ARECALL.LOG
if exist ASSISTC.LOG del ASSISTC.LOG

echo HW_CAPTURE_BEGIN>HWVALID.LOG
echo Run this batch from the staged C:\GPT2 directory.>>HWVALID.LOG
ver>>HWVALID.LOG

if not exist GPT2.EXE goto missing_exe
if not exist MODEL\GPT2CFG.TXT goto missing_model
if not exist MODEL\GPT2FX.BIN goto missing_model
if not exist MODEL\GPT2EXP.BIN goto missing_model

echo Running GPT2.EXE --quality-all...
echo HW_STEP^|quality_all>>HWVALID.LOG
GPT2.EXE --quality-all > QUAL.LOG

echo Running GPT2.EXE --perf...
echo HW_STEP^|perf>>HWVALID.LOG
GPT2.EXE --perf > PERF.LOG

if exist GPT2SRC\ASSIST.BAS goto compile_assist
if exist ASSIST.EXE goto run_assist
echo ASSIST_SKIPPED^|reason=no_source_or_exe>>HWVALID.LOG
goto done

:compile_assist
echo Compiling ASSIST.EXE...
echo HW_STEP^|assist_compile>>HWVALID.LOG
fbc -x ASSIST.EXE GPT2SRC\ASSIST.BAS > ASSISTC.LOG
if errorlevel 1 goto assist_compile_failed
echo ASSIST_COMPILE_OK>>ASSISTC.LOG
goto run_assist

:assist_compile_failed
echo ASSIST_COMPILE_FAILED>>ASSISTC.LOG
echo ASSIST_SKIPPED^|reason=compile_failed>>HWVALID.LOG
goto done

:run_assist
echo Running ASSIST.EXE --scripted...
echo HW_STEP^|assistant>>HWVALID.LOG
ASSIST.EXE --scripted > ASSIST.LOG
echo Running ASSIST.EXE --stress-probe...
echo HW_STEP^|assistant_stress>>HWVALID.LOG
ASSIST.EXE --stress-probe > ASTRESS.LOG
echo Running ASSIST.EXE --recall-probe...
echo HW_STEP^|assistant_recall>>HWVALID.LOG
ASSIST.EXE --recall-probe > ARECALL.LOG
goto done

:missing_exe
echo HW_CAPTURE_FAILED^|missing=GPT2.EXE>>HWVALID.LOG
echo Missing GPT2.EXE.
goto done

:missing_model
echo HW_CAPTURE_FAILED^|missing=MODEL>>HWVALID.LOG
echo Missing model files under C:\GPT2\MODEL.
goto done

:done
echo HW_CAPTURE_END>>HWVALID.LOG
echo.
echo Hardware validation capture complete.
echo Copy HWVALID.LOG, QUAL.LOG, PERF.LOG, ASSIST.LOG, ASTRESS.LOG, ARECALL.LOG,
echo ASSISTC.LOG, and HWNOTES.TXT back to the host for verification.

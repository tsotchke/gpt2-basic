# Physical Hardware Capture Kit

This directory is for running the QEMU-proven release on a real DOS machine.
The first solid-release gate is a physical 486-class system. Pentium hardware is
optional scaling evidence.

## Stage The DOS Directory

Create a DOS directory like this:

```text
C:\GPT2\GPT2.EXE
C:\GPT2\MODEL\GPT2CFG.TXT
C:\GPT2\MODEL\GPT2FX.BIN
C:\GPT2\MODEL\GPT2EXP.BIN
C:\GPT2\MODEL\VOCAB.BIN
C:\GPT2\PACKS\PACKS.TXT
C:\GPT2\PACKS\CHAT\...
C:\GPT2\PACKS\DOSHELP\...
C:\GPT2\PACKS\OFFICE\...
C:\GPT2\PACKS\DEV\...
C:\GPT2\PACKS\PORTABLE\...
C:\GPT2\GPT2SRC\ASSIST.BAS
C:\GPT2\HWVALID.BAT
C:\GPT2\HWNOTES.TXT
C:\GPT2\RETURN.TXT
```

The release zip keeps source paths readable for host systems, so copy the files
into the short DOS layout above before testing on plain DOS.

## Run On The Machine

From DOS:

```bat
C:
CD \GPT2
HWVALID.BAT
```

The batch file writes:

```text
HWVALID.LOG
QUAL.LOG
PERF.LOG
ASSIST.LOG
ASTRESS.LOG
ASSISTC.LOG
```

`ASSIST.LOG` is the five-pack scripted assistant proof. `ASTRESS.LOG` is the
50-reply stress probe for CHAT, DOSHELP, OFFICE, DEV, and PORTABLE on the same
machine.

Fill in `HWNOTES.TXT` with CPU, clock, RAM, DOS version, storage, cache/turbo
state, FreeBASIC version, and any setup notes.
The `Machine key:` value must match the host `--machine-key` argument used when
staging release evidence.

`RETURN.TXT` is included inside `C:\GPT2` as the DOS-side copy-back checklist.
Use it after the run to make sure every required log returns to the host.

## Verify On The Host

After copying the logs back to the host:

```sh
python3 scripts/verify_hardware_capture.py \
  --capture-dir /path/to/capture \
  --require-filled-notes
```

For release evidence, stage the accepted logs with a stable machine key:

```sh
python3 scripts/stage_hardware_capture_evidence.py \
  --capture-dir /path/to/capture \
  --machine-key 486dx2_66_dos622
```

The staging command writes the normalized `hardware_<machine>_*.log` files and
manifest documented in `docs/hardware-validation.md`.

After staging physical captures, regenerate the physical performance matrix:

```sh
python3 scripts/hardware_performance_matrix.py
```

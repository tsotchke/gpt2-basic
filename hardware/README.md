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
C:\GPT2\PACKS\DOSHELP\...
C:\GPT2\PACKS\OFFICE\...
C:\GPT2\GPT2SRC\ASSIST.BAS
C:\GPT2\HWVALID.BAT
C:\GPT2\HWNOTES.TXT
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
ASSISTC.LOG
```

Fill in `HWNOTES.TXT` with CPU, clock, RAM, DOS version, storage, cache/turbo
state, FreeBASIC version, and any setup notes.

## Verify On The Host

After copying the logs back to the host:

```sh
python3 scripts/verify_hardware_capture.py --capture-dir /path/to/capture
```

For release evidence, rename the accepted logs using the scheme documented in
`docs/hardware-validation.md`.

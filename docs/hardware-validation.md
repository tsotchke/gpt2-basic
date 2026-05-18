# GPT2-BASIC Physical Hardware Validation Plan

The preview release is already QEMU-verified. A solid release should add
physical hardware evidence, but a Pentium is not required to start or to close
the core release gate.

## Hardware Tiers

| Tier | Status | Hardware | Release Role | Required Logs |
|---|---|---|---|---|
| 0 | Complete | QEMU 486 profiles | Preview release gate | compile, quality, perf, assistant, vectors |
| 1 | Next gate | Any working 486-class DOS PC with 32-64 MB RAM | Solid release baseline | `QUAL.LOG`, `PERF.LOG`, `ASSIST.LOG`, `ASSISTC.LOG` |
| 2 | Useful | Faster 486DX2/DX4 or comparable late 486 board | Performance confidence | repeated `PERF.LOG`, optional kernel perf |
| 3 | Optional | Pentium 60/90/133+ | Scaling comparison only | `PERF.LOG`, optional quality confirmation |
| 4 | Optional | 386 or 486SX no-FPU class system | Compatibility stress test | quality and perf if memory allows |

Tier 1 is the important milestone. A Pentium broadens the performance table, but
it is not a blocker for a credible 486-focused release.

## Minimum Physical Test Set

The release package includes `hardware/HWVALID.BAT`, `hardware/HWNOTES.TXT`,
`RETURN.TXT`, and `hardware/README.md`. Stage the package into a short
`C:\GPT2` DOS layout, copy `HWVALID.BAT`, `HWNOTES.TXT`, and `RETURN.TXT` into
that directory, then run:

```bat
C:
CD \GPT2
HWVALID.BAT
```

Before moving to a physical machine, rehearse that exact DOS layout in QEMU:

```sh
bash qemu/run_hardware_capture_486.sh
```

The rehearsal stages `C:\GPT2`, runs `HWVALID.BAT`, extracts the same log set to
`qemu/evidence/hardware_capture_486_qemu/`, and writes
`qemu/evidence/hardware_capture_486_qemu_probe.log`.

Build the actual minimal transfer directory with:

```sh
python3 scripts/build_hardware_transfer.py --force
```

That writes `/private/tmp/gpt2-basic-hardware-transfer/GPT2`, a matching zip,
and a zip-level `.sha256` sidecar for host verification before copying to the
target machine.

That batch file runs the minimum test set:

```bat
GPT2.EXE --quality-all > QUAL.LOG
GPT2.EXE --perf > PERF.LOG
ASSIST.EXE --scripted > ASSIST.LOG
```

Also keep the assistant compile log when building on the target:

```bat
fbc -x ASSIST.EXE GPT2SRC\ASSIST.BAS > ASSISTC.LOG
```

If the machine cannot compile `ASSIST.EXE`, record that separately and run the
prebuilt assistant only if the binary was produced from the same release tree.
Use `C:\GPT2\RETURN.TXT` as the machine-side copy-back checklist after the run.

Back on the host, verify the captured directory with:

```sh
python3 scripts/verify_hardware_capture.py \
  --capture-dir /path/to/capture \
  --require-filled-notes
```

Then stage the accepted logs into release evidence filenames:

```sh
python3 scripts/stage_hardware_capture_evidence.py \
  --capture-dir /path/to/capture \
  --machine-key 486dx2_66_dos622
```

The staging command reruns the verifier first, refuses to overwrite existing
evidence unless `--force` is passed, and writes a manifest plus normalized log
names under `qemu/evidence/`. It requires filled hardware notes by default;
`--allow-template-notes` is only for scratch or emulator captures that should
not be treated as release evidence.

After staging one or more physical captures, regenerate the measured hardware
performance matrix:

```sh
python3 scripts/hardware_performance_matrix.py
```

That report only reads normalized `hardware_<machine>_perf.log` files. It does
not ingest QEMU `perf_486_*` logs or estimates.

## Acceptance Criteria

- `QUAL.LOG` shows the same prompt suite completing without runtime failure.
- `PERF.LOG` contains all `PERF_*` rows and reports runtime memory.
- `ASSIST.LOG` includes CHAT, DOSHELP, and OFFICE pack records plus per-pack
  usage instructions.
- `ASSISTC.LOG` includes `ASSIST_COMPILE_OK` when target-side compilation is
  attempted.
- The hardware notes identify machine key, CPU, clock, RAM, DOS version,
  FreeBASIC version, storage type, and whether caches/turbo were enabled.

## Evidence Naming

Use stable filenames so the reports can be added without ambiguity:

```text
qemu/evidence/hardware_<machine>_capture.log
qemu/evidence/hardware_<machine>_quality.log
qemu/evidence/hardware_<machine>_perf.log
qemu/evidence/hardware_<machine>_assistant.log
qemu/evidence/hardware_<machine>_assistant_compile.log
qemu/evidence/hardware_<machine>_notes.md
qemu/evidence/hardware_<machine>_manifest.md
qemu/evidence/hardware_performance_matrix.md
```

Example machine keys:

- `486dx2_66_dos622`
- `486dx4_100_freedos`
- `pentium133_dos622`

The `--machine-key` argument must match the `Machine key:` value in
`HWNOTES.TXT`. Release staging and the physical performance matrix both reject
mismatches so evidence filenames cannot drift away from the machine notes.

## Practical Acquisition Path

Start with whatever 486-class hardware is easiest to borrow, buy, or assemble.
A Pentium can wait until after the 486 baseline is captured. Useful sources are:

- local retro-computing groups or repair shops
- eBay or marketplace listings for tested 486 boards or complete systems
- university surplus or electronics recyclers
- friends with old industrial PCs, DOS tooling rigs, or ISA test benches

The release story improves most when we can say “this ran on a physical 486.”
Pentium numbers are nice, but secondary.

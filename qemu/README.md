# GPT2-BASIC on QEMU 486

This directory contains the FreeDOS/QEMU helpers for the full GPT2-BASIC port. The primary target is now the real module graph rooted at:

```bat
GPT2SRC\MAIN.BAS
```

The older `src/dos_gpt2_basic.bas` target is only a small smoke program retained for diagnostics. It is not the main runtime.

## Prerequisites

These scripts assume the local, ignored VM artifacts already exist:

```text
qemu/boot-test.img
qemu/gpt2hdd.img
```

The hard disk image must contain the DOS FreeBASIC compiler at `C:\FBC`.

## Compile

Train/export a baseline GPT2-BASIC model on the host with:

```sh
python3 scripts/train_gpt2_basic.py --profile 486sx-safe
```

The promoted default is a 4096-token lexicon checkpoint. To train a new
DOS-loadable lexicon candidate:

```sh
python3 scripts/train_gpt2_basic.py --profile 486sx-safe --tokenizer lexicon --vocab-size 4096 --include-docs --corpus-file data/domain_curriculum/gold_curriculum_v2.txt --output assets/gpt2_basic/MODEL_LEXICON_NEW
```

That writes `VOCAB.BIN` into the model directory. The QEMU helpers copy it to
`C:\MODEL`, and `GPT2.EXE` loads `MODEL\VOCAB.BIN` before validating the model
vocabulary size and tokenizer mode.

For the next production training pass, fetch the online corpus with provenance
and train a candidate from that corpus instead of relying on the tiny embedded
documents alone:

```sh
python3 scripts/fetch_online_training_corpus.py
python3 scripts/train_gpt2_basic.py --profile 486sx-safe --include-docs --corpus-file data/online_corpus/online_training_corpus.txt --corpus-weight 1 --output assets/gpt2_basic/MODEL_ONLINE_PRETRAIN
```

The corpus manifest is `data/online_corpus/SOURCE_MANIFEST.json`; the source
audit is `qemu/evidence/online_training_data_audit.md`. ShareAlike/GFDL online
sources are opt-in through `--include-sharealike`.
The first online-only candidate did not beat the active held-out baseline. The
current default instead comes from the audited gold lexicon curriculum; see
`qemu/evidence/gold_curriculum_v2_report.md` and
`qemu/evidence/domain_training_strategy_report.md`.

This writes the DOS-readable checkpoint into:

```text
assets/gpt2_basic/MODEL/GPT2CFG.TXT
assets/gpt2_basic/MODEL/GPT2WT.BIN
assets/gpt2_basic/MODEL/GPT2FX.BIN
assets/gpt2_basic/MODEL/GPT2EXP.BIN
assets/gpt2_basic/MODEL/VOCAB.BIN
```

The checkpoint is a real GPT-style transformer: learned token embeddings,
learned position embeddings, causal attention, feed-forward blocks, layer norms,
and an output head. Lexicon and BPE checkpoints add `VOCAB.BIN` beside those
files. The DOS demo prefers `GPT2FX.BIN`, which contains Q20.12 fixed-point
weights, plus `GPT2EXP.BIN`, the fixed-point attention exp lookup table. The
runtime can still load legacy `TINY*` checkpoint names, but new production
exports use the `GPT2*` files.

Validate the checkpoint with:

```sh
python3 scripts/model_report.py --model-dir assets/gpt2_basic/MODEL --strict
```

The QEMU compile/run helpers perform this validation before copying files into the FAT image.

```sh
bash qemu/compile_main_486.sh
```

The script regenerates the 8.3 DOS staging tree, copies the full source tree and exported `MODEL` directory into the FAT16 hard-disk image, installs `qemu/fdauto_compile.bat` as the boot floppy's `FDAUTO.BAT`, and starts:

```sh
qemu-system-i386 -machine isapc -cpu 486
```

Expected success marker:

```text
COMPILE_OK
```

The current in-VM build produces `C:\GPT2.EXE`.

## Run

```sh
bash qemu/run_main_486.sh
```

If the FreeDOS language menu appears, press Enter for English. At the GPT2-BASIC splash screen, press any key. Use menu option `1` for text completion.

After QEMU exits or is stopped after the FreeDOS poweroff message, the helper extracts `C:\RUN.LOG` to `qemu/evidence/run_main_486.log`.

The run boot script requires these files before launching `C:\GPT2.EXE`:

```text
C:\MODEL\GPT2CFG.TXT
C:\MODEL\GPT2FX.BIN
C:\MODEL\GPT2EXP.BIN
```

If those files are missing, the program stops instead of falling back to fake generated output.

## Quality Suite

Run the runtime-regression and held-out quality prompts through the compiled DOS fixed-point runtime:

```sh
bash qemu/run_quality_486.sh
```

The suite boots FreeDOS, launches `C:\GPT2.EXE --quality-all`, writes
`C:\QUAL.LOG`, and scores the extracted log into:

```text
qemu/evidence/quality_report_dos.md
qemu/evidence/quality_report_dos_heldout.md
qemu/evidence/quality_report_dos_all.md
```

The runtime-regression suite intentionally uses prompts that overlap the
training corpus. It is a model-loading and deterministic-output regression
check. Held-out quality is reported separately and should drive training and
architecture decisions.

Current DOS quality evidence:

```text
runtime-regression: PASS, 5/5 prompts, average 0.948
heldout:            PASS, 5/5 prompts, average 0.973
all:                PASS, 10/10 prompts, average 0.961
```

## Profile Pareto Report

Rank exported checkpoints against held-out quality, runtime memory, and the
available QEMU `--perf` measurements with:

```sh
python3 scripts/profile_pareto_report.py --refresh-heldout-float
```

The report is written to `qemu/evidence/profile_pareto_report.md`. The active
`MODEL` row prefers DOS fixed-point held-out evidence; non-active checkpoint
rows use host float held-out probes until they are copied into `C:\MODEL` and
run through DOS.

## Architecture Profile Sweep

Rank the trainer's actual architecture profiles with:

```sh
python3 scripts/architecture_profile_sweep.py
```

The report is written to `qemu/evidence/architecture_profile_sweep.md`. Unlike
the checkpoint Pareto report, this includes missing profiles as explicit
planning rows. QEMU scripts accept an optional model directory as their last
argument, so profile exports can be staged into `C:\MODEL` without replacing the
active checkpoint:

```sh
bash qemu/run_vectors_486.sh assets/gpt2_basic/MODEL_PROFILE_386_MIN
bash qemu/run_quality_486.sh assets/gpt2_basic/MODEL_PROFILE_386_MIN
bash qemu/run_perf_486.sh 486dx2-66 assets/gpt2_basic/MODEL_PROFILE_386_MIN
```

Current measured profile contrast: `386-min` is smaller and fastest under QEMU
`486dx2-66`, but its DOS held-out quality is lower than `486sx-safe`.
`486dx2-usable` has also been exported and measured; it is slower than
`486sx-safe` at the current training budget and does not improve held-out
quality yet.

## Performance Suite

Run the dedicated timing contract through the compiled DOS fixed-point runtime:

```sh
bash qemu/run_perf_486.sh 486dx2-66
```

Available profiles match the era-speed runner: `386dx-33`, `486sx-25`,
`486dx-33`, `486dx2-66`, `486dx4-100`, `pentium-60`, `pentium-133`, and
`host`.

The suite boots FreeDOS, launches `C:\GPT2.EXE --perf`, writes `C:\PERF.LOG`,
extracts it to `qemu/evidence/perf_486_<profile>.log`, and builds
`qemu/evidence/hardware_perf_report.md`.

The emitted `PERF_*` records are generated by the DOS executable itself. QEMU
profile rows are emulator evidence; physical 386/486/Pentium rows should use
the same `GPT2.EXE --perf` mode and the same report parser.

## Era-Speed Run

For approximate 486-era performance rather than host-speed emulation:

```sh
bash qemu/run_main_486_era.sh 486dx2-66
```

Available presets:

```text
386dx-33     QEMU 486,-fpu CPU model, icount shift=7, about 7.8 MIPS
486sx-25     QEMU 486,-fpu CPU model, icount shift=6, about 15.6 MIPS
486dx-33     QEMU 486 CPU model, icount shift=5, about 31.3 MIPS
486dx2-66    QEMU 486 CPU model, icount shift=4, about 62.5 MIPS
486dx4-100   QEMU 486 CPU model, icount shift=3, about 125 MIPS upper bound
pentium-60   QEMU pentium CPU model, icount shift=3, about 125 MIPS
pentium-133  QEMU pentium CPU model, icount shift=2, about 250 MIPS
host         No icount throttle
```

These profiles use QEMU `-icount ... sleep=on` with TCG. This is repeatable instruction-count throttling, not cycle-accurate emulation of a specific 486 motherboard, cache, memory bus, VGA card, or disk controller.

## Verified Path

The current port has been verified under QEMU with:

```sh
qemu-system-i386 -machine isapc -cpu 486 -m 64
```

Verified behavior:

- `GPT2SRC\MAIN.BAS` compiles with DOS FreeBASIC.
- `C:\GPT2.EXE` starts under FreeDOS.
- Text completion initializes the trained GPT2-BASIC model from `C:\MODEL`.
- A prompt encodes through the same byte/BPE/lexicon tokenizer contract used by the host tools.
- The primary demo path uses fixed-point weights and fixed-point inference kernels.
- The real GPT runtime uses a KV decode cache for normal in-window generation, while preserving the full-prefix path as a fallback.
- The old quality prior is disabled by default and is not the demo path.

Current measured demo output:

```text
What makes this real inference? The prompt is encoded into tokens, the transformer updates hidden state, the output head produces logits, and the sampler chooses the next token from the trained checkpoint.
```

Current performance baseline:

```text
┌──────────────────────────────┬────────────────────┬───────────────────┬───────────────────┐
│ Configuration                │ Tokens per Second  │ 70-Token Demo     │ 100-Token Equiv.  │
├──────────────────────────────┼────────────────────┼───────────────────┼───────────────────┤
│ 386DX/33-class no-FPU        │ 0.68               │ 103.3 seconds     │ 147.5 seconds     │
│ 486SX/25 no-FPU              │ 1.36               │ 51.6 seconds      │ 73.8 seconds      │
│ 486DX/33                     │ 2.71               │ 25.8 seconds      │ 36.9 seconds      │
│ 486DX2/66                    │ 5.42               │ 12.9 seconds      │ 18.4 seconds      │
│ 486DX4/100                   │ 8.14               │ 8.6 seconds       │ 12.3 seconds      │
│ Pentium 60                   │ 7.91               │ 8.9 seconds       │ 12.6 seconds      │
│ Pentium 133                  │ 16.27              │ 4.3 seconds       │ 6.1 seconds       │
│ QEMU 486dx2-66 --perf        │ 2.46               │ 28.5 seconds      │ 40.7 seconds      │
│ Host-speed QEMU --perf       │ 127.27             │ 0.55 seconds      │ 0.8 seconds       │
└──────────────────────────────┴────────────────────┴───────────────────┴───────────────────┘
```

The first seven rows come from the current code-count performance model in `qemu/evidence/era_performance_report.md`; the QEMU `486dx2-66 --perf` row comes from the promoted 4096-token lexicon default running inside FreeDOS. Real hardware timing is still required before making claims about a specific PC. This QEMU build does not expose a true 386 CPU model, so the `386DX/33-class` row remains a conservative target estimate, not a true 386 compatibility proof.

Current production memory footprint is 2,055,940 bytes. `--vectors` reaches about 1.96 MB peak memory during phase-parity validation.

QEMU's curses display can emit CP437 conversion warnings on macOS terminals. They are display-backend noise, not DOS program failures.

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

Speed-focused release candidates may include `GPT2HSL.BIN`, an optional
output-head shortlist. If present and valid, DOS keeps the full checkpoint
resident but scores only the listed output rows and clamps every other logit.
The measured candidate is
`assets/gpt2_basic/MODEL_HEADSHORTLIST2048_PROD_PROBE`.

Low-memory release candidates may also include `GPT2TQ4.BIN` and `GPT2HQ4.BIN`,
optional q4/log compressed token-embedding and output-head artifacts. If present
and valid, `GPT2.EXE` skips the full resident token embedding and output head,
dequantizes only the current token row, builds a small fixed-point output-head
decode table, and computes final logits from the compressed head. The measured
candidate is `assets/gpt2_basic/MODEL_TOKHEADQ4_PROD_PROBE`.

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
runtime-regression: PASS, 5/5 prompts, average 0.965
heldout:            PASS, 5/5 prompts, average 0.973
all:                PASS, 10/10 prompts, average 0.969
```

## Trace Suite

Run the educational step trace through the compiled DOS fixed-point runtime:

```sh
bash qemu/run_trace_486.sh
```

The suite boots FreeDOS, launches `C:\GPT2.EXE --trace`, writes `C:\TRACE.LOG`,
and extracts it to:

```text
qemu/evidence/trace_486.log
```

The log is intentionally machine-readable. It records `TRACE_MODEL`,
`TRACE_TOKENIZER`, prompt token pieces, each `TRACE_STAGE` forward/sample step,
each generated `TRACE_STEP`, the final decoded text, and `TRACE_END`. This is
the implemented DOS teaching surface for step-by-step inference inspection.

Run the VGA visual trace through the same compiled runtime:

```sh
bash qemu/run_visual_trace_486.sh
```

The suite launches `C:\GPT2.EXE --trace`, compiles the optional lab
`C:\VISUAL.EXE` trace visualizer, draws a Mode 13h token/progress view when
graphics are available, writes `C:\VISUAL.LOG`, and extracts it to:

```text
qemu/evidence/visual_trace_486.log
```

The log records `VISUAL_MODEL`, `VISUAL_GRAPHICS`, prompt-token color rows,
generated-token color rows, and `VISUAL_END`. The visual mode is intentionally
small: it proves the DOS graphics path exists without making VGA a dependency
for the production text runner.

## Assistant Packs

Run the optional Clippy-style assistant shell through FreeDOS:

```sh
bash qemu/run_assistant_486.sh
```

That command is the scripted evidence path. It compiles `C:\ASSIST.EXE`,
copies `assets/gpt2_basic/PACKS` to `C:\PACKS`, runs `ASSIST.EXE --scripted`,
and extracts:

```text
qemu/evidence/assistant_486.log
qemu/evidence/assistant_compile_486.log
```

Run the headless assistant stress gate with:

```sh
bash qemu/run_assistant_stress_486.sh
```

That path compiles `C:\ASSIST.EXE`, runs `ASSIST.EXE --stress-probe`, asks 18
original prompts across `CHAT`, `DOSHELP`, and `OFFICE`, then validates the
structured `ASSIST_REPLY` records with `scripts/stress_assistant_behavior.py`.
The validator rejects visible prompt leakage, repeated chunks, token-soup
outputs, empty answers, off-topic answers, and model-unavailable records. It
writes:

```text
qemu/evidence/assistant_stress_486.log
qemu/evidence/assistant_stress_compile_486.log
qemu/evidence/assistant_stress_report.md
```

The stress launcher uses QEMU curses mode and a timeout watchdog, so it does
not open or focus a GUI window.

Run the interactive QEMU window demo with:

```sh
bash qemu/run_assistant_interactive_486.sh
```

That launcher stages the same source, model, and packs, then boots a graphical
QEMU display and starts `ASSIST.EXE` normally. It does not pass `--scripted`,
does not redirect output to `ASSIST.LOG`, and does not power off after startup.
Inside DOS, type questions directly. The demo starts in `/pack CHAT`, the
general conversation pack. Use `/about` for current-pack instructions, `/packs`,
`/pack CHAT`, `/pack DOSHELP`, `/pack OFFICE`, `/history`, `/up`, `/down`,
`/home`, `/end`, `/clear`, and `/quit`.
`ASSIST.EXE` loads the active pack model before it prints the first `>` prompt.
When you switch with `/pack NAME`, the selected pack model loads before the
next prompt returns, so generation starts immediately after the first question
for that pack.
During generation, `ASSIST.EXE` streams `Thinking:` progress for prompt tokens,
context prefill, and output-token sampling, then streams `Answer:` pieces as
fixed-point inference produces tokens.
The `/up` and `/down` commands page through the in-DOS transcript because QEMU
graphics windows do not provide terminal scrollback. The shorter `/u`, `/d`,
and `/h` aliases do the same paging/history work with less typing.
The current CHAT model is a pack-local 4096-token lexicon checkpoint. It is
trained from broader casual English dialogue plus `CHAT\LEXICON.TSV`, passes
the fixed quality suite at `25/25`, and has been manually probed in QEMU with
casual prompts including `i am bored`, `tell me a joke`, and
`do you like music`.

The assistant shell is intentionally separate from `GPT2.EXE`. Packs are
listed in `PACKS\PACKS.TXT`; each pack has `PACK.INI`, `HELP.TXT`, `USAGE.TXT`,
and optional pack assets. `CHAT` also ships `GOLDEN.TXT` common English
dialogue and `LEXICON.TSV` grammar entries; exact `GOLDEN.TXT` matches are used
as high-confidence runtime answers before retrieval and generation. `USAGE.TXT`
explains what the individual pack is for, how it works, and what to type;
`ASSIST.EXE` displays it through `/about`.
`PACK.INI` can point `MODEL=` at `C:\MODEL` or a pack-local checkpoint, so a
future DOS, Windows, or OS/2 shell can share the same pack format while
rendering a richer UI. The current release focus is the DOS reference shell and
DOS demo package; the OS/2/Warp package is deferred to a later release. The
current DOS renderer uses a text-mode assistant bubble and action list, which
keeps it compatible with plain VGA text, serial capture, and OS/2 DOS sessions.

Train and test every listed assistant pack model with:

```sh
python3 scripts/train_assistant_pack_models.py
```

The sweep generates `TRAIN.TXT` from each pack's metadata and retrieval notes,
fine-tunes `PACKS\<ID>\MODEL` from the current 486-safe base checkpoint,
validates the exported model files, runs short assistant-window quality
prompts, and writes:

```text
qemu/evidence/assistant_pack_model_training_summary.md
qemu/evidence/assistant_pack_backend_probe.log
qemu/evidence/train_assistant_<pack>.log
qemu/evidence/model_report_assistant_<pack>.log
qemu/evidence/quality_report_assistant_<pack>.md
```

The host quality gate uses a 96-token assistant reply window and requires all
pack prompts to pass at `0.90`. It rejects prompt-label leakage, truncated
endings, triple-character spelling glitches, and replies that do not reach the
tail of the expected pack answer. `ASSIST.EXE` keeps interactive generation
bounded to 64 tokens with early sentence stopping, and the scripted 486 probe
uses retrieval-only bubbles so the DOS evidence run still exercises pack-local
model loading without turning every release check into a long generation run.

## Sampling Matrix

Run the non-greedy sampling matrix through the compiled DOS fixed-point runtime:

```sh
bash qemu/run_sampling_486.sh
```

The suite boots FreeDOS, launches `C:\GPT2.EXE --sampling-matrix`, writes
`C:\SAMPLE.LOG`, and extracts it to:

```text
qemu/evidence/sampling_486.log
```

The matrix uses fixed seeds and records greedy, top-k, and nucleus-style rows.
Each `SAMPLING_RESULT` includes generated-token count, elapsed seconds, byte
fallback count, alphabetic byte fallback count, sentence-ending flag, last
token, and decoded text. Greedy quality remains the deterministic release gate;
this suite is the product evidence surface for interactive sampling settings.

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

Audit all exported model directories, including assistant pack-local models,
with:

```sh
python3 scripts/audit_exported_models.py --refresh-model-reports
```

The inventory is written to
`qemu/evidence/exported_model_quality_inventory.md`. It runs strict artifact
validation for each checkpoint, records `model_inventory_<model>.log`, links
the best matching quality report, and separates promoted/passable checkpoints
from models that still need training.

Refresh strict all-suite quality reports for every DOS-ready root export with:

```sh
python3 scripts/refresh_model_quality_reports.py
```

Then generate the repair plan:

```sh
python3 scripts/plan_model_quality_repairs.py
```

Those reports distinguish release-ready exports from historical failed
candidates, host-only prototypes, and the small set of profiles worth another
training run. The stricter gate treats malformed fragments, unclean endings,
and high phrase repetition as hard failures.

Build the iterative preview-release manifest with:

```sh
python3 scripts/build_preview_release.py --manifest-only
```

The manifest is written to `qemu/evidence/preview_release_manifest.md` and
includes only strict-quality release models plus assistant packs, source,
scripts, and host verification tests. To create the local package tree and zip
under `/private/tmp`, run:

```sh
python3 scripts/build_preview_release.py --force
```

Build the physical-DOS transfer bundle with:

```sh
python3 scripts/build_hardware_transfer.py --force
```

Both release zip builders use deterministic archive metadata, and the
preview manifest uses a pinned release date plus portable artifact names by
default. The `Preview Release` workflow checks that a no-change rebuild keeps
the same zip checksum; pass `--generated-date YYYY-MM-DD` only for a deliberate
release respin.

Publish the preview zip, hardware-transfer zip, both `.sha256` sidecars, and
`qemu/evidence/preview_release_manifest.md`. The `Preview Release` GitHub
Actions workflow uploads those same files as the
`gpt2-basic-v0.1.0-preview` workflow artifact after verification. Verify the
release archives before attaching them:

```sh
python3 scripts/verify_preview_artifacts.py
```

That verifier checks checksums, zip sidecars, extracted zip payloads, live tree
versus zip payload consistency, the required DOS demo evidence, the exact
release model/packs set, absence of deferred media/VM payloads and transient
host-cache artifacts, and the 8.3-safe hardware transfer manifest. It also
rejects malformed SHA-256 fields and duplicate, unsorted, escaping, or
non-normalized archive/manifest paths. The verifier enforces deterministic ZIP
entry metadata as well. The release notes include the equivalent consumer-side
command for verifying both downloaded zips from inside an extracted preview
tree.

QEMU scripts extract DOS `.LOG` files through `fat_image_put.py --get-text` so
tracked evidence stays LF-normalized after a fresh FreeDOS run. Raw `--get` is
reserved for binary artifacts such as `GPT2.EXE`.

Refresh the evidence-driven improvement queue with:

```sh
python3 scripts/write_improvement_backlog.py
```

That writes `qemu/evidence/improvement_backlog.md`, separating preview-release
work, retraining, runtime tuning, assistant packs, Windows/OS2 shells, and real
hardware validation. For the current preview, only the DOS demo and DOS package
are in release scope; OS/2/Warp packaging remains backlog work for a later
release.

The physical-hardware release ladder is in
[`docs/hardware-validation.md`](../docs/hardware-validation.md). The next
non-emulator gate is one real 486-class DOS machine running the same
`--quality-all`, `--perf`, and assistant probes. Pentium data is useful for
scaling comparisons but is not required before calling the 486-focused release
solid.

Before the package leaves the emulator, rehearse that exact capture path with:

```sh
bash qemu/run_hardware_capture_486.sh
```

That stages `C:\GPT2`, runs `hardware/HWVALID.BAT` under FreeDOS, extracts the
same logs expected from a physical machine, and verifies them with
`scripts/verify_hardware_capture.py`.

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

Use the optional third argument `kernel` to enable DOS-side per-stage timing:

```sh
bash qemu/run_perf_486.sh 486dx2-66 assets/gpt2_basic/MODEL kernel
```

That mode launches `C:\GPT2.EXE --kernel-perf`, writes `KERNEL_PERF_*` records
into the same log format, and preserves the normal `PERF_*` records for timing
comparison. The full-head large-vocabulary release spends about 73.7% of
measured decode time in the final output head on the QEMU 486DX2/66 profile.
The 2,048-row `GPT2HSL.BIN` candidate lowers that final-head kernel time from
30.75 seconds to 17.93 seconds.

The emitted `PERF_*` records are generated by the DOS executable itself. QEMU
profile rows are emulator evidence; physical hardware rows should use the same
`GPT2.EXE --perf` mode and the same report parser. The first non-emulator gate
is a 486-class DOS machine; Pentium rows can be added later as optional scaling
evidence.

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
- `GPT2SRC\MAIN.BAS` is the slim production entrypoint generated from
  `src/main_prod.bas`; the old combined lab driver is staged as `LABMAIN.BAS`.
- `C:\GPT2.EXE` starts under FreeDOS.
- Text completion initializes the trained GPT2-BASIC model from `C:\MODEL`.
- A prompt encodes through the same byte/BPE/lexicon tokenizer contract used by the host tools.
- The primary demo path uses fixed-point weights and fixed-point inference kernels.
- Optional `GPT2TQ4.BIN` token-embedding compression and `GPT2HQ4.BIN`
  output-head compression load and run in DOS.
- Optional `GPT2HSL.BIN` output-head shortlists load and run in DOS, with
  vector probes that verify non-shortlisted logits are clamped.
- `GPT2.EXE --trace` emits prompt-token and generation-step records from the
  real fixed-point runtime into `C:\TRACE.LOG`.
- `GPT2.EXE --sampling-matrix` emits fixed-seed temperature/top-k/top-p rows
  from the same fixed-point sampler into `C:\SAMPLE.LOG`.
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
│ QEMU 386dx-33 no-FPU         │ 0.31               │ 228.1 seconds     │ 325.8 seconds     │
│ QEMU 486sx-25 no-FPU         │ 0.61               │ 114.0 seconds     │ 162.9 seconds     │
│ QEMU 486dx-33                │ 1.23               │ 57.0 seconds      │ 81.4 seconds      │
│ QEMU 486dx2-66 --perf        │ 2.46               │ 28.4 seconds      │ 40.6 seconds      │
│ QEMU 486dx4-100              │ 4.91               │ 14.2 seconds      │ 20.4 seconds      │
│ QEMU pentium-60              │ 4.92               │ 14.2 seconds      │ 20.3 seconds      │
│ QEMU pentium-133             │ 9.85               │ 7.1 seconds       │ 10.2 seconds      │
│ QEMU 486dx2-66 head shortlist│ 3.35               │ 20.9 seconds      │ 29.9 seconds      │
│ QEMU 486dx2-66 q4 head       │ 2.12               │ 33.0 seconds      │ 47.1 seconds      │
│ QEMU 486dx2-66 q4 tok+head   │ 2.12               │ 33.0 seconds      │ 47.1 seconds      │
│ QEMU 486dx2-66 q4 streaming  │ 0.81               │ 86.3 seconds      │ 123.5 seconds     │
│ Host-speed QEMU --perf       │ 43.55              │ 1.6 seconds       │ 2.3 seconds       │
└──────────────────────────────┴────────────────────┴───────────────────┴───────────────────┘
```

All rows in this table are `GPT2.EXE --perf` measurements from the promoted
4096-token lexicon default and q4/log release candidates running inside
FreeDOS. Real hardware timing is still required before making claims about a
specific PC. This QEMU build does not expose a true 386 CPU model, so the
`386dx-33` row uses QEMU's `486,-fpu` CPU model with conservative instruction
throttling; it is a no-FPU class timing gate, not a true 386 compatibility
proof.

Current production memory footprint is 2,055,940 bytes. `--vectors` reaches about 1.96 MB peak memory during phase-parity validation. The head-shortlist speed candidate uses 2,064,148 runtime bytes, passes 3/3 vectors and 39/39 phase checks with masked-logit probes, and raises QEMU 486DX2/66 throughput to 3.35 tok/s by scoring 2,048 output rows instead of the full 4,096. The q4/log output-head candidate uses 1,646,404 runtime bytes and reaches about 1.57 MB peak memory. The q4/log token+head candidate uses 974,724 runtime bytes and reaches about 0.93 MB peak memory while still passing 3/3 vectors and 39/39 phase checks. The q4/log streamed-head candidate uses 616,324 runtime bytes and passes the same vector gate, but slows to 0.81 tok/s on the QEMU 486DX2/66 profile.

Keep the full resident, head-shortlist, resident compressed, and streamed
compressed checkpoints. The head-shortlist path is the fastest measured
large-vocabulary mode. Resident q4 token+head mode is the preferred low-memory
release profile because it stays close to the full-head speed. Streamed q4
output-head mode is the maximum-compatibility fallback for tighter RAM budgets;
its lower throughput is the cost of replacing resident output-head codes and
decode tables with disk-row reads.

QEMU's curses display can emit CP437 conversion warnings on macOS terminals. They are display-backend noise, not DOS program failures.

# GPT2-BASIC Architectural and Codebase Audit

Audit date: 2026-05-07

Scope: current `gpt2-basic` workspace, ICC repo `gpt2_basic`, target `gpt2-basic-486-production`, evidence directory `qemu/evidence`.

## ICC Summary

| Check | Result |
|---|---|
| Indexed files | 437 |
| Indexed lines | 120,347 |
| Contract gaps | 0 |
| Completion oracle | complete |
| Runtime evidence | present |
| Readiness score | 100 |
| Aspirational software readiness | ready, score 100 |
| Source drift | false |
| Main blind spot | BASIC files indexed as text, so ICC cannot prove BASIC function structure |

ICC confirms that the current evidence directory contains compile, vector, quality, run, perf, tokenizer, training-helper, FAT image, and profile-Pareto artifacts. After adding focused probe logs for the host helpers, ICC reports no contract gaps, no liveness actions, and no unmarked stubbed production paths.

ICC also confirms the formerly aspirational software target
`gpt2-basic-aspirational-software`: `ready`, score 100, zero contract gaps, and
zero stub/fallback blockers. The closure probe is
`scripts/verify_aspirational_software.py`; evidence is stored in
`qemu/evidence/aspirational_software_closure.md` and
`qemu/evidence/aspirational_software_closure.json`.

ICC index-quality also flags all large BASIC sources as symbol blind spots, including `src/real_gpt.bas`, `src/main.bas`, `src/matrix_ops.bas`, `src/simd_ops.bas`, and `src/memory_manager.bas`. Manual BASIC audit is therefore required for production claims.

## Current Release Update

This audit began before the final lexicon and q4/log release work. The current
promoted default is no longer the 258-token byte checkpoint described in the
older baseline rows below. The release state is:

- Default checkpoint: `assets/gpt2_basic/MODEL`, promoted from
  `MODEL_LEXICON_GOLD_V2_S3000`.
- Shape: `2L 48D 4H ctx192 hidden192 vocab4096`, 463,168 parameters.
- Tokenizer: DOS-loadable 4096-token lexicon vocabulary with byte fallback.
- Default DOS runtime memory: 2,055,940 bytes.
- Default QEMU 486DX2/66 perf: 2.45 tok/s.
- Default DOS all-suite quality: 10/10 average 0.961.
- Optional speed release mode: `MODEL_HEADSHORTLIST2048_PROD_PROBE` with
  `GPT2HSL.BIN`, a 2,048-row output-head shortlist.
- Head-shortlist mode DOS runtime memory: 2,064,148 bytes.
- Head-shortlist mode QEMU 486DX2/66 kernel perf: 3.35 tok/s, with final-head
  time reduced from 30.75 s to 17.93 s.
- Head-shortlist mode vector parity: 3/3 vectors, 39/39 phases,
  `VECTOR_CHECK_OK`, including masked-logit probe indexes.
- Optional compressed release mode: `MODEL_TOKHEADQ4_PROD_PROBE` with
  `GPT2TQ4.BIN` q4/log token-embedding artifact and `GPT2HQ4.BIN` q4/log
  output-head artifact.
- Q4 token+head mode DOS runtime memory: 974,724 bytes.
- Q4 token+head mode QEMU 486DX2/66 perf: 2.12 tok/s.
- Q4 token+head mode vector parity: 3/3 vectors, 39/39 phases,
  `VECTOR_CHECK_OK`.
- Q4 token+head streamed-head mode: `MODEL_TOKHEADQ4_STREAM_PROD_PROBE`,
  `GPT2HQS.ON` marker, 616,324 bytes DOS runtime memory, 0.81 tok/s on the
  QEMU 486DX2/66 gate.
- Q4 streamed-head vector parity: 3/3 vectors, 39/39 phases,
  `VECTOR_CHECK_OK`.
- Production entrypoint: `src/main_prod.bas`, staged as `GPT2SRC\MAIN.BAS`.
  The old combined driver is staged separately as `LABMAIN.BAS`.
- Slim production executable: `COMPILE_OK`, `GPT2.EXE` 309,760 bytes.
- Educational trace mode: `GPT2.EXE --trace`, launched by
  `qemu/run_trace_486.sh`, emits prompt-token and generation-step `TRACE_*`
  records from the real DOS fixed-point runtime.
- Non-greedy sampling matrix: `GPT2.EXE --sampling-matrix`, launched by
  `qemu/run_sampling_486.sh`, emits fixed-seed greedy/top-k/top-p
  `SAMPLING_*` rows from the real DOS fixed-point sampler.
- Default production QEMU 486DX2/66 perf after the split: 108 tokens in
  44.00 seconds, 2.45 tok/s.
- Kernel timing mode: `GPT2.EXE --kernel-perf`, emitted through
  `qemu/run_perf_486.sh ... kernel`.
- Formerly aspirational software surfaces are source-verified: memory tracker,
  matrix pool, parameter streaming, block-sparse attention, SIMD-like packed
  operations, assembly/fallback fixed point, production Q20.12 fixed point, and
  benchmark/emulator evidence hooks.
- Current kernel hot spot in the full-head baseline: the 4096-token final
  output head accounts for about 73.7% of measured decode time on the QEMU
  486DX2/66 profile. The head-shortlist variant is now the measured mitigation.
- Fixed decode now has rolling-window continuation after the exported context
  window, and interactive fixed-point sampling supports temperature, top-k, and
  top-p. Evidence runs keep temperature 0 for deterministic parity.

The stale findings below are retained as the historical audit trail. Their
resolved items are superseded by `qemu/evidence/domain_training_strategy_report.md`,
`qemu/evidence/hardware_perf_report.md`, the current slim production build, and
the current ICC readiness report.

## Current Production Architecture

The actual production path is now the trained GPT2-BASIC runtime:

1. Host training/export writes `GPT2CFG.TXT`, `GPT2WT.BIN`, `GPT2FX.BIN`, `GPT2EXP.BIN`, `VOCAB.BIN`, and `PROFILE.TXT` from `scripts/train_tiny_gpt.py`.
2. DOS `GPT2.EXE` initializes the byte fallback tokenizer in `src/tokenizer.bas` and loads `VOCAB.BIN` from the executable directory or `MODEL\` when a BPE or lexicon checkpoint provides one.
3. `src/main_prod.bas` loads `C:\MODEL` through `GPT2BasicLoadModel`.
4. `src/real_gpt.bas:438` prefers Q20.12 fixed weights through `TinyGPTLoadFixedModel`.
5. `src/real_gpt.bas:1627` runs the fixed-point cached token forward pass.
6. `src/real_gpt.bas` applies the printable/EOT/tokenizer output mask, greedy temperature-0 selection for evidence, and fixed-point temperature/top-k/top-p sampling for interactive use.
7. `src/main_prod.bas` exposes `GPT2.EXE --perf`, `--kernel-perf`, `--trace`, `--sampling-matrix`, `--vectors`, and quality suites with machine-readable evidence records.

This is a real inference path. The current checkpoint is `486sx-safe`, shape `2L 48D 4H ctx192 hidden192 vocab4096`, with 463,168 parameters and 2,055,940 bytes reported runtime memory.

## Evidence Baseline

| Evidence | Result |
|---|---|
| DOS compile | `COMPILE_OK`, `GPT2.EXE` 309,760 bytes |
| Vector parity | `VECTOR_SUMMARY passed 3 of 3`, `PHASE_SUMMARY passed 39 of 39`, `VECTOR_CHECK_OK` |
| DOS quality suite | `PASS`, 10/10 prompts, average 0.961 |
| Held-out DOS quality suite | `PASS`, 5/5 prompts, average 0.973 |
| DOS educational trace | `TRACE_BEGIN`, prompt token records, 12 generation steps, `TRACE_END` |
| DOS sampling matrix | greedy/top-k/nucleus rows, finite temperatures, byte-fallback counts |
| QEMU `486dx2-66 --perf` | 35-token `real_inference`: 14.12 s, 2.48 tok/s |
| QEMU `486dx2-66 --perf` full suite | 108 tokens in 44.00 s, 2.45 tok/s |
| QEMU `486dx2-66 --kernel-perf` | final output head: 30.75 s, 73.7% of measured kernel time |
| QEMU head-shortlist variant | 106 tokens in 31.64 s, 3.35 tok/s; final head 17.93 s |

The new `--perf` contract is a major improvement over host-side stopwatch claims because timing is emitted by the DOS executable itself. QEMU `-icount` remains emulator evidence, not physical-board proof.

## High-Priority Findings

### 1. Quality Evidence Is Overfit To Training Text

Severity: high

The quality suite prompts in `src/main.bas:1204` through `src/main.bas:1212` and `scripts/evaluate_gpt2_basic_quality.py:64` through `scripts/evaluate_gpt2_basic_quality.py:90` appear verbatim or near-verbatim in the curated training corpus. Examples:

| Quality prompt | Training corpus overlap |
|---|---|
| `What makes this real inference?` | `scripts/train_tiny_gpt.py:230` |
| `GPT2 BASIC on a 486` | `scripts/train_tiny_gpt.py:46` |
| `A BASIC transformer runtime` | `scripts/train_tiny_gpt.py:100`, `:105`, `:110`, `:115`, `:120` |
| `To improve performance on real hardware` | `scripts/train_tiny_gpt.py:125`, `:130`, `:135`, `:140`, `:145` |

The DOS quality report confirms this: completions in `qemu/evidence/quality_report_dos.md:27`, `:36`, `:44`, `:52`, and `:60` are direct continuations of training phrases. This proves the runtime can load weights and reproduce learned sequences, but it does not prove useful generalization.

Implemented next step: quality is now split into two tiers:

1. Runtime-golden suite: the current prompts, explicitly labeled as memorization/regression probes.
2. Held-out quality suite: prompts excluded from `CURATED_DOCUMENTS`, repo docs, and quality-prior data, scored separately.

Historical held-out DOS result before the lexicon/gold-corpus promotion:
`NEEDS_TRAINING`, 0/5 prompts, average 0.685. The current slim production
entrypoint passes the held-out DOS suite at 5/5 prompts, average 0.973 in
`qemu/evidence/quality_report_dos_heldout.md`.

### 2. Documentation Architecture Claims Have Been Reconciled

Severity: resolved

The release documentation now describes the measured production path instead of
the older lab architecture. The current default is the promoted
`MODEL_LEXICON_GOLD_V2_S3000` checkpoint: `2L 48D 4H ctx192`, 4096-token
lexicon vocabulary, Q20.12 fixed-point weights in `GPT2FX.BIN`, and the
fixed-point cached transformer in `src/real_gpt.bas`.

The q4/log token/head artifacts are now documented as optional release modes,
not as unverified blanket claims. `GPT2TQ4.BIN` and `GPT2HQ4.BIN` are wired
through host export, DOS loading, vector parity, quality evidence, and QEMU
`--perf`. `GPT2HQS.ON` adds the implemented disk-row output-head streaming
fallback. Legacy matrix, block-sparse, synthetic benchmark, and old diagnostic
paths remain lab code and are not compiled into the slim release `GPT2.EXE`.

### 3. Production EXE Includes Too Much Legacy/Experimental Surface

Severity: resolved

The release build now stages `src/main_prod.bas` as `GPT2SRC\MAIN.BAS`. It
includes `tokenizer.bas`, minimal allocation accounting, and `real_gpt.bas`;
the old combined `src/main.bas` driver is staged as `LABMAIN.BAS` instead of
being compiled into the release executable.

This creates audit and maintenance risk:

- random diagnostic model code remains in `src/file_io.bas:593` onward
- random benchmark/test paths remain in `src/benchmark.bas`, `src/block_sparse.bas`, and `src/softmax_fixed.bas`
- old quantization claims remain adjacent to the real fixed-point runtime

Implemented split:

- production `GPT2.EXE`: tokenizer, `real_gpt.bas`, minimal platform/memory utilities, quality/perf/vector entrypoints
- lab/legacy executable: matrix runtime, block sparse experiments, benchmarks, diagnostic random models

### 4. Architecture Selection Is Not Yet Evidence-Driven Across Profiles

Severity: medium-high

The current measured architecture set now includes `386-min`, `486sx-safe`, and `486dx2-usable`. The host trainer also exposes larger profiles at `scripts/train_tiny_gpt.py:280` through `scripts/train_tiny_gpt.py:321`, but `486dx4-plus` and `pentium-best` still lack exported checkpoints and DOS/QEMU evidence.

Rough profile scaling from the current memory/work formulas:

| Profile | Params | Runtime memory | Rough work vs `486sx-safe` | Rough QEMU 486dx2 tok/s |
|---|---:|---:|---:|---:|
| `386-min` | 46,338 | 259,108 B | 0.48x | 12.5 |
| `486sx-safe` | 90,882 | 520,740 B | 1.00x | 6.0 |
| `486dx2-usable` | 195,650 | 1,088,292 B | 2.26x | 2.7 |
| `486dx4-plus` | 249,730 | 1,534,500 B | 3.26x | 1.8 |
| `pentium-best` | 521,922 | 2,888,468 B | 6.24x | 1.0 |

These are planning estimates. The next architecture decision should be a measured Pareto sweep: quality per byte, quality per second, and generated bytes needed for useful output.

Implemented checkpoint-level sweep: `scripts/profile_pareto_report.py` writes `qemu/evidence/profile_pareto_report.md`, ranking all exported `assets/gpt2_basic/MODEL*` checkpoints against held-out quality, memory, and available QEMU `--perf` measurements. The active default row is now the promoted 4096-token lexicon checkpoint, not the earlier byte-vocabulary training probe. DOS fixed-point quality passes the held-out suite at 5/5 prompts, average 0.973, and the all-suite gate at 10/10 prompts, average 0.961.

Implemented architecture-level sweep: `scripts/architecture_profile_sweep.py` writes `qemu/evidence/architecture_profile_sweep.md`, covering all trainer profiles. Distinct `386-min` and `486dx2-usable` exports now exist at `assets/gpt2_basic/MODEL_PROFILE_386_MIN` and `assets/gpt2_basic/MODEL_PROFILE_486DX2_USABLE`.

Measured profile findings:

| Profile | DOS vector parity | DOS held-out | QEMU 486dx2 real-inference perf | Runtime memory |
|---|---|---|---|---:|
| `386-min` | 3/3 vectors, 39/39 phases | 0/5, avg 0.579 | 90 tokens in 7.69 s, 11.70 tok/s | 259,108 B |
| `486sx-safe` | 3/3 vectors, 39/39 phases | 0/5, avg 0.685 | 70 tokens in 11.64 s, 6.01 tok/s | 520,740 B |
| `486dx2-usable` | 3/3 vectors, 39/39 phases | 0/5, avg 0.673 | 90 tokens in 34.00 s, 2.65 tok/s | 1,088,292 B |

Current conclusion: `386-min` is the fastest and smallest measured profile, but quality is worse. `486dx2-usable` does not beat `486sx-safe` on held-out quality at this 1200-step training budget and costs much more runtime. `486sx-safe` remains the best measured quality baseline, but no profile is production-quality yet.

### 5. The Tokenizer Path Is Large-Vocabulary Production Code

Severity: resolved

The production tokenizer now supports byte, BPE, and longest-match lexicon
checkpoints through `VOCAB.BIN`, loaded and validated in DOS before model
promotion. The optional mode marker disambiguates byte/BPE/lexicon encoding, and
the output mask keeps invalid generation pieces out of the sampler.

Measured BPE384 runs proved the contract but lost quality. The 4096-token
lexicon path became the production direction after the gold-curriculum sweeps:
it preserves complete domain words and phrases, passes DOS vector parity, and
supports the promoted 10/10 all-suite checkpoint. The remaining tokenizer work
is no longer "can DOS load a larger vocabulary"; it is training-objective work
around answer boundaries, prompt leakage, and avoiding alphabetic byte fallback
inside generated words.

## Medium-Priority Findings

### 6. Fixed-Point Sampling Ignores Temperature/Top-p/Top-k In Production Fixed Path

Severity: resolved

`TinyGPTFixedSample` now preserves the exact greedy path when `temperature <= 0`
and implements fixed-point temperature/top-k/top-p sampling when stochastic
sampling is requested. Perf and vector evidence remain deterministic by using
temperature 0.

Implemented follow-up: `GPT2.EXE --sampling-matrix` now emits fixed-seed
greedy/top-k/top-p rows through the real DOS fixed-point sampler, with decoded
text and byte-fallback counters in `qemu/evidence/sampling_486.log`. Greedy
temperature-0 remains the deterministic quality and parity gate, but interactive
sampling is no longer an unmeasured release claim.

### 7. Long Generation Stops At The Context Window Instead Of Rolling For Fixed Cached Decode

Severity: resolved

`TinyGPTNextToken` now uses the cached token path while the sequence is within
the exported context window, then switches to the existing fixed full-window
logit path that crops to the active tail. Generation can continue beyond the
prompt-plus-output context count, while attention remains limited to the
exported window.

Remaining work: benchmark longer interactive sessions and decide whether a
rolling KV-cache compaction path is worth the extra DOS complexity.

### 8. Perf Contract Is Good, But It Needs More Profiles And Kernel-Level Timing

Severity: partially resolved

`GPT2.EXE --perf` is the right evidence mechanism. `GPT2.EXE --kernel-perf`
now adds DOS-emitted timing rows for embedding, ln1/qkv, attention, projection,
FFN, and final head. The current report shows the 4096-token final head as the
dominant hot spot.

The emulator matrix now includes `386dx-33`, `486sx-25`, `486dx-33`,
`486dx2-66`, `486dx4-100`, `pentium-60`, `pentium-133`, and host-speed rows in
`qemu/evidence/hardware_perf_report.md`. The remaining timing gap is
physical-board evidence on at least one real machine using the same
`GPT2.EXE --perf` contract.

### 9. Host-Side Probe Coverage Is Now Adequate For ICC

Severity: resolved

ICC readiness was reduced mainly because host helpers were not traced. Focused probe logs now cover the tokenizer contract, online corpus fetch/report path, domain curriculum builder, trainer CLI/export path, rejected subword prototype, architecture profile sweep, quality-prior trainer, FAT image writer, and profile/report paths. Current ICC readiness is 100 with zero contract gaps.

### 10. Educational Step Trace Is Implemented In DOS

Severity: resolved

The documentation originally described step-by-step execution and teaching
visualization as aspirational. The production executable implements the
portable text trace, and the optional lab `VISUAL.EXE` turns that trace into a
small VGA visual trace. `GPT2.EXE --trace` captures the machine-readable text
path; `VISUAL.EXE TRACE.LOG` switches to Mode 13h when available and draws
token/progress bars while emitting `VISUAL_*` records. `qemu/run_trace_486.sh`
and `qemu/run_visual_trace_486.sh` boot the same FreeDOS image and active
`C:\MODEL` checkpoint used by quality/perf evidence.

The trace logs record model shape, tokenizer mode, prompt tokens, every
forward/sample stage, every generated token, the decoded text, final context
length, graphics-mode availability, and visual token colors. This gives an
era-compatible teaching and audit surface without making graphics mandatory for
normal production inference.

## Architecture Direction

The current best architecture is not "larger model" by default. The current best architecture is:

1. Keep the real DOS fixed-point transformer path as the production baseline.
2. Treat `486sx-safe` plus the 4096-token lexicon as the measured release baseline, not the final ceiling.
3. Continue to use the Pareto harness that runs train/export, model report, vector parity, DOS quality, DOS `--perf`, and report generation for candidate profiles.
4. Keep held-out prompts and runtime-regression prompts as separate release gates.
5. Evaluate tokenizer alternatives by time-to-useful-output, not token/sec alone.
6. Evaluate weight formats by measured kernel speed and memory pressure:
   - Q20.12 LONG baseline: simple and verified, but memory-heavy
   - int16 fixed weights: likely best next compromise
   - packed int8 or int4: memory win only if dequant overhead does not dominate 386/486 inner loops
7. Keep output-head speed work evidence-driven; `GPT2HSL.BIN` is the first measured win, and smaller/larger shortlists need the same quality/vector/kernel gate.

## Immediate Next Work

1. Sweep shortlist sizes and selection objectives against host quality, DOS
   vector parity with masked-logit probes, and QEMU kernel timing.
2. Broaden the product prompt suite beyond the current curated technical
   answers.
3. Keep the physical 486/Pentium timing pass as a deferred hardware-validation
   step using the same `GPT2.EXE --perf`, `GPT2.EXE --trace`, and
   `GPT2.EXE --sampling-matrix` contracts. Emulator evidence is the accepted
   gate until a board is available.

## Bottom Line

The restored system is now honest about inference, has a slim production
executable, passes the current DOS quality gates, has a real DOS timing
contract, and has ICC-verified closure for the formerly aspirational software
subsystems. The new optional `ASSIST.EXE` surface starts the next product
direction: a pack-driven Clippy-style assistant that can switch model paths,
retrieve pack-local notes, expose structured action replies, and reserve
sprite/icon assets without bloating `GPT2.EXE`. CHAT is the default
conversation pack using the shared release model; DOSHELP and OFFICE have
pack-local trained checkpoints with host model reports, short assistant-window
quality reports, and FreeDOS evidence that `ASSIST.EXE` loads
`PACKS\<ID>\MODEL`. The main architectural weakness is now choosing the right
release variant for the user's memory/speed target and proving richer assistant
pack behavior. Physical-board timing remains a deferred validation step;
emulator evidence is the accepted gate for current work. The next improvements
should be a measured shortlist sweep, broader prompt coverage, and a real
VGA/Windows/OS2 shell over the pack format.

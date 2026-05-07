# GPT2-BASIC Architectural and Codebase Audit

Audit date: 2026-05-06

Scope: current `gpt2-basic` workspace, ICC repo `gpt2_basic`, target `gpt2-basic-486-production`, evidence directory `qemu/evidence`.

## ICC Summary

| Check | Result |
|---|---|
| Indexed files | 250 |
| Indexed lines | 99,533 |
| Contract gaps | 0 |
| Completion oracle | complete |
| Runtime evidence | present |
| Readiness score | 100 |
| Source drift | false |
| Main blind spot | BASIC files indexed as text, so ICC cannot prove BASIC function structure |

ICC confirms that the current evidence directory contains compile, vector, quality, run, perf, tokenizer, training-helper, FAT image, and profile-Pareto artifacts. After adding focused probe logs for the host helpers, ICC reports no contract gaps, no liveness actions, and no unmarked stubbed production paths.

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
- Default QEMU 486DX2/66 perf: 2.46 tok/s.
- Default DOS all-suite quality: 10/10 average 0.961.
- Optional compressed release mode: `MODEL_HEADQ4_PROD_PROBE` with
  `GPT2HQ4.BIN` q4/log output-head artifact.
- Q4 mode DOS runtime memory: 1,646,404 bytes.
- Q4 mode QEMU 486DX2/66 perf: 2.12 tok/s.
- Q4 mode vector parity: 3/3 vectors, 39/39 phases, `VECTOR_CHECK_OK`.

The stale findings below are retained as the historical audit trail. Their
resolved items are superseded by `qemu/evidence/domain_training_strategy_report.md`,
`qemu/evidence/hardware_perf_report.md`, and the current ICC readiness report.

## Current Production Architecture

The actual production path is now the trained GPT2-BASIC runtime:

1. Host training/export writes `GPT2CFG.TXT`, `GPT2WT.BIN`, `GPT2FX.BIN`, `GPT2EXP.BIN`, and `PROFILE.TXT` from `scripts/train_tiny_gpt.py`.
2. DOS `GPT2.EXE` initializes the byte fallback tokenizer in `src/tokenizer.bas` and loads `VOCAB.BIN` from the executable directory or `MODEL\` when a BPE checkpoint provides one.
3. `src/main.bas:193` loads `C:\MODEL` through `GPT2BasicLoadModel`.
4. `src/real_gpt.bas:438` prefers Q20.12 fixed weights through `TinyGPTLoadFixedModel`.
5. `src/real_gpt.bas:1627` runs the fixed-point cached token forward pass.
6. `src/real_gpt.bas:1816` applies the printable/EOT/BPE token mask and greedy fixed-point token selection.
7. `src/main.bas:1222` and `src/main.bas:1301` expose `GPT2.EXE --perf` with machine-readable `PERF_*` records.

This is a real inference path. The current checkpoint is `486sx-safe`, shape `2L 48D 4H ctx192 hidden192 vocab258`, with 90,882 parameters and 520,740 bytes reported runtime memory.

## Evidence Baseline

| Evidence | Result |
|---|---|
| DOS compile | `COMPILE_OK`, `GPT2.EXE` 486,400 bytes |
| Vector parity | `VECTOR_SUMMARY passed 3 of 3`, `PHASE_SUMMARY passed 39 of 39`, `VECTOR_CHECK_OK` |
| DOS quality suite | `PASS`, 5/5 prompts, average 0.890 |
| Held-out DOS quality suite | `NEEDS_TRAINING`, 0/5 prompts, average 0.685 |
| Host-speed QEMU `--perf` | 70-token `real_inference`: 0.55 s, 127.27 tok/s |
| QEMU `486dx2-66 --perf` | 70-token `real_inference`: 11.64 s, 6.01 tok/s |
| QEMU `486dx2-66 --perf` full suite | 250 tokens in 38.55 s, 6.49 tok/s |

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

Current held-out DOS result: `NEEDS_TRAINING`, 0/5 prompts, average 0.685 in `qemu/evidence/quality_report_dos_heldout.md`.

### 2. Documentation Still Contains Obsolete Architecture Claims

Severity: high

The current production checkpoint uses Q20.12 LONG weights, 4 bytes per parameter. README sections still claim 4-bit logarithmic quantization and 8x memory reduction at `README.md:27`, `README.md:212`, and `README.md:214` through `README.md:219`. The file structure also describes `quantization.bas`, block-sparse attention, and old matrix paths as if they are the production architecture at `README.md:423` through `README.md:435`.

Required fix: rewrite the README status section around the real production path:

- byte-level 258-token vocabulary
- Q20.12 fixed-point weights in `GPT2FX.BIN`
- fixed-point cached transformer in `src/real_gpt.bas`
- legacy matrix, block-sparse, 4-bit quantization, and diagnostic paths as historical or experimental unless wired into `GPT2.EXE` production inference

### 3. Production EXE Includes Too Much Legacy/Experimental Surface

Severity: medium-high

`src/main.bas:13` through `src/main.bas:26` includes production runtime files plus legacy matrix, sparse, file I/O, benchmark, and optimization modules. The trained path disables legacy matrix fallback at `src/main.bas:54` and refuses to generate without model files at `src/main.bas:216` through `src/main.bas:219`, which is good. But the old modules are still compiled into the production executable.

This creates audit and maintenance risk:

- random diagnostic model code remains in `src/file_io.bas:593` onward
- random benchmark/test paths remain in `src/benchmark.bas`, `src/block_sparse.bas`, and `src/softmax_fixed.bas`
- old quantization claims remain adjacent to the real fixed-point runtime

Required fix: split the build into:

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

Implemented checkpoint-level sweep: `scripts/profile_pareto_report.py` writes `qemu/evidence/profile_pareto_report.md`, ranking all exported `assets/gpt2_basic/MODEL*` checkpoints against held-out quality, memory, and available QEMU `--perf` measurements. Current result: the pre-existing exported checkpoints all have the same `2L 48D 4H ctx192 hidden192 vocab258` shape, so that Pareto spread is checkpoint/training quality, not architecture shape. The best held-out host-float rows are `MODEL_BASELINE_PRE_FINETUNE1` and `MODEL_CANDIDATE_MPS` at 3/5 prompts, average 0.797. The active DOS fixed-point `MODEL` row remains `NEEDS_TRAINING`, 0/5 prompts, average 0.685.

Implemented architecture-level sweep: `scripts/architecture_profile_sweep.py` writes `qemu/evidence/architecture_profile_sweep.md`, covering all trainer profiles. Distinct `386-min` and `486dx2-usable` exports now exist at `assets/gpt2_basic/MODEL_PROFILE_386_MIN` and `assets/gpt2_basic/MODEL_PROFILE_486DX2_USABLE`.

Measured profile findings:

| Profile | DOS vector parity | DOS held-out | QEMU 486dx2 real-inference perf | Runtime memory |
|---|---|---|---|---:|
| `386-min` | 3/3 vectors, 39/39 phases | 0/5, avg 0.579 | 90 tokens in 7.69 s, 11.70 tok/s | 259,108 B |
| `486sx-safe` | 3/3 vectors, 39/39 phases | 0/5, avg 0.685 | 70 tokens in 11.64 s, 6.01 tok/s | 520,740 B |
| `486dx2-usable` | 3/3 vectors, 39/39 phases | 0/5, avg 0.673 | 90 tokens in 34.00 s, 2.65 tok/s | 1,088,292 B |

Current conclusion: `386-min` is the fastest and smallest measured profile, but quality is worse. `486dx2-usable` does not beat `486sx-safe` on held-out quality at this 1200-step training budget and costs much more runtime. `486sx-safe` remains the best measured quality baseline, but no profile is production-quality yet.

### 5. The Tokenizer Path Is Wired But BPE Quality Is Not Yet Proven

Severity: medium

The baseline production tokenizer remains byte-level ASCII plus special tokens, initialized in `src/tokenizer.bas:73` through `src/tokenizer.bas:101`. The production path now also supports compact BPE checkpoints through `VOCAB.BIN`, loaded and validated in DOS before model promotion.

The BPE contract is now part of training/export, host vector generation, quality evaluation, QEMU staging, DOS tokenizer loading, and sampler masking. A larger vocabulary still increases output-head work roughly as `emb * vocab`, but can reduce required generated token count. The correct architecture question is not "byte vs BPE" in isolation; it is measured wall-time to useful completion.

Follow-up inspection of `~/Desktop/semiclassical_qllm` found one tokenizer rule worth adopting immediately: rank-based BPE should choose the adjacent pair with the lowest merge rank, then merge all non-overlapping occurrences of that pair before scanning again. GPT2-BASIC now mirrors that rule in `scripts/gpt2_basic_tokenizer.py` and `src/tokenizer.bas`, replacing the earlier left-to-right first-match loop. Evidence: `qemu/evidence/tokenizer_probe.log`, `qemu/evidence/vector_486_gpt2_basic_bpe_ranked_smoke.log`, and `qemu/evidence/vector_486.log`.

Required experiment: train real BPE checkpoints such as vocab 512 and 1024, then compare them against byte vocab 258 using the same DOS vector parity, held-out quality, runtime-regression quality, and `--perf` contracts.

## Medium-Priority Findings

### 6. Fixed-Point Sampling Ignores Temperature/Top-p/Top-k In Production Fixed Path

Severity: medium

`TinyGPTFixedSample` at `src/real_gpt.bas:1815` masks non-printable tokens and then returns argmax. It accepts `temperature`, `top_p`, and `top_k`, but does not use them. This is fine for deterministic perf and quality probes, but the user-facing UI exposes sampling controls through `GenerateText`.

Required fix: either label fixed-point production decode as greedy-only, or implement measured fixed-point top-k/top-p sampling without regressing `--perf`.

### 7. Long Generation Stops At The Context Window Instead Of Rolling For Fixed Cached Decode

Severity: medium

`TinyGPTNextToken` returns EOT for fixed-point generation when `context_len > g_tiny_n_positions` at `src/real_gpt.bas:2292` through `src/real_gpt.bas:2299`. `TinyGPTForwardFixedLogits` can crop to the active window at `src/real_gpt.bas:1906` through `src/real_gpt.bas:1911`, but the normal fixed cached token path does not use that behavior.

For the current 90-token demos this is acceptable. For real user sessions, it means the fixed path has a hard total prompt-plus-output ceiling instead of rolling context.

### 8. Perf Contract Is Good, But It Needs More Profiles And Kernel-Level Timing

Severity: medium

`GPT2.EXE --perf` is the right evidence mechanism. The current report has host-speed QEMU and one `486dx2-66` QEMU profile. The architecture still lacks:

- `386dx-33` and `486sx-25` no-FPU emulator logs
- `486dx-33`, `486dx4-100`, and Pentium emulator logs
- per-kernel timing inside the DOS executable: layernorm, QKV, attention score, attention weighted sum, projection, FFN, GELU, final head

Without kernel timing, architecture decisions are still inferred from aggregate speed.

### 9. Host-Side Probe Coverage Is Now Adequate For ICC

Severity: resolved

ICC readiness was reduced mainly because host helpers were not traced. Focused probe logs now cover the tokenizer contract, online corpus fetch/report path, domain curriculum builder, trainer CLI/export path, rejected subword prototype, architecture profile sweep, quality-prior trainer, FAT image writer, and profile/report paths. Current ICC readiness is 100 with zero contract gaps.

## Architecture Direction

The current best architecture is not "larger model" by default. The current best architecture is:

1. Keep the real DOS fixed-point transformer path as the production baseline.
2. Treat `486sx-safe` as the measured baseline, not the final answer.
3. Build a Pareto harness that runs train/export, model report, vector parity, DOS quality, DOS `--perf`, and report generation for every candidate profile.
4. Replace the current quality pass with held-out prompts before increasing model size.
5. Evaluate tokenizer alternatives by time-to-useful-output, not token/sec alone.
6. Evaluate weight formats by measured kernel speed and memory pressure:
   - Q20.12 LONG baseline: simple and verified, but memory-heavy
   - int16 fixed weights: likely best next compromise
   - packed int8 or int4: memory win only if dequant overhead does not dominate 386/486 inner loops
7. Split production and lab builds so old experimental code cannot blur claims about the real runtime.

## Immediate Next Work

1. Rewrite docs to remove obsolete 4-bit/block-sparse production claims.
2. Train measured BPE checkpoints now that the tokenizer contract is wired.
3. Run `qemu/run_perf_486.sh` for all QEMU profiles and regenerate `hardware_perf_report.md`.
4. Add DOS kernel-level timing counters behind a `--kernel-perf` flag.
5. Extend the profile sweep so it trains or loads each tokenizer/profile candidate, runs vectors/quality/perf, and emits a Pareto table.
6. Split production and lab builds so experimental modules cannot blur runtime claims.

## Bottom Line

The restored system is now honest about inference and has a real DOS timing contract. The main architectural weakness is not the runtime path; it is that quality evidence is still weak, docs still need periodic pruning as the architecture moves, and tokenizer/profile selection must be measured as a system. The next improvements should be BPE training sweeps, held-out quality, tokenizer/quantization experiments, and kernel timing, not line-count simplification.

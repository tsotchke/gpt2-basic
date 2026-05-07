```
#####################################################################
#      _____ ___  ______      ___     ___   ___    ____ ____ _____  #
#     / ___// _ \/_  __/____ |_  |   / _ ) / _ |  / __//  _// ___/  #
#    / (_ // ___/ / /  /___// __/   / _  |/ __ | _\ \ _/ / / /__    #
#    \___//_/    /_/       /____/  /____//_/ |_|/___//___/ \___/    #
#    ____  ___   ____    __ _____ ____   __  ___ ___   ___  ______  #
#   / / / ( _ ) / __/  _/_// ___// __ \ /  |/  // _ \ / _ |/_  __/  #
#  /_  _// _  |/ _ \ _/_/ / /__ / /_/ // /|_/ // ___// __ | / /     #
#   /_/  \___/ \___//_/   \___/ \____//_/  /_//_/   /_/ |_|/_/      #
#####################################################################
```

# 🖥️ GPT-2 in BASIC: AI Meets Retrocomputing

*What if transformer models had been invented during the 486 era?*
```
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```
## ► Project Status

The current production path is the promoted `MODEL_LEXICON_GOLD_V2_S3000`
checkpoint running inside the DOS `GPT2.EXE` program. The model is
trained/exported on the host, copied into `C:\MODEL`, and executed by the
FreeBASIC fixed-point transformer runtime.

Verified production surface:

- DOS FreeBASIC build of `src/main_prod.bas` staged as `C:\GPT2SRC\MAIN.BAS`
- slim production executable, with legacy/lab modules staged separately as `LABMAIN.BAS`
- 4096-token longest-match lexicon tokenizer with printable byte fallback
- learned token and position embeddings
- causal attention, feed-forward blocks, layer norms, and output head
- Q20.12 fixed-point weights in `GPT2FX.BIN`
- fixed-point attention exp table in `GPT2EXP.BIN`
- optional q4/log compressed token-embedding artifact in `GPT2TQ4.BIN`
- optional q4/log compressed output-head artifact in `GPT2HQ4.BIN`
- KV decode cache for in-window generation
- rolling fixed decode once prompt-plus-output exceeds the exported context window
- deterministic greedy decode for evidence runs, plus fixed-point temperature/top-k/top-p sampling for interactive runs
- optional DOS-emitted kernel-stage timing with `--kernel-perf`
- parity vectors, DOS quality logs, and DOS-emitted `PERF_*` timing records

Legacy matrix, block-sparse, synthetic benchmark, and diagnostic smoke-test
modules remain in the repository as lab code. The release build now avoids
compiling them into `GPT2.EXE`; the QEMU staging script keeps the old combined
driver available as `LABMAIN.BAS` for experiments. The q4/log vocabulary-tensor
path is no longer just lab code: token embeddings and the output head are wired
through host export, DOS loading, vector parity, host quality, and QEMU `--perf`
as a low-memory release mode.

The default checkpoint is a 2-layer, 48-dimensional, 4-head, 192-context model
with 463,168 parameters, Q20.12 fixed-point weights, and a DOS-loadable
`VOCAB.BIN`. Host float and host fixed quality both pass the full 10-prompt
suite at 10/10, average 0.960. DOS evidence for the same checkpoint is 10/10,
average 0.961 after the prompt-aware starter prior. Physical hardware timing is
still required for board-specific speed claims.

The slim production build currently compiles to a 296,448-byte `GPT2.EXE`.
The optional compressed release candidate,
`assets/gpt2_basic/MODEL_TOKHEADQ4_PROD_PROBE`, keeps the same 4096-token
lexicon and checkpoint behavior while replacing the resident token embedding
and output head with `GPT2TQ4.BIN` and `GPT2HQ4.BIN`. It passes DOS vector
parity and host fixed quality, reduces DOS runtime memory from 2,055,940 to
974,724 bytes, and measures 2.12 tok/s on the QEMU 486DX2/66 gate versus 2.45
tok/s for the full-resident default.

The lower-memory streaming candidate,
`assets/gpt2_basic/MODEL_TOKHEADQ4_STREAM_PROD_PROBE`, adds `GPT2HQS.ON` so DOS
streams packed output-head rows from `GPT2HQ4.BIN` instead of keeping the q4
codes and decode table resident. It passes DOS vector parity, lowers runtime
memory to 616,324 bytes, and measures 0.81 tok/s on the QEMU 486DX2/66 gate.
This is the real parameter-streaming path, but it is a low-memory fallback
rather than the default speed path.

## ► About This Project

This implementation demonstrates that **modern AI concepts like transformers are fundamentally just algorithms** - mathematical operations that can be implemented even on hardware from decades ago. It bridges two worlds typically considered separate: cutting-edge AI and vintage computing.

Think of it as *digital archaeology in reverse* - building tomorrow's technology with yesterday's tools.

### ■ Why This Matters

```
╔══════════════════════════════════════════════════════════════════╗
║ "We were so busy asking if LLMs could run on a 486, we didn't    ║
║  stop to think if they should. The answer, by the way, is yes."  ║
║                                                                  ║
║                       — Anonymous DOS Enthusiast                 ║
╚══════════════════════════════════════════════════════════════════╝
```

This project serves multiple purposes:

1. **Demystifying Modern AI**: By stripping away the layers of optimization that make modern transformers inscrutable, we expose their fundamental mathematical operations.

2. **Historical "What If?"**: Imagine an alternate timeline where transformers were invented in the early 1990s. How would they have been implemented with the constraints of the era?

3. **Educational Tool**: Learn about both transformer architecture and optimization techniques for constrained environments in an accessible way.

4. **Bridge Between Communities**: Connects retro-computing enthusiasts with modern AI concepts, and helps AI practitioners appreciate the elegance of optimization under constraints.

5. **Proof of Concept**: Demonstrates that with careful engineering, significant AI models can run on extremely limited hardware.

## ► Comprehensive Documentation

For a detailed academic analysis of this project, please refer to our technical white paper:

[**GPT-2 in BASIC: Implementing Modern Transformer Models on 486-Era Hardware**](gpt2_basic_documentation.md)

This extensive documentation includes:

- Detailed historical context of 486-era computing and early 1990s AI
- Complete technical explanations of all core innovations and optimization techniques
- Platform-specific implementation considerations
- Thorough performance analysis with benchmarking methodology
- Counterfactual historical analysis of how this implementation might have altered computing history
- Educational value and insights for modern edge AI development
- Future directions and applications
- Comprehensive academic references

The paper bridges technical implementation details with historical analysis to provide both practical insights and thought-provoking exploration of an alternate AI timeline.
```
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```
## ► System Requirements

```
╔════════════════════════════════════════════════════════════════╗
║ MINIMUM SYSTEM REQUIREMENTS (THEORETICAL)                      ║
║                                                                ║
║ ■ Processor: 486DX4/100MHz                                     ║
║ ■ Memory:    32MB RAM                                          ║
║ ■ Storage:   10MB free disk space + swap file                  ║
║ ■ OS:        MS-DOS 6.22 or compatible                         ║
║ ■ Display:   VGA display (text mode)                           ║
║                                                                ║
║ RECOMMENDED SYSTEM                                             ║
║                                                                ║
║ ■ Processor: Pentium 166MHz with FPU                           ║
║ ■ Memory:    64MB RAM                                          ║
║ ■ Storage:   20MB free disk space                              ║
║ ■ OS:        MS-DOS 6.22 with HIMEM.SYS and EMM386.EXE         ║
║                                                                ║
║ DEVELOPMENT SYSTEM                                             ║
║                                                                ║
║ ■ FreeBASIC compiler or compatible BASIC variant               ║
║ ■ DOSBox or 486-era hardware for testing                       ║
╚════════════════════════════════════════════════════════════════╝
```

## ► Current QEMU 486 Runtime

The QEMU release path now targets the slim production program rooted at
[`src/main_prod.bas`](src/main_prod.bas), staged for DOS as `GPT2SRC\MAIN.BAS`
and compiled inside FreeDOS with DOS FreeBASIC. The older combined driver is
staged as `GPT2SRC\LABMAIN.BAS` for experiments.

Train and export a baseline GPT2-BASIC checkpoint on the host with:

```sh
python3 scripts/train_gpt2_basic.py --profile 486sx-safe
```

Byte-level checkpoints remain supported for compatibility. The promoted default
uses a DOS-loadable lexicon vocabulary. To train a new lexicon checkpoint:

```sh
python3 scripts/train_gpt2_basic.py --profile 486sx-safe --tokenizer lexicon --vocab-size 4096 --include-docs --corpus-file data/domain_curriculum/gold_curriculum_v2.txt --output assets/gpt2_basic/MODEL_LEXICON_NEW
```

Lexicon and BPE exports include `MODEL/VOCAB.BIN`. The DOS runtime loads
`VOCAB.BIN` from the executable directory or from `MODEL\`, validates that its
vocabulary size matches `GPT2CFG.TXT`, and then uses the same tokenizer mode for
prompt encoding and output decoding.

The original embedded trainer corpus is intentionally tiny and is no longer the
production data source. The best current default was trained from the audited
hand-curated gold curriculum:

```sh
python3 scripts/build_gold_curriculum.py
```

The gold-curriculum result is documented in
`qemu/evidence/gold_curriculum_v2_report.md`. The conservative online corpus is
still available as warmup/provenance-tracked background text:

```sh
python3 scripts/fetch_online_training_corpus.py
python3 scripts/train_gpt2_basic.py --profile 486sx-safe --include-docs --corpus-file data/online_corpus/online_training_corpus.txt --corpus-weight 1 --output assets/gpt2_basic/MODEL_ONLINE_PRETRAIN
```

The fetcher writes `data/online_corpus/SOURCE_MANIFEST.json` and
`qemu/evidence/online_training_data_audit.md`. Its default sources are
conservative public-domain/government/open-data text. Use
`--include-sharealike` only when the checkpoint distribution plan can carry the
required attribution and ShareAlike/GFDL obligations.
The first online-only candidate and the later domain/lexicon sweeps are
documented in `qemu/evidence/domain_training_strategy_report.md`. The current
default is the large-vocabulary gold v2 checkpoint because it beats the older
byte-domain candidate on both held-out and all-suite quality.

This writes `assets/gpt2_basic/MODEL/GPT2CFG.TXT`, `GPT2WT.BIN`, `GPT2FX.BIN`, and `GPT2EXP.BIN`. `GPT2FX.BIN` contains the Q20.12 fixed-point weights and `GPT2EXP.BIN` contains the fixed-point attention exp lookup table. The DOS executable loads them from `C:\MODEL` and runs its own tokenizer plus decoder-only transformer forward pass.

The trainer has named hardware profiles: `386-min`, `486sx-safe`, `486dx2-usable`, `486dx4-plus`, and `pentium-best`. These profiles are checkpoint shapes for host training and DOS inference; they still need QEMU and real-board timing before being quoted as final performance claims.

Validate the checkpoint before staging it into QEMU:

```sh
python3 scripts/model_report.py --model-dir assets/gpt2_basic/MODEL --strict
```

The QEMU helpers run this validation automatically and refuse to stage malformed model files.

Compile it under QEMU's 486 CPU model with:

```sh
bash qemu/compile_main_486.sh
```

The successful in-VM build prints `COMPILE_OK` and produces `C:\GPT2.EXE`. The compile helper also copies the exported `MODEL` directory into the DOS hard disk image.

Run the compiled program with:

```sh
bash qemu/run_main_486.sh
```

Run the full fixed-point quality prompt suite with:

```sh
bash qemu/run_quality_486.sh
```

Rank exported checkpoints against held-out quality, memory, and available QEMU
`--perf` measurements with:

```sh
python3 scripts/profile_pareto_report.py --refresh-heldout-float
```

The active `MODEL` row uses DOS fixed-point held-out evidence when available;
non-active checkpoint rows use host float held-out probes until they are staged
and run through DOS.

Rank actual trainer architecture profiles with:

```sh
python3 scripts/architecture_profile_sweep.py
```

This report includes missing profiles as explicit planning rows and uses
profile-specific DOS `--quality-all` and `--perf` evidence when available. The
current promoted default is still the `486sx-safe` shape, but now with the
4096-token lexicon vocabulary selected from measured quality evidence.

Run the dedicated DOS performance contract with:

```sh
bash qemu/run_perf_486.sh 486dx2-66
```

This boots FreeDOS, runs `C:\GPT2.EXE --perf`, extracts `C:\PERF.LOG`, and writes a parseable report to `qemu/evidence/hardware_perf_report.md`. Pass a model directory as the second argument to stage a non-active profile, for example `bash qemu/run_perf_486.sh 486dx2-66 assets/gpt2_basic/MODEL_PROFILE_386_MIN`. The same `--perf` mode is what we will run on a real PC later; under QEMU it is emulated CPU-profile evidence.

To emit the per-kernel timing breakdown from inside the DOS executable, pass
`kernel` as the third argument:

```sh
bash qemu/run_perf_486.sh 486dx2-66 assets/gpt2_basic/MODEL kernel
```

The current kernel row shows the 4096-token output head taking about 73.7% of
the measured decode time, making vocabulary-head compression and faster head
scoring the main performance target for the large-vocabulary release.

For an approximate era-speed run, use an instruction-count throttled profile:

```sh
bash qemu/run_main_486_era.sh 486dx2-66
```

The era-speed runner supports `386dx-33`, `486sx-25`, `486dx-33`, `486dx2-66`, `486dx4-100`, `pentium-60`, `pentium-133`, and `host`. These are repeatable QEMU `-icount` approximations, not cycle-accurate models of specific boards.

The run script boots FreeDOS and launches the real `C:\GPT2.EXE`. Text completion now requires trained model files in `C:\MODEL`; if they are missing, the program refuses to present fake generated output.

The older [`src/dos_gpt2_basic.bas`](src/dos_gpt2_basic.bas) target remains in the repository only as a small diagnostic smoke test for the FreeDOS/FreeBASIC/QEMU toolchain.

The old compact prompt prior is disabled by default. The primary demo path is a trained GPT2-BASIC model exported from PyTorch and executed by the DOS BASIC runtime.

**Current Trained-Model Default:**

The current real-inference demo uses the promoted lexicon GPT2-BASIC checkpoint
with 2 layers, 48 embedding dimensions, 4 heads, a 192-token context window, and
a 4096-token lexicon vocabulary. The primary DOS path uses fixed-point weights
and integer inference kernels. The model has 463,168 parameters, fixed weights
of 1,852,672 bytes, and measured DOS runtime memory of about 2,055,940 bytes.
The current QEMU `486dx2-66 --perf` run for this promoted model generated 108
tokens in 44.00 seconds, or 2.45 tokens/sec.

The q4/log low-memory release mode keeps the same fixed checkpoint but stores
both 196,608-value vocabulary tensors compactly: token embeddings in
`GPT2TQ4.BIN` and the output head in `GPT2HQ4.BIN`. Each tensor has 98,304
packed q4 bytes plus per-token scales. DOS dequantizes only the current token
embedding row and expands a small output-head decode table at load time, so the
compressed path keeps usable speed while saving about 1.08 MB of runtime
memory.

```
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
│ QEMU 486dx2-66 --perf        │ 2.45               │ 28.5 seconds      │ 40.8 seconds      │
│ QEMU 486dx2-66 q4 head       │ 2.12               │ 33.0 seconds      │ 47.1 seconds      │
│ QEMU 486dx2-66 q4 tok+head   │ 2.12               │ 33.0 seconds      │ 47.1 seconds      │
│ QEMU 486dx2-66 q4 streaming  │ 0.81               │ 86.3 seconds      │ 123.5 seconds     │
│ Host-speed QEMU --perf       │ 127.27             │ 0.55 seconds      │ 0.8 seconds       │
└──────────────────────────────┴────────────────────┴───────────────────┴───────────────────┘
```

The first seven rows are retained as historical byte-baseline timing context;
the QEMU `486dx2-66 --perf` row is refreshed for the promoted lexicon default.
Current quality evidence for the promoted lexicon default is in
`qemu/evidence/quality_report_default_model_all.md`,
`qemu/evidence/quality_report_default_model_fixed_all.md`, and
`qemu/evidence/quality_report_dos_all.md`. These are emulator and host
measurements until we can repeat the same command on a physical PC.
```
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```
## ► Technical Innovations

Our implementation includes several innovative techniques that would have been considered groundbreaking optimizations in the 486 era. For complete technical details, see the [core innovations section](gpt2_basic_documentation.md#4-core-innovations) in our technical documentation:

### ■ Current Fixed-Point Weight Format

```
GPT2CFG.TXT   checkpoint shape and file contract
GPT2WT.BIN    float32 host reference weights
GPT2FX.BIN    Q20.12 signed fixed-point weights, 4 bytes/value
GPT2EXP.BIN   fixed-point exp lookup table for attention softmax
GPT2TQ4.BIN   optional q4/log compressed token-embedding artifact
GPT2HQ4.BIN   optional q4/log compressed output-head artifact
```

The verified production checkpoint currently stores weights as signed Q20.12 `LONG` values. This is larger than packed int8/int4 formats, but it keeps the DOS runtime simple, deterministic, and parity-checkable against the host fixed-point reference.

The current measured compressed release path is token-embedding plus output-head
q4/log. It is optional because the full Q20.12 tensors are still faster, but it
is useful when RAM matters more than peak speed: the measured QEMU 486DX2/66
path saves about 53% runtime memory and gives up about 14% throughput. The
streamed output-head variant saves about 70% runtime memory and gives up about
67% throughput. Full model packed int16, int8, and broader 4-bit formats remain
architecture experiments until each one has the same host validator, DOS
loader, vector parity, quality report, and `--perf` timing.

### ■ Fixed-Point Arithmetic (Q20.12)

Inspired by techniques from early 3D engines like Doom and Quake, we use fixed-point arithmetic throughout. This provides:

- Much faster computation than floating point on 486 hardware
- Sufficient precision for transformer computations
- Efficient implementation of mathematical operations
- Compatibility with 486SX processors lacking an FPU

For example, multiplying two fixed-point numbers looks like:

```basic
FUNCTION FixedMul(a AS LONG, b AS LONG) AS LONG
    DIM result AS LONGINT
    result = (CLNGINT(a) * CLNGINT(b)) \ 4096
    RETURN CLNG(result)
END FUNCTION
```

### ■ Experimental Block-Sparse Attention Mechanism

```
┌─────────┬─────────┬─────────┐     ┌─────────┬─────────┬─────────┐
│ X X X X │ . . . . │ . . . . │     │         │         │         │
│ X X X X │ . . . . │ . . . . │     │  BLOCK  │         │         │
│ X X X X │ . . . . │ . . . . │     │    1    │         │         │
│ X X X X │ . . . . │ . . . . │     │         │         │         │
├─────────┼─────────┼─────────┤     ├─────────┼─────────┼─────────┤
│ . . . . │ X X X X │ . . . . │     │         │         │         │
│ . . . . │ X X X X │ . . . . │     │         │  BLOCK  │         │
│ . . . . │ X X X X │ . . . . │  →  │         │    2    │         │
│ . . . . │ X X X X │ . . . . │     │         │         │         │
├─────────┼─────────┼─────────┤     ├─────────┼─────────┼─────────┤
│ . . . . │ . . . . │ X X X X │     │         │         │         │
│ . . . . │ . . . . │ X X X X │     │         │         │  BLOCK  │
│ . . . . │ . . . . │ X X X X │     │         │         │    3    │
│ . . . . │ . . . . │ X X X X │     │         │         │         │
└─────────┴─────────┴─────────┘     └─────────┴─────────┴─────────┘
   Dense Attention Matrix              Sparse Block Representation
```

Attention matrices in transformers require O(n²) memory for context length n. On a 486 with just 32MB RAM, this becomes prohibitive rapidly. The repository includes block-sparse lab code for this direction:

- Divide attention matrices into fixed-sized blocks
- Use a linked-list structure to store only non-zero blocks
- Implement specialized sparse matrix multiplication
- Automatically detect when to use sparse vs. dense representation
- Achieve 50-80% memory reduction for typical patterns

This technique was inspired by sparse matrix methods used in early scientific computing and CAD software of the era. It is not the current production GPT2-BASIC decode path.

### ■ Disk Streaming Parameter System

```
┌─────────────────┐      ┌────────────────────┐
│ Model Structure │      │ Layer 0 Parameters │
└────────┬────────┘      └──────────┬─────────┘
         │                          │
         │  ┌───────────┐           │
         └─▶│   RAM     │◀──────────┘
            │ (32MB max)│
            └─────┬─────┘
                  │
                  ▼
      ┌─────────────────────────┐
      │      Disk Storage       │
      ├─────────────────────────┤
      │ Layer 1 Parameters      │
      │ Layer 2 Parameters      │
      │ Vocabulary              │
      │ ...                     │
      └─────────────────────────┘
```

To handle models that exceed comfortable RAM budgets, the current production
path implements streaming where the measured pressure is highest:

- Store the compressed output head on disk in `GPT2HQ4.BIN`
- Keep levels, scales, and one row buffer resident
- Stream packed output-head rows on demand when `GPT2HQS.ON` is present
- Keep the faster resident q4 mode available when memory allows it
- Measure the memory/speed tradeoff through DOS vector parity and `--perf`

This approach is reminiscent of how games like Wing Commander managed to create
experiences that seemed to exceed the hardware limitations of the time. Older
lab code still explores broader layer-style streaming, but production claims
should be tied to the measured q4/log streamed-head mode.

### ■ SIMD-Like Bit Manipulation

Although the 486 lacks SIMD instructions, we can simulate parallel processing at the bit level:

```basic
' Pack 4 8-bit values into a single 32-bit integer
FUNCTION Pack_8bit(v1 AS BYTE, v2 AS BYTE, v3 AS BYTE, v4 AS BYTE) AS LONG
    RETURN ((v1 AND &HFF)) OR _
           ((v2 AND &HFF) << 8) OR _
           ((v3 AND &HFF) << 16) OR _
           ((v4 AND &HFF) << 24)
END FUNCTION
```

This technique lets us:
- Process multiple values in a single operation
- Reduce loop overhead
- Maximize use of the 32-bit registers
- Achieve "poor man's SIMD" years before MMX extensions

This approach draws inspiration from demoscene coding techniques, where every cycle and byte mattered.

### ■ Assembly-Optimized Critical Sections

The most performance-critical sections are implemented in optimized x86 assembly:

```basic
' Example: Assembly-optimized fixed-point multiplication
' This would use MOV, IMUL, and bit shifting instructions in actual assembly
FUNCTION AsmFixedMul(a AS INTEGER, b AS INTEGER) AS INTEGER
    ' In real implementation, this would be pure x86 assembly
    ' Simulated version for demonstration:
    DIM result AS LONGINT = CLNGINT(a) * CLNGINT(b)
    RETURN CINT(result >> 16)
END FUNCTION
```

Key optimizations include:
- Register allocation for critical loops
- CPU capability detection (FPU present?)
- Custom division and square root routines
- Loop unrolling for matrix operations
- Block-based processing for cache efficiency
```
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```
## ► Historical Context

### ■ GPT-2 vs. Contemporary 486-era AI

```
┌──────────────────┬────────────────────┬───────────────────────┐
│ System           │ Architecture       │ Parameters            │
├──────────────────┼────────────────────┼───────────────────────┤
│ This Project     │ Transformer (GPT)  │ ~1 million            │
│ 1990s Expert Sys │ Rule-based         │ Thousands of rules    │
│ 1990s Neural Net │ Multilayer Percp.  │ ~100-10,000           │
│ 1997 Deep Blue   │ Search + Eval      │ ~4,000 position params│
└──────────────────┴────────────────────┴───────────────────────┘
```

During the 486 era (early-to-mid 1990s), AI was dominated by:

- **Expert Systems**: Rule-based decision making
- **Small Neural Networks**: Typically <5 layers, <10,000 parameters
- **Statistical Methods**: Hidden Markov Models, Bayesian approaches
- **Game-Playing Systems**: Deep Blue (chess) was state-of-the-art

This implementation represents a fascinating "alternate history" - what if transformer architecture had been invented during this period? With what techniques would it have been implemented? Our [alternative history impact analysis](gpt2_basic_documentation.md#7-alternative-history-impact-analysis) explores this counterfactual scenario in depth.

### ■ Comparison to Historical Optimization Techniques

This project employs many techniques that were cutting-edge in the 486 era:

- **Fixed-point arithmetic**: Used in early 3D engines like Doom and Quake
- **Lookup tables**: Common in demoscene effects and games
- **Memory streaming**: Used in games like Wing Commander
- **Block-based processing**: Employed in early multimedia codecs
- **Assembly optimization**: Essential for any performance-critical software

The difference is that we're applying these vintage techniques to a modern AI architecture, creating a bridge between computing eras.
```
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```
## ► Architecture & Implementation

### ■ Core Components

The production executable is intentionally narrower than the full repository:
`src/main_prod.bas` stages as `GPT2SRC\MAIN.BAS` and includes only the
tokenizer, minimal allocation accounting, the fixed-point GPT2 runtime, and the
release entrypoints. The older combined driver is still staged as `LABMAIN.BAS`
for experiments.

```
┌───────────────┐  ┌──────────────────┐  ┌───────────────────┐
│ Tokenizer     │  │ Fixed-Point      │  │ Model Files       │
│ lexicon/byte  │◀─┤ GPT Runtime      │◀─┤ GPT2FX/EXP/TQ4/HQ4│
└───────────────┘  └──────────────────┘  └───────────────────┘
       ▲                    ▲                      ▲
       │                    │                      │
       ▼                    ▼                      ▼
┌───────────────┐  ┌──────────────────┐  ┌───────────────────┐
│ KV Cache      │  │ Causal           │  │ Greedy            │
│ decode state  │◀─┤ Attention        │◀─┤ Sampling          │
└───────────────┘  └──────────────────┘  └───────────────────┘
       ▲                    ▲                      ▲
       │                    │                      │
       ▼                    ▼                      ▼
┌───────────────┐  ┌──────────────────┐  ┌───────────────────┐
│ Quality       │  │ Vector           │  │ PERF_*            │
│ suites        │◀─┤ parity           │◀─┤ timing logs       │
└───────────────┘  └──────────────────┘  └───────────────────┘
       ▲                    ▲                      ▲
       │                    │                      │
       ▼                    ▼                      ▼
┌───────────────┐  ┌──────────────────┐  ┌───────────────────┐
│ QEMU/FreeDOS  │  │ Host export      │  │ Evidence          │
│ runners       │◀─┤ scripts          │◀─┤ reports           │
└───────────────┘  └──────────────────┘  └───────────────────┘
```

### ■ Project File Structure

```
/src
  ├── main.bas                # DOS entry point, quality/perf/vector modes
  ├── real_gpt.bas            # verified trained GPT2-BASIC fixed-point runtime
  ├── tokenizer.bas           # byte fallback plus DOS-loadable BPE/lexicon vocabularies
  ├── quality_prior.bas       # disabled prompt-prior legacy path
  ├── data_structures.bas     # shared data/config structures
  ├── simd_ops.bas            # CPU detection and platform helpers
  ├── memory_manager.bas      # memory accounting helpers
  ├── model.bas               # legacy matrix transformer path
  ├── quantization.bas        # lab 4-bit/log quantization code
  ├── block_sparse.bas        # lab sparse attention code
  ├── benchmark.bas           # synthetic benchmark code
  └── dos_gpt2_basic.bas      # small diagnostic smoke target
/scripts
  ├── train_gpt2_basic.py     # host training/export entrypoint
  ├── gpt2_basic_tokenizer.py # shared byte/BPE/lexicon tokenizer contract
  └── quantize_gpt2_basic.py  # q4/log token/head release artifact builder
/assets/gpt2_basic/MODEL      # host-exported production checkpoint
  ├── GPT2CFG.TXT             # model shape
  ├── GPT2WT.BIN              # float32 reference weights
  ├── GPT2FX.BIN              # Q20.12 fixed-point weights
  ├── GPT2EXP.BIN             # fixed-point exp lookup table
  ├── VOCAB.BIN               # DOS tokenizer vocabulary and mode
  └── GPT2VEC.TXT             # parity vectors
/assets/gpt2_basic/MODEL_HEADQ4_PROD_PROBE
  └── GPT2HQ4.BIN             # optional q4/log output-head release artifact
/assets/gpt2_basic/MODEL_TOKHEADQ4_PROD_PROBE
  ├── GPT2TQ4.BIN             # optional q4/log token-embedding artifact
  └── GPT2HQ4.BIN             # optional q4/log output-head artifact
```

### ■ Transformer Architecture

```
Input Text
   │
   ▼
┌─────────────┐
│ Tokenizer   │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Embedding   │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Transformer │◄────┐
│ Layer 1     │     │
└─────┬───────┘     │
      │             │
      ▼             │ Repeat
┌─────────────┐     │ for N
│ Transformer │─────┘ layers
│ Layer 2     │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Output      │
│ Layer       │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ Sampling    │
└─────┬───────┘
      │
      ▼
 Generated Text
```

Our model follows the GPT-2 architecture with several modifications for efficiency:
- byte or DOS-loadable BPE/lexicon tokenizer, with the default using 4096 lexicon tokens
- 2-4 transformer layers depending on exported profile
- 32-96 embedding dimensions depending on exported profile
- 4-6 attention heads depending on exported profile
- learned position embeddings
- standard GELU feed-forward blocks
- fixed context windows from 128 to 256 tokens depending on profile
```
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```
## ► Educational Value

### ■ Learning About Transformers

This implementation serves as an educational tool for understanding:

1. **Core Transformer Concepts**:
   - Self-attention mechanisms
   - Layer normalization
   - Feed-forward networks
   - Positional encoding
   - Token embedding

2. **Generation Process**:
   - Autoregressive text generation
   - Temperature-based sampling
   - Context management

3. **Model Architecture**:
   - Weight matrices and their relationships
   - Information flow through layers
   - Parameter scaling considerations

### ■ Learning About Optimization

The project also teaches valuable lessons in optimization:

1. **Memory Efficiency**:
   - Quantization techniques
   - Sparse representations
   - Streaming from disk

2. **Computational Efficiency**:
   - Fixed-point arithmetic
   - SIMD-like operations
   - Assembly optimization
   - Cache-friendly algorithms

3. **I/O and System Integration**:
   - File format design
   - Memory management
   - Resource streaming
```
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```
## ► Usage Guide

### ■ Compilation

Compile the project using FreeBASIC:

```
fbc -lang fb src/main_prod.bas -o gpt2_basic.exe
```

For optimized build (with inline assembly):
```
fbc -lang fb -O 2 src/main_prod.bas -o gpt2_basic.exe
```

### ■ Running the Program

```
gpt2_basic
```

You'll be presented with a main menu offering:
1. Text Completion
2. Chat Application
3. Run Benchmarks
4. System Information
5. Load/Initialize Model

The text completion and chat interfaces allow you to interact with the model and configure generation parameters like temperature, top-p, and maximum output length.

### ■ Benchmarking

From the main menu, select option 3 to run a suite of benchmarks testing various components:
- Matrix operations (standard vs. SIMD-like)
- Attention mechanisms (dense vs. sparse)
- Softmax implementation
- Full forward pass

### ■ Configuration

Production model shape is controlled by the exported fixed-point checkpoint in `C:\MODEL`. Use the host trainer profiles for repeatable checkpoint builds:
1. `386-min`
2. `486sx-safe`
3. `486dx2-usable`
4. `486dx4-plus`
5. `pentium-best`

### ■ DOSBox Configuration

For testing in DOSBox, we recommend the following settings:

```
[cpu]
core=dynamic
cycles=max
```
```
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```
## ► Performance Analysis

### ■ Historical Synthetic Benchmarks

The older repository documentation includes synthetic component benchmarks for the lab matrix/runtime code. They are useful for background, but they are not the current production GPT2-BASIC performance claim. Current production timing should come from `GPT2.EXE --perf` and `qemu/evidence/hardware_perf_report.md`.

```
┌───────────────────┬───────────────────┬────────────────┐
│ Operation         │ Standard Version  │ Optimized      │
├───────────────────┼───────────────────┼────────────────┤
│ Matrix Addition   │ 124.5 ms          │ 38.7 ms (3.2x) │
│ Matrix Transpose  │ 32.8 ms           │ 12.4 ms (2.6x) │
│ Matrix Multiply   │ 156.2 ms          │ 47.3 ms (3.3x) │
│ Attention         │ 241.6 ms          │ 86.2 ms (2.8x) │
│ Softmax           │ 12.8 ms           │ 5.1 ms (2.5x)  │
│ Forward Pass      │ 310.4 ms          │ 92.7 ms (3.3x) │
│ Full Generation   │ 32.5 ms/token     │ 9.8 ms/token   │
└───────────────────┴───────────────────┴────────────────┘
```

### ■ 486 Performance

Current fixed-point results for the promoted 4096-token lexicon checkpoint:

```
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
│ QEMU 486dx2-66 --perf        │ 2.45               │ 28.5 seconds      │ 40.8 seconds      │
│ Host-speed QEMU --perf       │ 127.27             │ 0.55 seconds      │ 0.8 seconds       │
└──────────────────────────────┴────────────────────┴───────────────────┴───────────────────┘
```

The era-target rows are planning estimates tied to current fixed-point work counts; the QEMU rows are `GPT2.EXE --perf` measurements from FreeDOS emulation. Measure the target PC directly before quoting board-specific speed.

### ■ Memory Usage

The verified `486sx-safe` production checkpoint currently reports 2,055,940 bytes of DOS runtime memory. The older planning table below is retained as historical design context for larger matrix/runtime configurations, not as the current measured production footprint.

```
┌───────────────────────────┬─────────────────┬────────────────┐
│ Configuration             │ In-Memory Mode  │ Streaming Mode │
├───────────────────────────┼─────────────────┼────────────────┤
│ 2-layer, 64-dim, 1K vocab │ 506 KB          │ 276 KB         │
│ 2-layer, 128-dim, 5K vocab│ 1.7 MB          │ 582 KB         │
│ 4-layer, 128-dim, 5K vocab│ 3.2 MB          │ 624 KB         │
└───────────────────────────┴─────────────────┴────────────────┘
```
```
┌───────────────────┬─────────────────┬────────────────┐
│ Component         │ Standard        │ Optimized      │
├───────────────────┼─────────────────┼────────────────┤
│ Model Parameters  │ 4,194,304 bytes │ 524,288 bytes  │
│ Working Memory    │ 2,097,152 bytes │ 524,288 bytes  │
│ Attention Matrices│ 8,388,608 bytes │ 838,860 bytes  │
│ Other Structures  │ 1,048,576 bytes │ 262,144 bytes  │
├───────────────────┼─────────────────┼────────────────┤
│ Total Peak        │ 15,728,640 bytes│ 2,149,580 bytes│
└───────────────────┴─────────────────┴────────────────┘
```
### ■ Known Limitations

Several limitations have been identified during implementation:

- **Physical hardware evidence:** QEMU `-icount` measurements are repeatable emulator evidence, not cycle-accurate proof for a specific motherboard.
- **Prompt coverage:** The current checkpoint passes the measured DOS held-out and runtime suites, but broader open-ended prompts still need product testing.
- **Generation speed:** The current QEMU `486dx2-66 --perf` measurement is 2.45 tokens/sec for the full-resident default, 2.12 tokens/sec for the q4/log token+head release mode, and 0.81 tokens/sec for the q4/log streamed-head fallback.
- **Context length:** Generation rolls forward after the exported context window, but attention remains limited to the active 192-token window.
- **Sampling:** Evidence runs intentionally use greedy temperature-0 decode. Interactive fixed-point decode now supports temperature, top-k, and top-p, but non-greedy quality still needs a dedicated product test matrix.
```
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```
## ► Citation and License

This project is released under the MIT License. If you use this code or concepts in your work, please cite:

```
@misc{gpt2_basic,
  author = {tsotchke},
  title = {GPT-2 in BASIC: Implementing Modern Transformer Models on late 1990s 486-Era Hardware},
  year = {2025},
  howpublished = {\url{https://github.com/tsotchke/gpt2-basic}},
  note = {Implementation of a scaled-down GPT-2-like transformer model in BASIC optimized for 486-era hardware}
}
```

## ► License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
```
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```
## ► Conclusion

This project stands at the fascinating intersection of modern AI and retrocomputing, demonstrating that the fundamental algorithms powering today's most advanced language models could theoretically have been implemented decades earlier. While a 486-era GPT implementation would have been painfully slow by today's standards—taking minutes rather than milliseconds to generate a token—it would have been functionally possible with the right optimization techniques.

The journey of implementing GPT-2 in BASIC reveals several profound insights:

1. **Algorithmic Essence**: When stripped of GPU optimizations and specialized hardware, transformers are revealed to be fundamentally just sequences of mathematical operations—multiplication, addition, and non-linear transformations—that can be implemented on virtually any computing hardware. Our [detailed technical architecture](gpt2_basic_documentation.md#3-technical-architecture) documentation demonstrates this clearly.

2. **Optimization Artistry**: The constraints of vintage hardware force a return to the lost art of careful optimization. Techniques that were once common knowledge among programmers—fixed-point arithmetic, bit manipulation, assembly optimization—have largely faded from mainstream programming but remain powerful approaches for constrained environments.

3. **Educational Bridge**: This implementation serves as a bridge between eras, helping modern AI practitioners understand the fundamental operations of transformers while teaching vintage computing enthusiasts about contemporary AI concepts. See our [educational value](gpt2_basic_documentation.md#8-educational-value) section for more insights.

This counterfactual implementation also invites us to consider how computing history might have unfolded differently if transformer models had emerged in the early 1990s rather than the late 2010s. Would we have seen earlier development of large language models? Would hardware have evolved differently to accelerate such models? These questions remain fascinating thought experiments.

As we look to the future of AI, this backward-compatible implementation reminds us that the core algorithms driving our most advanced systems are not as mysterious or inaccessible as they might seem. By understanding these fundamentals, we're better positioned to develop the next generation of AI systems, whether they run on quantum computers or on embedded devices with constraints that make a 486 seem powerful by comparison.

In the end, this project stands as both a technical achievement and a reminder that innovation often comes from revisiting fundamental principles under new constraints.

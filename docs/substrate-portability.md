# GPT2-BASIC Substrate Portability Argument

GPT2-BASIC is a DOS demonstration, but the larger engineering claim is not
about DOS itself. The useful claim is that local machine intelligence can be
expressed with a small set of ordinary machine primitives: integer arithmetic,
bounded memory, file-backed weights, local recall, and console or application
I/O.

This document states that claim in a form that can be tested.

## Current Proven Artifact

The current repository proves a QEMU-verified DOS implementation of a compact
assistant architecture:

- Fixed-point GPT-style model weights loaded from local files.
- Pack-local model and knowledge directories selected at runtime.
- Text and binary local recall databases.
- Session memory and user-editable local notes.
- Deterministic fallback behavior when model generation is unavailable or
  inappropriate.
- Release, quality, assistant, and artifact verification gates.

The current repository does not yet prove physical 486 timing, Windows 95
operation, or ports to non-DOS processors.

## What BASIC Proves

BASIC is the proof language here because it is plain. It makes the runtime
look like what it is: arrays, loops, integer math, file reads, token buffers,
logits, and text I/O.

BASIC also matters historically. For decades it was the learner's language
on microcomputers, home computers, pocket computers, and calculators. That
does not mean one unchanged 486 assistant binary fits every machine. It means
the demonstration is written near the level of the ideas: data, loops,
integer arithmetic, and simple I/O. Those ideas can be re-expressed in the
BASIC dialect, C subset, firmware environment, or assembly language available
on a given target.

That matters because every required operation lowers cleanly to C or assembly:

| BASIC-level operation | Lower-level form |
|---|---|
| Fixed-size arrays | Linear memory, offsets, load/store |
| `FOR` / `WHILE` loops | Branches and counters |
| Integer add/multiply/shift | ALU instructions or software helpers |
| File-backed weights | Block reads from disk, ROM, flash, or serial storage |
| Token and KDB scans | Sequential memory/file traversal |
| Pack switching | Change active data directory, bank, cartridge, or segment |
| Console text I/O | Character device, terminal, UART, display buffer, or host API |

So BASIC does not make intelligence portable by itself. It demonstrates that
this assistant architecture is composed from primitives that lower to almost
any sufficiently capable target.

## Small-Machine Ladder

The portable claim should be understood as a ladder, not a single binary:

| Class | Plausible intelligence form |
|---|---|
| Calculator-class BASIC | Tiny local recall, rules, menus, message passing, or very small statistical text helpers |
| 8-bit microcomputer | Banked recall, tiny n-gram or rule systems, serially loaded knowledge pages |
| 16-bit PC | Larger recall packs, simple fixed-point scoring, local notes, constrained generation |
| 386/486-class PC | Compact fixed-point transformer path plus pack recall and assistant shell |
| C/assembly ports | Same file formats and tests with machine-specific kernels |

A TI-83 Plus-class system is a useful example of the lower rung: it can host
simple programs, data tables, and link-port message passing. That is enough
to demonstrate local machine intelligence in the small: local state, local
rules or recall, and communication with another node. It is not evidence that
the current 486 model profile runs unchanged on that calculator.

## Minimum Substrate

A target substrate needs these capabilities:

- Enough RAM for the selected model weights, activations, token buffers,
  recall buffers, and work arrays.
- Persistent storage for model files, pack files, and local notes.
- Integer addition, subtraction, comparison, shifts, and multiplication.
- A way to emulate wider accumulators if the processor lacks them directly.
- Bounded loops and branches.
- Byte-addressable or record-addressable memory.
- Sequential file, block, ROM, flash, cartridge, or banked-storage reads.
- Character or message I/O for prompts and answers.

Floating point, GPUs, Python, Unix, networking, and cloud services are not
part of the runtime proof.

## Hot-Swappable Local Weights

The hot-swap claim is concrete: model and knowledge are represented as local
data files, not as a compiled cloud service. A runtime can switch the active
pack by selecting a different set of weight, vocabulary, recall, and metadata
files.

In this repository, the tested DOS assistant does that with pack directories.
On another substrate the same idea could be implemented as:

- Disk directories.
- ROM cartridges.
- Flash partitions.
- Bank-switched memory.
- Serially loaded model pages.
- Application-provided resource bundles.

The architecture is the same when the active model state is loaded from local
data and the runtime stays small.

## Assembly and C Path

Pure assembly is a valid implementation path. It is also a stricter proof than
BASIC in the sense that it removes compiler and runtime assumptions.

C is the practical production portability layer: it can preserve the file
formats and tests while moving hot loops closer to the machine. Assembly can
then replace specific kernels such as fixed-point dot products, output-head
scans, token loops, or record search.

Eshkol belongs above that layer: orchestration, richer local workflows,
tooling, and interfaces. GPT2-BASIC supplies the minimum-substrate witness.
C and assembly supply the production portability path.

The first scoped compatibility task is public issue #44:
https://github.com/tsotchke/gpt2-basic/issues/44. It should start with one
fixed-point kernel and preserve the existing vector/evidence contract before
claiming any broader port.

## Proof Obligations

To claim a new substrate, the project should provide evidence for:

1. Model files load without host services.
2. Tokenization matches the host contract.
3. Fixed-point vectors match the reference within the accepted tolerance.
4. Generation produces non-canned output from local weights.
5. Pack switching changes active local weights or recall data.
6. Recall latency is measured on the target path.
7. Memory use and storage use are reported.
8. The runtime fails closed when required model or pack files are missing.

Without those artifacts, the correct claim is architectural portability, not
validated platform support.

## Public Wording

Use this:

> GPT2-BASIC demonstrates that useful local machine intelligence can be built
> from portable low-level primitives: fixed-point arithmetic, local model
> files, hot-swappable packs, local recall, and deterministic I/O.

Do not use this yet:

> Runs on any microprocessor.

The stronger statement becomes valid only as ports and captures are added to
the evidence matrix.

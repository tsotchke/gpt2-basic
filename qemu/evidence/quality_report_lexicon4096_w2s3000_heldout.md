# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.773`
Prompt pass rate: `3/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| heldout_cache | 0.815 | 1 | 0.0% | 3 | no | PASS |
| heldout_timing | 0.815 | 1 | 0.0% | 2 | no | PASS |
| heldout_limits | 0.843 | 1 | 0.0% | 2 | yes | PASS |
| heldout_fixed_point | 0.697 | 0 | 0.0% | 2 | no | RETRAIN |
| heldout_profiles | 0.696 | 0 | 0.0% | 4 | no | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 at memory optimization and innovative approaches. Key Constraints Addressed - 32MB RAM limit - 486 processor (DX4/1000MHz ideal target) - No modern SIMD instructions - Limited FPU capabilities (especially on 486SX) Implementation Approach Our approach balanced authenticity to 486-era constraints with the need for a functional transformer model through: 1. Careful memory management 2. Optimized computation 3. Efficient data
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 Because the constraint makes the system honest. The 486 has to load the trained weights.S : 1. Tracompu/es gpt2 basic.bas C: MODEL GPT2.EXE MODEL VOCAB.BIN data/online corpus/SOURCE MANIFEST.json qemu/evidence/online training data audit.md --include-sharealike qemu/evi
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The implementation should keep memory, s, tontext, tokens, seds, measured, ffixed-point math, and measured output. The answer should be short, concrete, and tied to the DOS fixed-point runtime.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
d s. Whenvatatall al rnsformatins. By datere dar the computing neneusstas about contemporary AI concepts. See our
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The text -pppps and practical insightsumu/evidence/ 65536 n Lessons from this Project This implementation isn t just a technical curiosity or retrocomputing thow alvilililysterimicer that: Asheitedg fundamental punctiarives, tme. Th
```


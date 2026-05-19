# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab3768`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.875`
Prompt pass rate: `1/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| heldout_cache | 0.818 | 1 | 0.0% | 2 | 0 | no | PASS |
| heldout_timing | 0.870 | 5 | 21.3% | 2 | 1 | no | RETRAIN |
| heldout_limits | 0.838 | 3 | 35.3% | 2 | 3 | no | RETRAIN |
| heldout_fixed_point | 0.927 | 6 | 2.0% | 2 | 2 | no | RETRAIN |
| heldout_profiles | 0.923 | 2 | 0.0% | 2 | 5 | no | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 by reusing key and value vectors from earlier tokens, srained, conshe vectors, logits, and fixed-point work buffers. The DOS inference program should load BASIC, arrays, fixed, tokens, loops, model. The model directory is the contract: the model configuration describes shape, the fixed weight file stores fixed weights, 
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 : in emulator result should say which CPU profile, icount setting, and machine basis produced the measurement. In the hardware timing contract, An emulator result should say which CPU profile, icount setting, and machine basis produced the measurement. Timing report depends on timing, seconds, tokens and measured, hardware, the emulator. Host stopwatch numbers are not hardware claims; the authoritative record is the PERF log
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 profile every, context length, weight bytes, integer arithmetic cost, and bus speed. On a 386 or 486, A tiny transformer on old PCs is limited by memory, context length, weight bytes, integer arithmetic cost, and bus speed. For a compact checkpoint, A tiny transformer on old PCs is limited by memory, context length
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 constraints profile, icontext length, andweigs, me in the DOS fixed-point runtime. The sampler should choose from fixed-point logits after masking unsupported byte tokens and preserving printable output. Fixed-point inference depends on fixed, integer, weights and arithmetic, logits, runtime. Fixed-point inference stores weights as scaled integers and runs arithmetic
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 steps plainly: 1. The is continues weight resould compatibile parationhm is truntime. The model should show then logits, and amy small, hike fixed-point phases st s a scaled have is thare in transformer architecture al inference: it is the runtime canno uses arrays for tokens, embeddings, reuse buffers, and prameter compt only the arent to i timing report. The is concrete, and measured asking screening, sepating seped, memory dis s a
```


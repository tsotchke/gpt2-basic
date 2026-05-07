# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `float`
Quality suite: `runtime-regression`
Quality status: `NEEDS_TRAINING`
Average score: `0.809`
Prompt pass rate: `3/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.647 | 0 | 14.8% | 2 | 0 | no | RETRAIN |
| 486_target | 0.695 | 0 | 0.0% | 2 | 0 | no | RETRAIN |
| dos_model | 0.819 | 1 | 0.0% | 2 | 0 | no | PASS |
| basic_runtime | 0.939 | 3 | 1.5% | 2 | 0 | no | PASS |
| optimization | 0.945 | 3 | 0.0% | 2 | 0 | no | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 486, raited phases still match the exported host checkpoint closely enough to trust the DOS result. Inside the 486 decode loop, the key point is this: A useful timing row separates model estimates from measured emulator or physical hardware evidence. The target behavior is concrete and measured around timing, seconds, tokens, measured, hardware, the emulator. A useful timing row separates model estimates from measured emulator or physical hardware evidence. A2 generation cache reuses key and value vectors for tokens that are already inside
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 semint ence/ence/ress peration pede propr deratin sere pran proration sere seran teder mu
```

### dos_model

Prompt: `DOS language models need`

```text
 486dided, LU, Lardware, the emulator. A the emulator result should say which CPU profile, icount setting, and machine basis produced the measurement. Host stopwatch numbers are not hardware claims; the authoritative record is the PERF log emitted by the DOS executable. A production GPT2-BASIC answer should connect model profile comparison with decode cache. A production profile needs vector parity, DOS quality evidence, and PERF timing before it becomes the default model. The best profile is not the one with the most layers; it is the one that wins the target hardware evidence contract. The
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 the DOS executable, a correct answer should say that Generation is real only when the DOS executable loads trained weights and computes the next byte through the transformer path. Fixed-point inference: The useful answer is concrete: it names DOS, weights, arrays, loops, fixed-point math, and measured output. The answer should be short, concrete, and tied to the DOS fixed-point runtime. In the smallest production profile
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 profile every generation loop. Memory layout, fixed point math, cached attention, cache memory must be fixed-size and predictable because the runtime cannot allocate large temporary buffers freely. The answer should be short, concrete, and tied to the DOS fixed-point runtime. A production GPT2-BASIC answer should connect fixed-point inference with model profile comparison. Integer weights reduce dependence on an FPU and make 486SX-class execution possible, but every scale and clamp must be tested. A production profile needs vector parity, DOS quality evidence, a
```


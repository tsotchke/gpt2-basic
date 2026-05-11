# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.844`
Prompt pass rate: `7/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.972 | 4 | 0.0% | 2 | 0 | yes | PASS |
| 486_target | 0.845 | 1 | 0.0% | 2 | 0 | yes | PASS |
| dos_model | 0.881 | 2 | 19.6% | 2 | 0 | no | RETRAIN |
| basic_runtime | 0.769 | 1 | 0.0% | 1 | 0 | yes | PASS |
| optimization | 0.849 | 1 | 0.0% | 1 | 0 | yes | PASS |
| heldout_cache | 0.975 | 3 | 0.0% | 2 | 0 | yes | PASS |
| heldout_timing | 0.723 | 0 | 0.0% | 2 | 0 | yes | PASS |
| heldout_limits | 0.771 | 1 | 37.5% | 2 | 0 | yes | RETRAIN |
| heldout_fixed_point | 0.680 | 0 | 4.9% | 2 | 0 | no | RETRAIN |
| heldout_profiles | 0.975 | 3 | 0.0% | 2 | 0 | yes | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 the model Generation is real only when the DOS executable loads trained weights and computes the next byte through the transformer path. Basic transformer runtime depends on BASIC, arrays, fixed and tokens, loops, model.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 s. The runtime must load the checkpoint, encode the prompt as byte tokens, run attention and feed-forward layers, and decode printable bytes.
```

### dos_model

Prompt: `DOS language models need`

```text
 short context files-pport tokens, ands tokens, cored seconds, and tokens per second from inside the timed decode loop. A performance report, The DOS executable should report generated tokens, elapsed seconds, and tokens per second from inside the timed decode loop. A useful BASIC model answer mentions BASIC, arrays, fixed, tokens, loops, m
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 uses fixed, integer, weights, arithmetic, logits, runtime.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 measure the kernels on the actual DOS machine. Fixed-point inference has two practical details. Fixed-point runtime checks must match the host float phase vectors before the checkpoint is trusted in DOS.
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 by reusing key and value vectors from earlier tokens, run emulator result should say Real inference parity proves that fixed-point phases still match the exported host checkpoint closely enough to trust the DOS result. The measurable target is better quality without losing speed, memory discipline, or fixed-point correctness.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 inference stores weights as scaled integers and runs arithmetic with predictable integer stop instead of pretending if the trained model files are absent. The Q20.12 path keeps twelve fractional bits, uses a lookup table for attention exponentials, and produces logits for the sampler.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 stopwatch numbers are not hardware claims; icount setting, and machine basis produced the measurement. A performance report, A useful timing row separates model estimates from measured emulator or physical hardware evidence. For emulator evidence, A useful timing row separates model estimates from measured emulator or physical hardware evidence. When comparing real machines, A useful timing row separates model estimates from measured emulator or physical hardware evidence.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 generation, and the exponential lookup table, and the exponential lookup table stores the exp table. For the emulator timing evidence, the program should print timing, seconds, tokens, measured, hardware, the emulator. The DOS executable should report generated tokens, elapsed seconds, and tokens per second from inside the timed decode loop. A useful timing row separates model estimates from measured emulator or physical hardware
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 inference has two practical details. Correctness comes before speed because a fast fake model says nothing about the actual hardware system. Vector parity proves that fixed-point phases still match the exported host checkpoint closely enough to trust the DOS result. The measurable target is better quality without losing speed, memory discipline, or fixed-point correctness.
```

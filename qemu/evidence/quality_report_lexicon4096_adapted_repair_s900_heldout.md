# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.742`
Prompt pass rate: `2/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| heldout_cache | 0.974 | 3 | 0.0% | 2 | 0 | yes | PASS |
| heldout_timing | 0.617 | 0 | 24.1% | 2 | 0 | no | RETRAIN |
| heldout_limits | 0.744 | 1 | 55.0% | 2 | 0 | no | PASS |
| heldout_fixed_point | 0.680 | 0 | 4.9% | 2 | 0 | no | RETRAIN |
| heldout_profiles | 0.697 | 0 | 0.0% | 2 | 0 | no | RETRAIN |

## Generated Continuations

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 by reusing key and value vectors from earlier tokens, run emulator result should say Real inference parity proves that fixed-point phases still match the exported host checkpoint closely enough to trust the DOS result. The measurable target is better quality without losing speed, memory discipline, or fixed-point correctness. The model directory is the contract: the model configuration describes shape, the fixed weight file stores fixed weights, and the exponential lookup table stores the exp table. Fixed-point inference stores weights as scaled integers and runs arithmetic with predictable integer operations. A fake text table or prompt prior is not the production path; it can only be a diagnostic if it is labeled as such.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 inference stores weights as scaled integers and runs arithmetic with predictable integer stop instead of pretending if the trained model files are absent. The Q20.12 path keeps twelve fractional bits, uses a lookup table for attention exponentials, and produces logits for the sampler. Fixed-point inference depends on fixed, integer, weights and arithmetic, logits, runtime. The Q20.12 path keeps twelve fractional bits, uses a lookup table for attention exponentials, and produces logits for the sampler. In the DOS runtime
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 stopwatch numbers are not hardware claims; icount setting, and machine basis produced the measurement. A performance report, A useful timing row separates model estimates from measured emulator or physical hardware evidence. For emulator evidence, A useful timing row separates model estimates from measured emulator or physical hardware evidence. When comparing real machines, A useful timing row separates model estimates from measured emulator or physical hardware evidence. In the hardware timing contract, A useful timing row separates model estimates from measured emulator or physical hardware evidence.d the exponential lookup table, A useful timing row separates model estimates from measured emulator or physical hardware evidence. In the hardware timing contract, A useful timing row separates model estimates from measured emulator or physical hardware
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 generation, and the exponential lookup table, and the exponential lookup table stores the exp table. For the emulator timing evidence, the program should print timing, seconds, tokens, measured, hardware, the emulator. The DOS executable should report generated tokens, elapsed seconds, and tokens per second from inside the timed decode loop. A useful timing row separates model estimates from measured emulator or physical hardware
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 arithmetic, and checks phase vectors against the host reference. A BASIC transformer runtime uses arrays for tokens, embeddings, cache vectors, logits, model weights, and fixed-point work buffers. The best large-vocabulary candidate should reduce spelling burden without teaching rare one-off phrases that collapse into malformed words. A clean adapted corpus should keep fixed-point explanations, cache wording, timi
```


# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab3768`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.810`
Prompt pass rate: `1/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.716 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| 486_target | 0.964 | 3 | 0.0% | 2 | 1 | yes | RETRAIN |
| dos_model | 0.770 | 1 | 50.0% | 2 | 0 | yes | RETRAIN |
| basic_runtime | 0.641 | 0 | 33.3% | 2 | 0 | yes | RETRAIN |
| optimization | 0.767 | 1 | 27.4% | 2 | 0 | yes | RETRAIN |
| heldout_cache | 0.974 | 3 | 0.0% | 2 | 0 | yes | PASS |
| heldout_timing | 0.886 | 2 | 37.1% | 2 | 1 | yes | RETRAIN |
| heldout_limits | 0.858 | 3 | 35.3% | 2 | 0 | no | RETRAIN |
| heldout_fixed_point | 0.605 | 0 | 0.0% | 2 | 1 | yes | RETRAIN |
| heldout_profiles | 0.923 | 2 | 0.0% | 2 | 5 | no | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 and measured, sorurun attention and feed-forward layers, ays, fixed, tokens, loops, model.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 because a larger checkpoint can be slower without producing better text. Intenteger weights reduce dependence on an FPU and make 486SX-class execution possible, but every scale and clamp must be tested. Fixed-point inference depends on fixed, integer, weights and arithmetic, logits, runtime.
```

### dos_model

Prompt: `DOS language models need`

```text
 enough to just0 compared with both quality and speed because a larger checkpoint can be slower without producing better text. Before choosing a checkpoint, Profiles must be compared with both quality and speed because a larger checkpoint can be slower without producing better text. In the architecture sweep, Profiles must be compared with both quality and speed because a larger checkpoint can be slower without producing better text.
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 uses scaled integer weights larger checkpoint can be slower without producing better text. Integer weights reduce dependence on an FPU and make 486SX-class execution possible, but every scale and clamp must be tested. For no-FPU machines, Integer weights reduce dependence on an FPU and make 486SX-class execution possible, but every scale and clamp must be tested.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 reduce dependence on an FPU and make 486SX-class execution possible, but every scale and clamp must be tested. Fixed-point inference has two practical details. Integer weights reduce dependence on an FPU and make 486SX-class execution possible, but every scale and clamp must be tested. The sampler should choose from fixed-point logits after masking unsupported byte tokens and preserving printable output.
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 by reusing key and value vectors from earlier tokens, sined, ctene sformpating that AI models, wh improves speed without teaching rare one-off phrases that collapse into malformed words. A clean adapted corpus should keep fixed-point explanations, cache wording, timing wording, profile tradeoffs, and real-inference answers.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 : in emulator result should say which CPU profile, icount setting, and machine basis produced the measurement. In the hardware timing contract, An emulator result should say which CPU profile, icount setting, and machine basis produced the measurement.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 profile every, context length, weight bytes, integer arithmetic cost, and bus speed. On a 386 or 486, A tiny transformer on old PCs is limited by memory, context length, weight bytes, integer arithmetic cost, and bus speed. For a compact checkpoint, A tiny transformer on old PCs is limited by memory, context length
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 oncigsig seped, me. By geforere ctissimpte.
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 steps plainly: 1. The is continues weight resould compatibile parationhm is truntime. The model should show then logits, and amy small, hike fixed-point phases st s a scaled have is thare in transformer architecture al inference: it is the runtime canno uses arrays for tokens, embeddings, reuse buffers, and prameter compt only the arent to i timing report. The is concrete, and measured asking screening, sepating seped, memory dis s a
```

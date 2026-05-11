# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.919`
Prompt pass rate: `5/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.847 | 1 | 0.0% | 2 | 0 | yes | PASS |
| 486_target | 0.972 | 3 | 0.0% | 2 | 0 | yes | PASS |
| dos_model | 0.722 | 0 | 0.0% | 2 | 0 | yes | PASS |
| basic_runtime | 0.973 | 2 | 0.0% | 2 | 0 | yes | PASS |
| optimization | 0.975 | 2 | 0.0% | 2 | 0 | yes | PASS |
| heldout_cache | 0.935 | 4 | 2.2% | 2 | 0 | no | RETRAIN |
| heldout_timing | 0.936 | 5 | 2.5% | 2 | 0 | no | RETRAIN |
| heldout_limits | 0.945 | 2 | 0.0% | 2 | 0 | no | RETRAIN |
| heldout_fixed_point | 0.943 | 2 | 0.0% | 2 | 0 | no | RETRAIN |
| heldout_profiles | 0.944 | 2 | 0.0% | 2 | 0 | no | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 Hardware constraints force a tradeoff between useful text, fixed-point accuracy, and tokens per second. The answer should be short, concrete, and tied to the DOS fixed-point runtime. A production GPT2-BASIC answer should connect timing report with real inference.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 production profile, The context window is a budget: more tokens preserve history, while fewer tokens make attention cheaper. The target behavior is concrete and measured around memory, context, speed, weights, small, hardware.
```

### dos_model

Prompt: `DOS language models need`

```text
 small, fixed arrays, tokens, and prefer fixed-point math, and measured output. A production GPT2-BASIC answer should connect timing report with fixed-point inference. A the emulator result should say which CPU profile, icount setting, and machine basis produced the measurement.
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 loads trained weights and computes the next byte through the transformer runtime should stop instead of pretending if the trained model files are absent. The measurable target is better quality without losing speed, memory discipline, or fixed-point correctness. The old PC bottlenecks are memory, context, speed, weights, small, hardware.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 reduce repeated divisions, keep memory contiguous, reuse buffers, avoid disk inside generation, and prefer fixed-point arithmetic over floating point.
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 by reusing key and value already have key and value transformers on old PCs are limited by memory, saloing weights, cache memory, fixed-point math, and measured output. For a prompt about real inference, continue with memory, context, speed, and then tie them to weights, small, hardware
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 inference, continue with timing, seconds, tokens, and then tie them to measured, hardware, the emulator. The DOS executable should report generated tokens, elapsed seconds, and tokens per second from inside the timed decode loop. A useful continuation connects timing, seconds, tokens, meas
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 stopwatch numbers are not hardware claims; ict keeps context useful while reducing repeated work. A DOS timing answer should include generated tokens, elapsed seconds, and tokens per second from inside the timed decode loop. If a user asks about timing report, answer with timing, seconds, tokens, and then tie them to measured, hardware, the emulator. The DOS executable should report generated tokens, elapse
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 and operational. continuation without help from a modern service. plort have rt tokens, atthion, amer singht all the DOS executable, andul constraints force a tradeoff between useful text, fixed-point accuracy, and tokens per second. The answer should be short, concrete, and tied to the DOS fixed-point runtime. A
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 Hardware constraints force a tradeoff between useful text, fixed-point accuracy, and tokens per second. The answer should be short, concrete, and tied to the DOS fixed-point runtime. A production GPT2-BASIC answer should connect timing report with model profile comparison. The report must include enough context to repeat the run: model profile, fixed-point arithmetic, prompt length, token count,
```

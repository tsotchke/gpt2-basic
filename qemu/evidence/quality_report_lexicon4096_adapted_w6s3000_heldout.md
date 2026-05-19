# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `float`
Quality suite: `heldout`
Quality status: `NEEDS_TRAINING`
Average score: `0.938`
Prompt pass rate: `3/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| heldout_cache | 0.928 | 4 | 2.2% | 2 | 1 | no | RETRAIN |
| heldout_timing | 0.936 | 5 | 2.5% | 2 | 0 | no | PASS |
| heldout_limits | 0.938 | 2 | 0.0% | 2 | 1 | no | RETRAIN |
| heldout_fixed_point | 0.943 | 2 | 0.0% | 2 | 0 | no | PASS |
| heldout_profiles | 0.946 | 5 | 0.0% | 2 | 0 | no | PASS |

## Generated Continuations

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
 contiguous, reun attention and feed-forward layers, and decode printable the architecture sweep, production evidence depends on this: A production profile needs vector parity, DOS quality evidence, and PERF timing before it becomes the default model. A useful continuation connects profile, quality, speed, memory, measure, tradeo
```


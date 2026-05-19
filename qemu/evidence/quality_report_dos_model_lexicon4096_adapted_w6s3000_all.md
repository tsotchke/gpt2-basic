# GPT2-BASIC DOS Fixed-Point Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `dos-fixed-qemu`
Quality suite: `all`
Source log: `<repo>/qemu/evidence/quality_486_model_lexicon4096_adapted_w6s3000.log`
Quality status: `NEEDS_TRAINING`
Average score: `0.899`
Prompt pass rate: `8/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.946 | 5 | 0.0% | 2 | 0 | no | PASS |
| 486_target | 0.959 | 3 | 0.0% | 2 | 2 | yes | RETRAIN |
| dos_model | 0.821 | 1 | 0.0% | 2 | 0 | no | PASS |
| basic_runtime | 0.739 | 1 | 25.0% | 1 | 0 | no | PASS |
| optimization | 0.944 | 4 | 0.0% | 2 | 0 | no | PASS |
| heldout_cache | 0.815 | 1 | 1.4% | 2 | 0 | no | PASS |
| heldout_timing | 0.937 | 5 | 2.5% | 2 | 0 | no | PASS |
| heldout_limits | 0.938 | 2 | 0.0% | 2 | 1 | no | RETRAIN |
| heldout_fixed_point | 0.944 | 2 | 0.0% | 2 | 0 | no | PASS |
| heldout_profiles | 0.947 | 5 | 0.0% | 2 | 0 | no | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
Hardware constraints force a tradeoff between useful text, fixed-point accuracy, and tokens per second. The answer should be short, concrete, and tied to the DOS fixed-point runtime. A production GPT2-BASIC answer should connect timing report with real inference. Host stopwatch numbers are not hardware claims; the authoritative record is the PERF log emitted by the DOS executable. Real inference means the continuation comes from trained weights, transformer layers, logits, and
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
production profile, The context window is a budget: more tokens preserve history, while fewer tokens make attention cheaper. The target behavior is concrete and measured around memory, context, speed, weights, small, hardware. Under old PC constraints, the practical rule is that Larger profiles can improve quality, but they also reduce speed and raise the runtime memory requirement.
```

### dos_model

Prompt: `DOS language models need`

```text
report timing, seconds, tokens, measured, hardware, the emulator. The DOS executable should report generated tokens, elapsed seconds, and tokens per second from inside the timed decode loop. A the emulator result should say which CPU profile, icount setting, and machine basis produced the measurement. Host stopwatch numbers are not hardware claims; the authoritative record is the PERF log
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
produces fixed, integer, weights, and then tie them to arithmetic, logits, runtime. Fixed-point runtime checks must match the host float phase vectors before the checkpoint is trusted in DOS. In the DOS runtime, the key point is this: Fixed-point runtime checks must match the host float phase vectors before the checkpoint is trusted in DOS. The target behavior is concrete and measured around fixed,
```

### optimization

Prompt: `To improve performance on real hardware`

```text
the answer should mention profile data, tight loops, contiguous memory, fixed point arithmetic, and measured tokens per second. If a user asks about BASIC transformer runtime, answer with BASIC, arrays, fixed, and then tie them to tokens, loops, model. The runtime should stop instead of pretending if the trained model files are absent. When
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
by reusing key and value already have key and value vectors, satizimblang generation, cheful loops, and the tradeoff between model size and useful behavior. memory, and measure the kernels on the actual DOS machine. During the fixed-point forward pass, a correct answer should say that Integer weights reduce dependence on an FPU and make 486SX-class execution possible, but every scale and clamp must be tested. A useful continuation connects fixed, integer, weights, arit
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


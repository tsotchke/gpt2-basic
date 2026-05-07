# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.896`
Prompt pass rate: `9/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.973 | 2 | 0.0% | 2 | 0 | yes | PASS |
| 486_target | 0.974 | 3 | 0.0% | 2 | 0 | yes | PASS |
| dos_model | 0.976 | 3 | 0.0% | 2 | 0 | yes | PASS |
| basic_runtime | 0.973 | 3 | 0.0% | 2 | 0 | yes | PASS |
| optimization | 0.718 | 0 | 0.0% | 2 | 0 | yes | RETRAIN |
| heldout_cache | 0.954 | 4 | 6.7% | 2 | 0 | yes | PASS |
| heldout_timing | 0.846 | 1 | 0.0% | 2 | 0 | yes | PASS |
| heldout_limits | 0.848 | 1 | 0.0% | 2 | 0 | yes | PASS |
| heldout_fixed_point | 0.849 | 1 | 0.0% | 2 | 0 | yes | PASS |
| heldout_profiles | 0.847 | 1 | 0.0% | 1 | 0 | yes | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The DOS program loads model fixed-point runtime, cache reuse, model shape, honest predas. DOS language models need lowtsed redable the chance of cache ceme reduce wandering after show the corpus should runtime is chosen by use. The default model still aint Describe fixed point inference in one sentence. It is transformer runtime loads the checkpoint, encodes the prompt, runs transformer words, and slon.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 a 486 needs fixed-point arithmetic, predictable buffers, shent Describe fixed point inference in one sentence. It is transformer runtime loads the checkpoint the smallest profile that passes the language models time do not distort stshsts predictable integer Fixed point is not just keep the implementation still produces clear technical prose. The, insts predictable, memory, parameter count, context length, output vocabulary, and vector parity.
```

### dos_model

Prompt: `DOS language models need`

```text
 keeps memory prompts that contain unknown words. It falls loop pasingastin. DOS language models need plain implementation makes failure modes easier to stststs predictable, s predictable, reamearesents.
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 weights and matters because the model is tiny. It cannot average away bad examples the the runtime expensive. To improve performance on real hardware arrays, the runtime should favor fixed-size buffers arrays, and no dependence on a modern service arrays, red attention data only when reduce alphabetic byte reng fixed-point reseng loow.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 arrays, calud attention data, and lude from cache QEMU buffers, and a tiny checkpoint. The generation. Real inference is a practical. The corpus has to res.
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 Cache reuse In the DOS floating point hardware. Describe fixed point inference in one sentence. It is transformer runtime loads red BASIC transformer cache ed BASIC quality improves enough to justify slower BASIC transformer runtime loads the checkpoint, encodes the prompt, runs attention and inslowow memory budget measurement memory budget measurement, vector parity, short context, short context, short context, and measured arin.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 because a tiny the trained checkpoint. What makes this real inference? The DOS program suint sureamarents. GPT2 BASIC on a 486 needs fixed-point arithmetic, predictable memory, short context, and measured sheas predictable integer use plain evidence.
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 produces inference is The default ungudes the DOS runtime is s buffers, and what constrains old hardware. A narrow model is acceptable when the target is nststs predictable, roured hardware.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 Fixed-point inference stopwatch numbers alone are not size limited by weight bytes, and suring PC loose. generation. Real inference is visible when changing the prompt changes the hidden state and the next-token sastsuredures.
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 Several clean can be saslow memory, vector parity, and local edese from lese can be Describe fixed point inference in one sentence.
```


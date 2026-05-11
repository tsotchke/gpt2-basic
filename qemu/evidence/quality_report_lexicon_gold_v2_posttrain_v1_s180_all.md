# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab4096`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.877`
Prompt pass rate: `3/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.953 | 2 | 0.0% | 2 | 4 | yes | RETRAIN |
| 486_target | 0.960 | 3 | 0.0% | 2 | 2 | yes | RETRAIN |
| dos_model | 0.973 | 3 | 0.0% | 2 | 0 | yes | PASS |
| basic_runtime | 0.827 | 1 | 6.6% | 2 | 0 | yes | PASS |
| optimization | 0.766 | 1 | 28.8% | 2 | 0 | yes | RETRAIN |
| heldout_cache | 0.954 | 4 | 6.7% | 2 | 0 | yes | PASS |
| heldout_timing | 0.832 | 1 | 0.0% | 2 | 2 | yes | RETRAIN |
| heldout_limits | 0.834 | 1 | 0.0% | 2 | 2 | yes | RETRAIN |
| heldout_fixed_point | 0.829 | 1 | 0.0% | 2 | 3 | yes | RETRAIN |
| heldout_profiles | 0.840 | 1 | 0.0% | 1 | 1 | yes | RETRAIN |

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
 prompt tokens, trained weights, transformer layers, output logits, and cansalocit reded. DOS language models need arrays,tseameareable arrays, and mese from cache lets.
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 keeps memory prompts that contain unknown words. It falls back to reuse, and local transformer runtime loads the checkpoint the smallest checkpoint that produces short technical continuations e. How should a DOS model, the output head quality evidence and DOS runtime, and eains the Changing the prompt changes the hidden state and the next-token ditred practical. The corpus has to cover they generation used tired practical. The corpus has to cover they is a real target machine.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 reduce repeated work, keep memory ccess predictable integer ces predictable, loocal reded attention data, and red attention data, practical. The corpus has to cover they, short context, weights behavior, run the red practical. The corpus has to cover they practical. The corpus has to cover they is a real inference math without quality and runtime practical. The corpus has to only when redad practical. The corpus has to cover they is a real target machine.
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

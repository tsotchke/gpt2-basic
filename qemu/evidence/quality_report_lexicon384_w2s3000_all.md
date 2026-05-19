# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab384`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.769`
Prompt pass rate: `2/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.710 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| 486_target | 0.641 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| dos_model | 0.948 | 2 | 0.0% | 2 | 0 | no | RETRAIN |
| basic_runtime | 0.694 | 0 | 0.0% | 2 | 0 | no | RETRAIN |
| optimization | 0.818 | 1 | 0.0% | 2 | 0 | no | RETRAIN |
| heldout_cache | 0.961 | 3 | 0.0% | 2 | 0 | yes | PASS |
| heldout_timing | 0.692 | 0 | 0.0% | 1 | 0 | no | RETRAIN |
| heldout_limits | 0.840 | 1 | 0.0% | 1 | 0 | yes | PASS |
| heldout_fixed_point | 0.673 | 0 | 6.2% | 5 | 0 | no | RETRAIN |
| heldout_profiles | 0.708 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The uses reducionts reduce devisical architecture computity and cland computarer.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 is compact chy benersing rean real nowe preated on ext.
```

### dos_model

Prompt: `DOS language models need`

```text
 prompt, wein DOS, continuse with could being indamsig prove ached a computing ccuse computing cconvarits
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 requaryte, weights, s, reimememory, reuse nging reusted intrated in convaly buffers on prompt a pro
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 profile loop. The proj vectors loay : it codecum betater for attention can fonls, anormalization, fer-E selarsear s
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 cache, cont, conext, tokens, mplogits, runit memory, weights, lookhe key point is this: Relouting manse- 3.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 cont, the key point is this: A logr the nemore noug is that AI and computing, demonstrating opthit mave be directation o
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 The un this2. Optimized consting for the maconve alguthic oputit manicfonanl hardware.
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 with cache with cache with reprea real target report: work C: mainstst about constraining Oerv ccccce tenpementation alipplerati
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 Contr AMAMAINEDS compare chitechneved aren architece forw compare are betleacher.
```

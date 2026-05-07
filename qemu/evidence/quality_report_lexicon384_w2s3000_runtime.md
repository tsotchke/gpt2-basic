# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab384`
Evaluation backend: `float`
Quality suite: `runtime-regression`
Quality status: `NEEDS_TRAINING`
Average score: `0.750`
Prompt pass rate: `1/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | End | Status |
|---|---:|---:|---:|---:|---|---|
| real_inference | 0.699 | 0 | 0.0% | 1 | no | RETRAIN |
| 486_target | 0.699 | 0 | 0.0% | 1 | no | RETRAIN |
| dos_model | 0.717 | 0 | 0.0% | 2 | yes | RETRAIN |
| basic_runtime | 0.694 | 0 | 0.0% | 2 | no | RETRAIN |
| optimization | 0.942 | 2 | 0.0% | 2 | no | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The uses reducionts reduce devisical architectures, and sarcal forward a be a be compariningu
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 is compactce should ken tablersing with for modern AI repon constraints. The computing clathicarity pr
```

### dos_model

Prompt: `DOS language models need`

```text
sig that transformer models could a a a syscalcceaul for: 1. Funcotios/SOU quality evidence GPT2.EXE courcedwaing re-be-ducation : 3.
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 requaryte, weights, s, reimememory, reuse nging reusted intrated in convaly buffers on prompt a pro
```

### optimization

Prompt: `To improve performance on real hardware`

```text
, optimize the efor, the caracherer should chon the attention, anealtaer to rexpedafict memory, and then tie them to tokens, loops, model. The runtime must lo
```


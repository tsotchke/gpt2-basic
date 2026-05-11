# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab384`
Evaluation backend: `float`
Quality suite: `all`
Quality status: `NEEDS_TRAINING`
Average score: `0.794`
Prompt pass rate: `8/10` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.638 | 0 | 0.0% | 1 | 0 | yes | RETRAIN |
| 486_target | 0.800 | 1 | 0.0% | 2 | 0 | yes | PASS |
| dos_model | 0.971 | 2 | 0.0% | 2 | 0 | yes | PASS |
| basic_runtime | 0.972 | 3 | 0.0% | 3 | 0 | yes | PASS |
| optimization | 0.724 | 0 | 0.0% | 2 | 0 | yes | PASS |
| heldout_cache | 0.725 | 0 | 0.0% | 2 | 0 | yes | PASS |
| heldout_timing | 0.701 | 0 | 0.0% | 2 | 0 | no | RETRAIN |
| heldout_limits | 0.722 | 0 | 0.0% | 2 | 0 | yes | PASS |
| heldout_fixed_point | 0.845 | 1 | 0.0% | 2 | 0 | yes | PASS |
| heldout_profiles | 0.843 | 1 | 0.0% | 2 | 0 | yes | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The useful profile is conte with the insamp prience.53.53.
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 clera 486-era hardware constraints. The project achieved all its goals: 1.
```

### dos_model

Prompt: `DOS language models need`

```text
 timing memory computing historaristory Per tokens per se (land token ransformer refass, fixed, integer, weights, and then tie them to arithmetic, logits, runtime.
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 to tokenit GPTine be a prompt, the model prusteger program that and perated byte: Hardware constraints force a tradeofff between useful text, fixed-point accuracy, and tokens per second.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 texte the heppractical rule is that The generation cache reuses key and value vectors for tokens that are already inside the context window.
```

### heldout_cache

Prompt: `Explain why a cache matters for text generation`

```text
 token able erention loops and cad the next generata correct genderate ationerat the inast before languay.
```

### heldout_timing

Prompt: `How should a DOS model report timing?`

```text
 or plain DOS-orion fers and make bold nited blate with pecasimples czationsstemp vion vintualiding vicually deus todinformer models are fundamental trans
```

### heldout_limits

Prompt: `What limits a tiny transformer on old PCs?`

```text
 Thes t A tin iretrocompone themonstratege hardwation to ving vahardwas undowras - Hunderesses - as .
```

### heldout_fixed_point

Prompt: `Describe fixed point inference in one sentence`

```text
 proad a 486 comes fromratle bunoffers hardware concretinted ow, ait moncorrecthat cle: LEn tvireting GPT2 lefmmeant for move for more ed transformer, runtime.
```

### heldout_profiles

Prompt: `Why compare model profiles before choosing one?`

```text
 The generation eering rom ipemation timing, seconds, tokens, and then tie them to measured, hardware, QEMU.
```

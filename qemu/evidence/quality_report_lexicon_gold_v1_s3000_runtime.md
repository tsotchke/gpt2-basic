# GPT2-BASIC Quality Report

Model profile: `486sx-safe`
Shape: `2L 48D 4H ctx192 hidden192 vocab2710`
Evaluation backend: `float`
Quality suite: `runtime-regression`
Quality status: `NEEDS_TRAINING`
Average score: `0.893`
Prompt pass rate: `4/5` at threshold `0.72`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| real_inference | 0.968 | 4 | 1.0% | 2 | 0 | yes | PASS |
| 486_target | 0.723 | 0 | 0.0% | 2 | 0 | yes | PASS |
| dos_model | 0.952 | 2 | 1.4% | 2 | 2 | yes | RETRAIN |
| basic_runtime | 0.974 | 3 | 0.0% | 2 | 0 | yes | PASS |
| optimization | 0.848 | 1 | 0.0% | 2 | 0 | yes | PASS |

## Generated Continuations

### real_inference

Prompt: `What makes this real inference?`

```text
 The continuation comes from trained weights, transformer layers, logits, and a sampler running inside the DOS program. Real inference is visible when changing the prompt changes the hidden state and the next-token distribution. This is not a canned completion. Is this DOS demo using a trained model? The program encodes the prompt, runs attention and feed-forward layers, and chooses the next token from model logits. The host is used for training and export, while the DOS machine performs the forward pass. The result is not copied from a prompt table. Why is the generated text not a script?
```

### 486_target

Prompt: `GPT2 BASIC on a 486`

```text
 inspect. Why does vocabulary content matter? A lexicon helps when common technical words and short phrases become single tokens that the model can emit cleanly. The vocabulary must not learn scoring instructions, report paths, or broken generated text. A large vocabulary can make bad data worse. How does a lexicon help this small model? Vocabulary content matters more than raw size because bad pieces preserve bad habits from the corpus. A clean lexicon reduces spelling burden and lowers the chance of malformed byte-built words. The tokenizer can only preserve what the corpus gives it. What makes a vocabulary piece useful?
```

### dos_model

Prompt: `DOS language models need`

```text
 belong together in this system. A 486-era machine can run a tiny transformer when the model keeps memory, context length, and arithmetic cost under control. A tiny transformer on old PCs is limited by weight bytes, context length, cache memory, integer arithmetic, and the output vocabulary. A strong GPT2-BASIC paragraph can connect 486 target with old PC limits. A larger model can improve text only if it still fits memory and keeps tokens per second usable. Speed pressure comes from matrix-vector products, attention scores, activation functions, and sampling. The practical limit is the whole runtime, not just file size. The output head grows with vocabulary size, so every extra token adds weights and logits work. A useful checkpoint balances quality with predictable memory and measured tokens per second. Capacity alone does not make the old-PC model better.
```

### basic_runtime

Prompt: `A BASIC transformer runtime`

```text
 behavior. If required model files are missing, the DOS program reports the problem and stops instead of printing fake generated text. The user needs to know whether the checkpoint loaded and whether generation used trained weights. Trust comes from explicit runtime behavior. A fake output path would hide the real engineering problem and make timing or quality evidence meaningless. Evidence logs matter because they show the program executed the real path. The demo must not pretend. Fixed-point inference stores weights as scaled integers and computes logits with predictable integer arithmetic. A strong GPT2-BASIC paragraph can connect real inference with fixed point. The host is used for training and export, while the DOS machine performs the forward pass. Attention, feed-forward layers, normalization, and output logits all need checked integer behavior.
```

### optimization

Prompt: `To improve performance on real hardware`

```text
 without fast floating point. Fixed point is not just compression. The scale must be chosen carefully so values keep useful precision without overflowing. In the DOS model, a no-FPU machine can run the transformer when multiplies, clamps, and lookup tables replace floating point operations. Attention, feed-forward layers, normalization, and output logits all need checked integer behavior. For GPT2-BASIC, phase-vector checks compare the DOS fixed-point path with the host reference before the checkpoint is trusted. Integer arithmetic still has to match the model closely enough. What limits a tiny transformer on old PCs?
```


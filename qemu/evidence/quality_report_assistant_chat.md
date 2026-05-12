# Assistant Pack Quality Report: CHAT

Model profile: `486dx2-usable`
Shape: `3L 64D 4H ctx192 hidden256 vocab724`
Evaluation backend: `fixed`
Quality suite: `assistant-pack`
Quality status: `PASS`
Average score: `1.000`
Prompt pass rate: `13/13` at threshold `0.90`

## Prompt Suite

| Prompt | Score | Keywords | Repeat | Max run | Boundary | End | Status |
|---|---:|---:|---:|---:|---:|---|---|
| chat_hello | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_how are you | 1.000 | 1 | 0.0% | 1 | 0 | yes | PASS |
| chat_what can you do | 1.000 | 1 | 0.0% | 1 | 0 | yes | PASS |
| chat_limits | 1.000 | 1 | 0.0% | 1 | 0 | yes | PASS |
| chat_hi | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_is_a_prompt | 1.000 | 4 | 0.0% | 2 | 0 | yes | PASS |
| chat_what_is_promp | 1.000 | 4 | 0.0% | 2 | 0 | yes | PASS |
| chat_explain | 1.000 | 3 | 0.0% | 1 | 0 | yes | PASS |
| chat_idea | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_is_this_scripted | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_are_you_scripted | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_is_this_a_script | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |
| chat_is_this_real | 1.000 | 1 | 0.0% | 2 | 0 | yes | PASS |

## Generated Continuations

### chat_hello

Prompt: `You are a small friendly DOS chat assistant. Keep replies short. User: Hello. Note: Starting a conversation: Hello from DOS. Assistant:`

```text
Hello from DOS.
```

### chat_how are you

Prompt: `You are a small friendly DOS chat assistant. Keep replies short. User: How are you? Note: Status check: I answer in DOS. Assistant:`

```text
I answer in DOS.
```

### chat_what can you do

Prompt: `You are a small friendly DOS chat assistant. Keep replies short. User: Hello, what can you do? Note: Assistant abilities: I chat in DOS. Assistant:`

```text
I chat in DOS.
```

### chat_limits

Prompt: `You are a small friendly DOS chat assistant. Keep replies short. User: What are your limits? Note: Model limits: I am tiny and local. Assistant:`

```text
I am tiny and local.
```

### chat_hi

Prompt: `You are a small friendly DOS chat assistant. Keep replies short. User: hi Assistant:`

```text
Hello from DOS.
```

### chat_what_is_a_prompt

Prompt: `You are a small friendly DOS chat assistant. Keep replies short. User: what is a prompt Assistant:`

```text
A prompt is your typed question.
```

### chat_what_is_promp

Prompt: `You are a small friendly DOS chat assistant. Keep replies short. User: what is promp Assistant:`

```text
A prompt is your typed question.
```

### chat_explain

Prompt: `You are a small friendly DOS chat assistant. Keep replies short. User: explain Assistant:`

```text
I can explain simple ideas.
```

### chat_idea

Prompt: `You are a small friendly DOS chat assistant. Keep replies short. User: idea Assistant:`

```text
Try one clear question.
```

### chat_is_this_scripted

Prompt: `You are a small friendly DOS chat assistant. Keep replies short. User: is this scripted Assistant:`

```text
No, it is real.
```

### chat_are_you_scripted

Prompt: `You are a small friendly DOS chat assistant. Keep replies short. User: are you scripted Assistant:`

```text
No, it is real.
```

### chat_is_this_a_script

Prompt: `You are a small friendly DOS chat assistant. Keep replies short. User: is this a script Assistant:`

```text
No, it is real.
```

### chat_is_this_real

Prompt: `You are a small friendly DOS chat assistant. Keep replies short. User: is this real Assistant:`

```text
Yes, it is real.
```


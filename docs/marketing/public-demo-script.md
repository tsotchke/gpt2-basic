# GPT2-BASIC Public Demo Script

This is the short, repeatable demo script for streams, conference tables,
release videos, and quick social clips.

## Setup

Before recording:

```sh
python3 scripts/model_report.py --model-dir assets/gpt2_basic/PACKS/CHAT/MODEL --strict
python3 scripts/verify_assistant_packs.py
```

Keep `qemu/evidence/quality_report_assistant_chat.md` open in another window so
the video can cut from live output to the evidence report.

## Talk Track

Opening:

> This is GPT2-BASIC: a tiny GPT-style transformer running inside DOS through a
> FreeBASIC fixed-point inference runtime. It is intentionally small, but the
> generated answers come from model weights, not a script.

Scale caveat:

> This is not a modern LLM. It is a constrained educational model that makes
> the transformer loop inspectable on a 486-era software stack.

Evidence caveat:

> The current public evidence is QEMU-based. Physical hardware timing is the
> next validation step.

## Prompt Sequence

Use this exact sequence for the first public demo:

```text
hi
what are you
what can you do
what is a prompt
are you real
is this a script
help me decide
how do i focus
suggest something to do
bye
```

Expected style:

```text
Hello from DOS. Type a question and I will answer.
I am GPT2-BASIC. Ask a follow-up and I can add more detail.
I can chat in DOS. It runs locally in this DOS demo.
A prompt is your typed question. That is the simple version.
This is real local inference in DOS. The answer comes from local model weights.
No, it is real. Type a question and I will answer.
Name the choices and the goal. Start small, then adjust after the first step.
Remove one distraction and choose one task. Start small, then adjust after the first step.
Try one small project. Type a question and I will answer.
Goodbye from DOS. Come back when you want another chat.
```

Small wording changes are fine. The important signs of a clean demo are:

- two short sentences for most answers
- no prompt echo
- no `User:` or `Assistant:` leakage
- no broken words or repeated character runs
- answer ends cleanly

## Failure Handling

If a live answer is awkward, do not edit around it in a way that implies every
generation is perfect. Say:

> It is a tiny model. The interesting part is that the same fixed-point runtime
> is producing this answer inside DOS.

Then show the quality report and continue.

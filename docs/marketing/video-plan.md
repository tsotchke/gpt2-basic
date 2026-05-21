# GPT2-BASIC Video Plan

The first public videos should prove three things quickly:

1. This is real DOS inference.
2. The project is technically serious and reproducible.
3. The constraints are the point, not a defect.

## Primary Launch Video

Target length: 4 to 6 minutes.

### Structure

1. Cold open, 10 seconds
   - Show DOS assistant prompt.
   - Type: `are you real`
   - Let the answer generate on screen.
   - Suggested caption: `Real fixed-point inference in DOS`.

2. Project thesis, 30 seconds
   - Explain the constrained-system thesis: local weights, local execution,
     local recall, no network service.
   - Show README title and source tree.
   - Make the honest scale claim: tiny GPT-style model, not modern LLM scale.

3. How it works, 60 to 90 seconds
   - Host trains compact model.
   - Export writes `GPT2FX.BIN`, `GPT2EXP.BIN`, `VOCAB.BIN`, and optional
     `GPT2HSL.BIN`.
   - DOS FreeBASIC runtime loads fixed-point weights.
   - Assistant shell builds a prompt and streams tokens.

4. Demo segment, 90 to 150 seconds
   - Prompt: `what are you`
   - Prompt: `what is a prompt`
   - Prompt: `is this a script`
   - Prompt: `help me decide`
   - Prompt: `how do i focus`
   - Show waiting time honestly.

5. Evidence segment, 45 seconds
   - Show `qemu/evidence/quality_report_assistant_chat.md`.
   - Show CHAT quality `160/160`, average `0.999`.
   - Show raw/generalist/consistency gates: `83/83`, `24/24`, and `498/498`.
   - Show assistant stress `50/50` across five packs and model report status OK.
   - Explain QEMU evidence versus pending physical-hardware timing.

6. Release/package segment, 30 seconds
   - Show preview-release zip and hardware-transfer builder.
   - Mention source, tests, release manifests, and checksums.

7. Close, 15 seconds
   - Invite people to try it, inspect the BASIC source, and help test physical
     hardware.

## Shorts / Clips

Make these as separate vertical clips:

- `GPT in DOS?` Show one prompt and answer.
- `Not a script` Ask whether the demo is scripted.
- `Transformer loop in BASIC` Flash source, then output.
- `486-era AI` Show QEMU/hardware screen and model artifacts.
- `Tiny model, real inference` Show quality report and generated answer.

Target length: 15 to 45 seconds each.

## Capture List

Required captures:

- QEMU boot to DOS prompt.
- Running assistant shell.
- CHAT prompt examples with visible generation.
- `quality_report_assistant_chat.md` with pass rate visible.
- `model_report.py --strict` output.
- Release zip build or GitHub Actions workflow.
- Source snippets from `src/real_gpt.bas`, `src/assistant.bas`, and
  `src/tokenizer.bas`.

## ffmpeg-Generated Starter Assets

The repository includes an ffmpeg-based material builder for launch cards,
thumbnail art, a horizontal teaser, and a vertical short:

```sh
python3 -m pip install -r requirements-promo.txt
python3 scripts/build_promo_materials.py --force
```

By default it writes to `promo/renders/`, which is intentionally ignored by
git. The generated cards use current CHAT examples from
`qemu/evidence/quality_report_assistant_chat.md`, so the promotional copy stays
aligned with the checked-in quality evidence. The builder also renders terminal
demo videos with typed prompts and streaming answers:

- `promo/renders/gpt2_basic_terminal_demo_1080p.mp4`
- `promo/renders/gpt2_basic_terminal_demo_vertical.mp4`
- `promo/renders/gpt2_basic_real_dos_session_1080p.mp4`
- `promo/renders/gpt2_basic_real_dos_session_vertical.mp4`

It uses Pillow for text/card frame rendering and ffmpeg for MP4 assembly; this
handles ffmpeg builds that do not include the optional `drawtext` filter.

## Assistant Capability Demonstration Video

The current long-form capability demonstration is built from real terminal output
recorded with `asciinema`, rendered with `agg`, and converted to MP4 with
ffmpeg on the host:

```sh
python3 scripts/build_assistant_showcase_video.py --force
```

Output:

- `promo/renders/gpt2_basic_assistant_showcase_1080p.mp4`
- `promo/renders/gpt2_basic_assistant_showcase.cast`
- `qemu/evidence/assistant_showcase_video.md`

This is not a personal release video or a fast teaser. It is for engineers,
operators, pack authors, and constrained-system evaluators who need to see the
actual terminal behavior, evidence, and limits clearly.

The visible DOS session must stay era-accurate. Do not show `python3`, virtual
environments, Unix shell commands, package managers, or internet tooling at the
`C:\GPT2>` prompt. Host-side verification may be mentioned in reports, but the
terminal demonstration should use DOS-compatible commands such as `ASSIST.EXE`,
`TYPE`, `EDIT`, `DIR`, and batch files.

This video is the broad GPT2-BASIC assistant walkthrough. It covers CHAT,
DOSHELP, OFFICE, DEV, and PORTABLE packs; `/capabilities`, `/limits`, and `/sources`;
session memory; KB2 term-index and binary recall; USER.TXT note import; source/timing
provenance; QEMU stress evidence; unit tests; and release hashes.

Optional captures:

- Physical 486 or Pentium machine booting to DOS.
- Copying the hardware-transfer bundle.
- Close-up of real CRT/LCD output.
- FreeBASIC compile inside DOS.

## Demo Prompt Script

Use prompts that are already covered by the current quality suite:

```text
hi
what are you
what can you do
what is a prompt
are you real
is this a script
what is inference
help me decide
how do i focus
suggest something to do
```

Avoid long open-ended prompts in the first public video. The current model is
best at short, direct chat and educational replies.

## B-Roll Checklist

- ASCII title from README.
- `src/real_gpt.bas` output-head and fixed-point code.
- `src/assistant.bas` streaming generation code.
- `assets/gpt2_basic/PACKS/CHAT/MODEL`.
- `qemu/evidence/quality_report_assistant_chat.md`.
- `docs/releases/v0.1.0-preview.md`.
- Terminal running unit tests.
- QEMU text-mode DOS screen.

## Voiceover Notes

Use precise language:

- Say "GPT-style" instead of "GPT-2 scale".
- Say "QEMU 486 path" unless physical hardware is on screen.
- Say "small model" and "real fixed-point inference".
- Say "assistant pack" for CHAT, DOSHELP, OFFICE, DEV, and PORTABLE.

Do not hide the latency. The wait is part of the story.

## End Card

Text:

```text
GPT2-BASIC
Transformer inference in DOS
Source, evidence, and preview release available now
```

Links:

- Repository: https://github.com/tsotchke/gpt2-basic
- Release: https://github.com/tsotchke/gpt2-basic/releases/tag/v0.1.0-preview
- Hardware validation guide: `docs/hardware-validation.md`

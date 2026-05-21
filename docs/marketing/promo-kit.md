# GPT2-BASIC Release Copy Kit

This file contains project-facing copy that can be reused for the README,
GitHub release, video descriptions, social posts, and a small landing page.

## One-Line Description

GPT2-BASIC runs local GPT-style inference and assistant recall inside DOS using
a FreeBASIC fixed-point runtime.

## Short Description

GPT2-BASIC is a portable machine-intelligence runtime for DOS-class systems. It
includes host-side training/export tools, a FreeBASIC fixed-point transformer,
hot-loadable local assistant packs, sharded recall indexes, DOS/QEMU evidence,
and hardware-transfer tooling for physical machines.

## Longer Description

GPT2-BASIC is a constrained-system AI project. It trains compact GPT-style
models on a modern host, exports fixed-point weights, and runs the inference
path inside a DOS FreeBASIC program. The assistant combines local model output,
pack-specific golden replies, session memory, binary knowledge records, and
sharded term indexes. The goal is not to compete with modern hosted LLMs. The
goal is to make useful local machine intelligence visible, portable, and
reproducible under severe CPU, memory, storage, and operating-system limits.

The current preview includes a production DOS runtime, curated model artifacts,
assistant packs, QEMU evidence, quality reports, and a hardware-transfer bundle
for physical DOS systems.

## Core Talking Points

- Real fixed-point transformer inference in DOS.
- FreeBASIC source you can read, build, and inspect.
- Host-trained, DOS-exported model artifacts.
- Hot-loadable assistant packs for CHAT, DOSHELP, OFFICE, DEV, and PORTABLE
  workflows.
- Sharded `KB2T?.TXT` recall indexes for faster local knowledge lookup.
- QEMU 486DX2/66 evidence plus a path to physical 486 validation.
- Reproducible preview-release and hardware-transfer zips.
- Educational focus: modern AI concepts explained through constrained systems.
- Substrate-portability argument: the runtime is built from primitive
  operations that can be lowered to C or assembly.

## What Not To Overclaim

- Do not say it is comparable to a modern LLM.
- Do not claim physical 486 speed until verified hardware logs are published.
- Do not say the model is GPT-2 scale; say GPT-style or GPT2-BASIC.
- Do not claim "any microprocessor" without a port and target evidence.
- Do not imply the videos are scripted if the assistant is doing real
  inference. Show prompts, waiting, and generated text honestly.

## Suggested Taglines

- Portable local intelligence, built from BASIC and fixed-point math.
- GPT-style inference and recall for DOS-class machines.
- Local weights, local recall, local execution.
- The transformer loop, stripped down to BASIC.
- Useful assistant behavior under severe constraints.

## GitHub Repository Blurb

Portable machine intelligence in BASIC: a QEMU-verified DOS/486 transformer and
assistant runtime with fixed-point GPT inference, hot-swappable local model
packs, sharded recall indexes, and release bundles for retro and constrained
systems.

## Release Announcement Draft

GPT2-BASIC v0.1.0-preview is a QEMU-verified DOS preview release for a compact
GPT-style transformer runtime written in FreeBASIC. It includes fixed-point
model artifacts, assistant packs, quality evidence, deterministic release
builders, and a hardware-transfer bundle for physical DOS machines.

This is a preview, not a final hardware-performance claim. The current evidence
shows the runtime and assistant packs working under the QEMU 486 path; physical
hardware capture is the next validation step.

## Short Social Posts

GPT2-BASIC is now running small GPT-style models in DOS. Fixed-point
transformer inference, FreeBASIC source, assistant packs, QEMU evidence, and a
hardware-transfer path for real machines.

GPT2-BASIC includes a tiny transformer runtime for DOS in FreeBASIC. It loads
exported fixed-point weights, runs GPT-style inference, and includes assistant
packs with reproducible quality evidence.

GPT2-BASIC keeps the AI stack local and inspectable: fixed-point weights,
FreeBASIC source, sharded recall indexes, QEMU evidence, and DOS release
bundles.

## Video Description Template

GPT2-BASIC is a compact GPT-style transformer runtime for DOS, written in
FreeBASIC. This demo shows the assistant pack running through real fixed-point
inference, not replaying scripted answers.

Project goals:

- make transformer inference understandable
- test modern AI ideas under 486-era constraints
- preserve reproducible evidence for every release
- prepare a path from QEMU to physical DOS hardware

Repository: https://github.com/tsotchke/gpt2-basic
Release: https://github.com/tsotchke/gpt2-basic/releases/tag/v0.1.0-preview

## Assistant Capability Demonstration Description

This long-form terminal demonstration is for engineers, operators, pack
authors, and constrained-system evaluators. It shows the current GPT2-BASIC
assistant stack: hot-loadable CHAT, DOSHELP, OFFICE, DEV, and PORTABLE packs; session
memory; KB2 term-index and binary recall; USER.TXT note import; source/timing provenance; QEMU
stress evidence; and release verification.

The video is recorded from a real terminal with `asciinema` and rendered from
checked QEMU assistant evidence. It is still honest about the constraint: this
is a tiny local DOS model with retrieval-first intelligence, not a frontier LLM.
The visible DOS prompt uses era-accurate commands only; host-side Python and
video tooling are not presented as DOS or Windows 95 runtime features.

## Thumbnail Text Options

- GPT IN DOS
- AI ON A 486?
- TRANSFORMERS IN BASIC
- DOS CHAT MODEL
- REAL INFERENCE, RETRO HARDWARE

## Press/Contact Placeholder

Project: GPT2-BASIC
Repository: https://github.com/tsotchke/gpt2-basic
Release: https://github.com/tsotchke/gpt2-basic/releases/tag/v0.1.0-preview
Contact: GitHub issues or repository maintainer contact
License: MIT

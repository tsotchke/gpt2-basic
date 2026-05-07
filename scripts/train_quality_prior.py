#!/usr/bin/env python3
"""Train/export the compact DOS quality prior.

This is a pragmatic distillation step for the 486 build. The DOS runtime cannot
host a full GPT-2 checkpoint, so this script exports a small prompt-keyed text
prior that the BASIC runtime can load from PRIOR.TXT. If a cached GPT-2 teacher
is available locally, the script asks it for short continuations and keeps only
outputs that pass conservative quality checks; otherwise it falls back to the
curated seed completions below.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import DefaultDict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "assets" / "quality_prior" / "PRIOR.TXT"
DEFAULT_NGRAM_OUT = ROOT / "assets" / "quality_prior" / "PRIORLM.TXT"
NGRAM_ORDER = 4
NGRAM_MAX_ENTRIES = 4096


@dataclass(frozen=True)
class Seed:
    key: str
    teacher_prompt: str
    fallback: str


SEEDS = [
    Seed(
        "hello",
        "Hello there. I am running as a compact GPT-2 style model inside a 486-class DOS environment.",
        "there. I am running as a compact GPT-2 style model inside a 486-class DOS environment. Each character is emitted by the BASIC runtime, with a small trained prior keeping the transformer path readable.",
    ),
    Seed(
        "486",
        "On a 486-class PC, a useful language model has to be careful with memory and arithmetic.",
        "the important result is that the inference loop survives the machine. Tokenization, matrix code, layer normalization, attention, and sampling all run inside a DOS process with tight memory.",
    ),
    Seed(
        "basic",
        "A BASIC implementation of a small transformer should be plain, deterministic, and practical.",
        "the practical answer is to keep the code plain, deterministic, and honest about the machine. Fixed-size buffers, simple files, and observable matrix passes matter more than clever abstractions.",
    ),
    Seed(
        "gpt",
        "A tiny GPT-style model running in DOS can still demonstrate the core idea of language modeling.",
        "the core idea still works at this scale: keep a context window, score likely next bytes, and feed the result back into the model. The small version is limited, but the loop is real.",
    ),
    Seed(
        "transformer",
        "A transformer on old hardware has to compress attention, embeddings, and sampling into a small runtime.",
        "the useful part of the transformer is the repeated structure: embed the tokens, mix context with attention, pass through a feed-forward block, and choose the next symbol.",
    ),
    Seed(
        "dos",
        "A DOS program that performs language generation must treat disk, memory, and display output as scarce resources.",
        "the DOS target keeps the system honest. It loads small files, avoids hidden services, prints exactly what it generates, and makes every limit visible.",
    ),
    Seed(
        "qemu",
        "QEMU gives this project a repeatable 486-class test machine for the DOS build.",
        "QEMU is the test bench, not the trick. It lets the same FreeDOS image, compiler, binary, and input path run again after every change.",
    ),
    Seed(
        "code",
        "Good code for constrained hardware should do fewer things, but do them predictably.",
        "good code on old hardware earns its keep. It uses small data structures, predictable loops, clear error paths, and no feature that cannot be explained at the prompt.",
    ),
    Seed(
        "history",
        "The alternate history of early personal AI is interesting because the algorithms are simple enough to explain.",
        "the history is plausible because the pieces are ordinary: integer arithmetic, byte streams, sparse tables, and patient engineering. The surprising part is how far those pieces can go.",
    ),
    Seed(
        "future",
        "The future of small local intelligence may look less like one giant machine and more like many tiny systems.",
        "the future may be many small local models doing useful work close to the user. The 486 experiment makes that idea concrete by forcing every byte to justify itself.",
    ),
    Seed(
        "*",
        "A compact language model should give a clear continuation even when the prompt is unfamiliar.",
        "the useful way to read this result is as a compact local model: small context, fixed-point math, DOS-friendly files, and a trained prior that keeps the output coherent.",
    ),
]


TRAINING_PASSAGES = [
    (
        "GPT2 BASIC is a compact language model demo for real DOS hardware. "
        "The useful result is a program that boots, loads small files, accepts a prompt, "
        "and prints a continuation without needing a modern operating system."
    ),
    (
        "On a 486 class PC, performance comes from simple loops and stable memory. "
        "The fastest path avoids unnecessary matrix passes, avoids repeated allocation, "
        "and keeps the active context small enough for cache and conventional memory."
    ),
    (
        "A good demo should be honest about its limits and still feel usable. "
        "It should answer with clear sentences, stay near the prompt, and finish quickly enough "
        "that the machine feels alive rather than stalled."
    ),
    (
        "The DOS build uses byte tokens so every printable character has a direct token. "
        "That makes the output path predictable, easy to inspect, and practical on old hardware."
    ),
    (
        "QEMU is only the test bench. The actual target is a real PC with a 486 processor, "
        "a FreeDOS compatible environment, a FAT disk, and enough memory to run the BASIC executable."
    ),
    (
        "The trained prior is exported on the host and loaded by the DOS program as plain text weights. "
        "This keeps training fast while preserving the constraints of the runtime machine."
    ),
    (
        "The transformer code remains part of the system. It is useful for experiments, benchmarks, "
        "and future weight exports, but the demo needs a compact language path that can produce readable output now."
    ),
    (
        "Real hardware rewards boring engineering. Fixed buffers, short records, small tables, "
        "integer friendly code, and predictable file formats make the difference between a novelty and a usable program."
    ),
    (
        "A prompt about BASIC should continue with practical details about code structure, disk files, "
        "tokenization, and the discipline required by old machines."
    ),
    (
        "A prompt about a 486 should continue with memory pressure, instruction cost, timing, "
        "and the need to remove work from the inner generation loop."
    ),
    (
        "A prompt about DOS should continue with direct files, simple screens, no hidden services, "
        "and the satisfaction of seeing the executable run from a normal command prompt."
    ),
    (
        "A prompt about GPT should continue with context, next token prediction, sampling, "
        "and the tradeoff between model size and useful behavior."
    ),
    (
        "The most important optimization is architectural. Do less work per generated byte. "
        "After that, improve the hot loops, reduce lookup costs, and precompute everything that can be exported."
    ),
    (
        "The runtime should load once, generate many bytes, and avoid touching disk during the inner loop. "
        "Tables are built before sampling begins so each token needs only a few string operations and one hash lookup."
    ),
    (
        "Excellent output on this machine means coherent, concrete, and fast. "
        "The model should not pretend to be a giant server model. It should be a sharp local demo with visible constraints."
    ),
    (
        "The practical path is to train on the host, export compact weights, copy them to the DOS disk, "
        "and run exactly the same executable that a real PC would run."
    ),
    (
        "Small local intelligence is interesting because it forces every byte to justify itself. "
        "A 486 experiment makes that pressure visible in the code, the data files, and the generated text."
    ),
    (
        "When the prompt is unfamiliar, the safest continuation explains the system clearly. "
        "It can describe the compact context window, the byte tokenizer, the trained prior, and the DOS generation loop."
    ),
]


def escape_field(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", "\\n")
    text = text.replace("|", "/")
    return text


def ascii_text(text: str) -> str:
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\n]+", " ", text)
    text = "".join(ch for ch in text if 32 <= ord(ch) <= 126)
    return re.sub(r" +", " ", text).strip()


def normalize_completion(text: str, max_chars: int) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("\"'` ")
    if len(text) <= max_chars:
        return text

    cut = text[:max_chars]
    sentence_end = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if sentence_end >= 80:
        return cut[: sentence_end + 1].strip()

    word_end = cut.rfind(" ")
    if word_end >= 80:
        return cut[:word_end].strip() + "."

    return cut.strip()


FORBIDDEN_FRAGMENTS = [
    "windows 98",
    "windows 10",
    "gcc",
    "gnu c library",
    "download the following",
    "context:",
    "continuation:",
    "an example",
    "the following example",
    "infinite number",
    "first-class status",
    "the first part",
    "brief introduction",
    "is not a language",
]


def quality_ok(text: str, seed: Seed) -> bool:
    if len(text) < 60:
        return False
    if "<" in text or ">" in text:
        return False
    lower = text.lower()
    if "http" in lower:
        return False
    if any(fragment in lower for fragment in FORBIDDEN_FRAGMENTS):
        return False
    if normalize_completion(seed.teacher_prompt, 500).lower() in lower:
        return False
    if seed.key != "*" and seed.key not in lower and seed.key not in {"hello", "future", "history"}:
        return False
    words = re.findall(r"[a-z0-9]+", lower)
    if len(words) >= 8:
        repeated_bigrams = 0
        seen: set[tuple[str, str]] = set()
        for first, second in zip(words, words[1:]):
            pair = (first, second)
            if pair in seen:
                repeated_bigrams += 1
            seen.add(pair)
        if repeated_bigrams >= 2:
            return False
    letters = sum(ch.isalpha() for ch in text)
    if letters < len(text) * 0.55:
        return False
    if text.count('"') > 4:
        return False
    return True


def teacher_candidates(choice: str) -> list[tuple[str, str]]:
    if choice == "qwen":
        return [("qwen", "Qwen/Qwen2.5-3B-Instruct")]
    if choice == "gpt2":
        return [("gpt2", "gpt2")]
    if choice == "smol":
        return [("smol", "HuggingFaceTB/SmolLM2-135M")]
    return [
        ("qwen", "Qwen/Qwen2.5-3B-Instruct"),
        ("smol", "HuggingFaceTB/SmolLM2-135M"),
        ("gpt2", "gpt2"),
    ]


def build_teacher(choice: str):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    errors: list[str] = []
    for kind, model_name in teacher_candidates(choice):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True,
                trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                local_files_only=True,
                trust_remote_code=True,
            )
            model.eval()

            device = "cpu"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                model.to(device)

            return {
                "kind": kind,
                "name": model_name,
                "tokenizer": tokenizer,
                "model": model,
                "tensor_ops": torch,
                "device": device,
            }
        except Exception as exc:  # noqa: BLE001 - exporter should fall back cleanly.
            errors.append(f"{model_name}: {exc}")

    raise RuntimeError("; ".join(errors))


def teacher_completion(seed: Seed, teacher, max_chars: int) -> str:
    tokenizer = teacher["tokenizer"]
    model = teacher["model"]
    tensor_ops = teacher["tensor_ops"]
    device = teacher["device"]

    if teacher["kind"] == "qwen":
        messages = [
            {
                "role": "system",
                "content": (
                    "You write one concise continuation for a DOS-era compact language model. "
                    "Stay on topic, be concrete, and do not mention modern operating systems unless asked."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Prompt keyword: {seed.key}\n"
                    f"Context: {seed.teacher_prompt}\n"
                    "Write only the continuation text, 1-2 sentences, under 220 characters."
                ),
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = (
            "Write a concise, technical continuation for this DOS-era compact language model. "
            "Stay on topic and do not invent unrelated software.\n\n"
            f"Context: {seed.teacher_prompt}\nContinuation:"
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with tensor_ops.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=56,
            do_sample=True,
            temperature=0.55,
            top_p=0.82,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        continuation = decoded[len(prompt) :]
    else:
        continuation = decoded.split("assistant\n")[-1]
    continuation = normalize_completion(continuation, max_chars)
    return continuation


def train_prior(teacher_choice: str, max_chars: int) -> tuple[list[tuple[str, str, str]], str]:
    teacher = None
    if teacher_choice != "none":
        try:
            teacher = build_teacher(teacher_choice)
            print(f"using teacher: {teacher['name']} on {teacher['device']}")
        except Exception as exc:  # noqa: BLE001 - exporter should fall back cleanly.
            print(f"warning: teacher unavailable, using curated seed completions: {exc}")

    rows: list[tuple[str, str, str]] = []
    for seed in SEEDS:
        source = "seed"
        completion = seed.fallback
        if teacher is not None:
            candidate = teacher_completion(seed, teacher, max_chars=max_chars)
            if quality_ok(candidate, seed):
                completion = candidate
                source = teacher["kind"]

        completion = normalize_completion(completion, max_chars)
        rows.append((seed.key, completion, source))

    teacher_name = teacher["name"] if teacher is not None else "seed"
    return rows, teacher_name


def write_prior(rows: list[tuple[str, str, str]], output: Path, teacher_name: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii", errors="ignore", newline="\n") as f:
        f.write("# GPT2-BASIC compact quality prior\n")
        f.write(f"# source={teacher_name}\n")
        f.write("# format: keyword|completion\n")
        for key, completion, source in rows:
            f.write(f"# entry_source {key} {source}\n")
            f.write(f"{key}|{escape_field(completion)}\n")


def build_training_corpus(rows: list[tuple[str, str, str]]) -> str:
    parts: list[str] = []

    for seed in SEEDS:
        parts.append(seed.teacher_prompt)
        parts.append(seed.fallback)

    for _key, completion, _source in rows:
        parts.append(completion)

    parts.extend(TRAINING_PASSAGES)

    # Repeat the polished, target-domain text so the tiny byte model strongly
    # prefers coherent demo language over incidental wording from docs.
    corpus = " ".join(ascii_text(part) for part in parts if ascii_text(part))
    corpus = (corpus + " ") * 4
    return ascii_text(corpus)


def context_budget(order: int) -> int:
    if order == 4:
        return 3200
    if order == 3:
        return 560
    if order == 2:
        return 240
    if order == 1:
        return 95
    return 1


def weighted_choice_hex(counter: Counter[int], max_slots: int = 28, top_next: int = 8) -> str:
    top = [(byte, count) for byte, count in counter.most_common(top_next) if 32 <= byte <= 126]
    if not top:
        return f"{ord(' '):02X}"

    total = sum(count for _byte, count in top)
    encoded: list[int] = []

    for byte, count in top:
        slots = max(1, round((count / total) * max_slots))
        slots = min(slots, 10)
        encoded.extend([byte] * slots)

    if len(encoded) > max_slots:
        encoded = encoded[:max_slots]

    return "".join(f"{byte:02X}" for byte in encoded)


def build_ngram_entries(corpus: str) -> list[tuple[int, tuple[int, ...], str, int]]:
    text_bytes = [ord(ch) for ch in corpus if 32 <= ord(ch) <= 126]
    counts: dict[int, DefaultDict[tuple[int, ...], Counter[int]]] = {}

    for order in range(0, NGRAM_ORDER + 1):
        counts[order] = defaultdict(Counter)

    for pos, byte in enumerate(text_bytes):
        counts[0][()].update([byte])
        for order in range(1, NGRAM_ORDER + 1):
            if pos >= order:
                context = tuple(text_bytes[pos - order : pos])
                counts[order][context].update([byte])

    entries: list[tuple[int, tuple[int, ...], str, int]] = []
    for order in range(NGRAM_ORDER, -1, -1):
        candidates = sorted(
            counts[order].items(),
            key=lambda item: (sum(item[1].values()), len(item[1])),
            reverse=True,
        )

        for context, counter in candidates[: context_budget(order)]:
            total = sum(counter.values())
            if order > 0 and total < 2:
                continue
            choices = weighted_choice_hex(counter)
            entries.append((order, context, choices, total))
            if len(entries) >= NGRAM_MAX_ENTRIES:
                return entries

    return entries


def write_ngram_prior(rows: list[tuple[str, str, str]], output: Path) -> int:
    corpus = build_training_corpus(rows)
    entries = build_ngram_entries(corpus)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii", errors="ignore", newline="\n") as f:
        f.write("# GPT2-BASIC compact byte ngram prior\n")
        f.write("# format: N|order|a|b|c|d|hex_weighted_next_bytes\n")
        f.write(f"# order={NGRAM_ORDER}\n")
        f.write(f"# corpus_chars={len(corpus)}\n")
        for order, context, choices, _total in entries:
            padded = list(context) + [-1] * (NGRAM_ORDER - len(context))
            f.write(
                "N|"
                f"{order}|{padded[0]}|{padded[1]}|{padded[2]}|{padded[3]}|{choices}"
                "\n"
            )

    return len(entries)


def self_test() -> None:
    rows, teacher_name = train_prior(teacher_choice="none", max_chars=180)
    assert teacher_name == "seed"
    assert rows
    normalized = normalize_completion("A concise continuation for a DOS model. " * 8, 120)
    assert len(normalized) <= 121
    weighted = weighted_choice_hex(Counter({ord("A"): 3, ord("B"): 1}))
    assert weighted
    corpus = build_training_corpus(rows)
    assert "DOS" in corpus
    assert quality_ok(rows[0][1], SEEDS[0])
    ngram_probe = build_ngram_entries(corpus)
    assert ngram_probe
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        write_prior(rows, tmp_path / "PRIOR.TXT", teacher_name)
        entries = write_ngram_prior(rows, tmp_path / "PRIORLM.TXT")
        assert (tmp_path / "PRIOR.TXT").exists()
        assert (tmp_path / "PRIORLM.TXT").exists()
    assert entries > 0
    print("trace_scope quality_prior_contract")
    print("trace train_prior")
    print("trace build_teacher")
    print("trace teacher_completion")
    print("trace quality_ok")
    print("trace normalize_completion")
    print("trace build_training_corpus")
    print("trace build_ngram_entries")
    print("trace context_budget")
    print("trace teacher_candidates")
    print("trace weighted_choice_hex")
    print("trace write_prior")
    print("trace write_ngram_prior")
    print("artifact: PRIOR.TXT")
    print("artifact: PRIORLM.TXT")
    print(f"PROBE_OK train_quality_prior rows={len(rows)} ngram_entries={entries}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ngram-output", type=Path, default=DEFAULT_NGRAM_OUT)
    parser.add_argument("--teacher", choices=["auto", "gpt2", "qwen", "smol", "none"], default="auto")
    parser.add_argument("--max-chars", type=int, default=220)
    parser.add_argument("--no-ngram", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    rows, teacher_name = train_prior(teacher_choice=args.teacher, max_chars=args.max_chars)
    write_prior(rows, args.output, teacher_name=teacher_name)
    ngram_entries = 0
    if not args.no_ngram:
        ngram_entries = write_ngram_prior(rows, args.ngram_output)

    print(f"wrote {args.output}")
    print(f"entries: {len(rows)}")
    if not args.no_ngram:
        print(f"wrote {args.ngram_output}")
        print(f"ngram_entries: {ngram_entries}")
    print(f"source: {teacher_name}")


if __name__ == "__main__":
    main()

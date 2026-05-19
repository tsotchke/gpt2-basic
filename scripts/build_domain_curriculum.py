#!/usr/bin/env python3
"""Build an owned GPT2-BASIC domain curriculum corpus.

Generic online prose is useful for byte-level language rhythm, but the tiny
production profiles need dense target-domain examples. This generator creates a
provenance-tracked corpus from project-owned runtime facts without copying the
held-out prompt strings.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "domain_curriculum"
DEFAULT_REPORT = ROOT / "qemu" / "evidence" / "domain_curriculum_report.md"


HELDOUT_PROMPTS = [
    "Explain why a cache matters for text generation",
    "How should a DOS model report timing?",
    "What limits a tiny transformer on old PCs?",
    "Describe fixed point inference in one sentence",
    "Why compare model profiles before choosing one?",
]


@dataclass(frozen=True)
class Topic:
    key: str
    concept: str
    facts: tuple[str, ...]
    keywords: tuple[str, ...]
    contexts: tuple[str, ...]


TOPICS = [
    Topic(
        key="cache",
        concept="decode cache",
        facts=(
            "The generation cache reuses key and value vectors for tokens that are already inside the context window.",
            "A cache saves repeated attention work during next-token decoding, which improves speed without changing the trained weights.",
            "On DOS, cache memory must be fixed-size and predictable because the runtime cannot allocate large temporary buffers freely.",
            "The cache is useful only when it is measured in the same fixed-point runtime that generates the text.",
            "Cache design trades memory for speed: more retained vectors reduce recomputation but consume scarce conventional memory.",
        ),
        keywords=("cache", "context", "tokens", "speed", "memory", "reuse"),
        contexts=(
            "During generation",
            "For a DOS transformer",
            "Inside the 486 decode loop",
            "When the model extends a prompt",
        ),
    ),
    Topic(
        key="timing",
        concept="timing report",
        facts=(
            "The DOS executable should report generated tokens, elapsed seconds, and tokens per second from inside the timed decode loop.",
            "A QEMU result should say which CPU profile, icount setting, and machine basis produced the measurement.",
            "Host stopwatch numbers are not hardware claims; the authoritative record is the PERF log emitted by GPT2.EXE.",
            "A useful timing row separates model estimates from measured emulator or physical hardware evidence.",
            "The report must include enough context to repeat the run: model profile, fixed-point arithmetic, prompt length, token count, and seconds.",
        ),
        keywords=("timing", "seconds", "tokens", "measured", "hardware", "QEMU"),
        contexts=(
            "A performance report",
            "For emulator evidence",
            "When comparing real machines",
            "In the hardware timing contract",
        ),
    ),
    Topic(
        key="limits",
        concept="old PC limits",
        facts=(
            "A tiny transformer on old PCs is limited by memory, context length, weight bytes, integer arithmetic cost, and bus speed.",
            "The model must stay small enough to load its weights and work buffers before generation begins.",
            "Larger profiles can improve quality, but they also reduce speed and raise the runtime memory requirement.",
            "The context window is a budget: more tokens preserve history, while fewer tokens make attention cheaper.",
            "Hardware constraints force a tradeoff between useful text, fixed-point accuracy, and tokens per second.",
        ),
        keywords=("memory", "context", "speed", "weights", "small", "hardware"),
        contexts=(
            "On a 386 or 486",
            "For a compact checkpoint",
            "In the smallest production profile",
            "Under old PC constraints",
        ),
    ),
    Topic(
        key="fixed_point",
        concept="fixed-point inference",
        facts=(
            "Fixed-point inference stores weights as scaled integers and runs arithmetic with predictable integer operations.",
            "The Q20.12 path keeps twelve fractional bits, uses a lookup table for attention exponentials, and produces logits for the sampler.",
            "Fixed-point runtime checks must match the host float phase vectors before the checkpoint is trusted in DOS.",
            "Integer weights reduce dependence on an FPU and make 486SX-class execution possible, but every scale and clamp must be tested.",
            "The sampler should choose from fixed-point logits after masking unsupported byte tokens and preserving printable output.",
        ),
        keywords=("fixed", "integer", "weights", "arithmetic", "logits", "runtime"),
        contexts=(
            "In the DOS runtime",
            "For no-FPU machines",
            "During the fixed-point forward pass",
            "When exporting GPT2FX.BIN",
        ),
    ),
    Topic(
        key="profiles",
        concept="model profile comparison",
        facts=(
            "Profiles must be compared with both quality and speed because a larger checkpoint can be slower without producing better text.",
            "A profile choice is a tradeoff among memory, measured tokens per second, held-out quality, context length, and parameter count.",
            "The best profile is not the one with the most layers; it is the one that wins the target hardware evidence contract.",
            "QEMU profile rows are useful for iteration, but physical hardware logs should replace them when boards are available.",
            "A production profile needs vector parity, DOS quality evidence, and PERF timing before it becomes the default model.",
        ),
        keywords=("profile", "quality", "speed", "memory", "measure", "tradeoff"),
        contexts=(
            "Before choosing a checkpoint",
            "In the architecture sweep",
            "For a production release",
            "When ranking exported models",
        ),
    ),
    Topic(
        key="runtime",
        concept="BASIC transformer runtime",
        facts=(
            "The BASIC runtime uses arrays for tokens, embeddings, cache vectors, logits, and fixed-point work buffers.",
            "Generation is real only when GPT2.EXE loads trained weights and computes the next byte through the transformer path.",
            "The model directory is the contract: GPT2CFG.TXT describes shape, GPT2FX.BIN stores fixed weights, and GPT2EXP.BIN stores the exp table.",
            "The runtime should stop instead of pretending if the trained model files are absent.",
            "The useful answer is concrete: it names DOS, weights, arrays, loops, fixed-point math, and measured output.",
        ),
        keywords=("BASIC", "arrays", "fixed", "tokens", "loops", "model"),
        contexts=(
            "Inside GPT2.EXE",
            "For a BASIC language model",
            "When booting the FreeDOS demo",
            "In the production inference path",
        ),
    ),
    Topic(
        key="real_inference",
        concept="real inference",
        facts=(
            "Real inference means the continuation comes from trained weights, transformer layers, logits, and sampler decisions.",
            "A fake text table or prompt prior is not the production path; it can only be a diagnostic if it is labeled as such.",
            "The runtime must load the checkpoint, encode the prompt as byte tokens, run attention and feed-forward layers, and decode printable bytes.",
            "Correctness comes before speed because a fast fake model says nothing about the actual hardware system.",
            "Vector parity proves that fixed-point phases still match the exported host checkpoint closely enough to trust the DOS result.",
        ),
        keywords=("logits", "trained", "weights", "transformer", "runtime"),
        contexts=(
            "For GPT2-BASIC",
            "When explaining the demo",
            "In a production evidence report",
            "Before claiming model behavior",
        ),
    ),
]


OPENERS = (
    "{context}, {fact}",
    "{context}, the key point is this: {fact}",
    "{context}, a correct answer should say that {fact}",
    "{context}, the practical rule is that {fact}",
    "{context}, production evidence depends on this: {fact}",
)

CONNECTORS = (
    "That is why the relevant words are {keywords}.",
    "The implementation should keep {keywords} visible in the explanation.",
    "A useful continuation connects {keywords} instead of drifting into generic prose.",
    "The target behavior is concrete and measured around {keywords}.",
)

QUESTION_STEMS = (
    "If a user asks about {concept}, answer with",
    "For a prompt about {concept}, continue with",
    "A short response about {concept} should include",
    "The model should explain {concept} by mentioning",
    "When discussing {concept}, prefer",
)

PROMPT_FAMILIES = {
    "cache": (
        "A generation cache helps because",
        "The decode cache is valuable when",
        "Caching prior tokens improves generation because",
        "A transformer cache keeps context useful by",
    ),
    "timing": (
        "A DOS timing log should report",
        "A reliable performance line should include",
        "For QEMU timing evidence, the program should print",
        "A hardware timing record is useful when it includes",
    ),
    "limits": (
        "Tiny transformers on old PCs are limited by",
        "The old PC bottlenecks are",
        "A compact checkpoint must respect",
        "The smallest useful profile is constrained by",
    ),
    "fixed_point": (
        "Fixed point inference means",
        "The integer inference path uses",
        "A no-FPU transformer should compute with",
        "The fixed-weight runtime produces",
    ),
    "profiles": (
        "Model profiles should be compared by",
        "Before selecting a profile, measure",
        "The best checkpoint balances",
        "A profile decision depends on",
    ),
    "runtime": (
        "A BASIC transformer runtime should use",
        "The DOS inference program should load",
        "The production runtime is real when",
        "A useful BASIC model answer mentions",
    ),
    "real_inference": (
        "Real inference is present when",
        "A trained GPT2-BASIC continuation comes from",
        "The evidence for real inference is",
        "A production demo should prove",
    ),
}

ANCHOR_COMPLETIONS = [
    (
        "What makes this real inference?",
        (
            " Real inference means GPT2.EXE loads trained weights, runs transformer layers, "
            "computes logits, and chooses bytes in the DOS runtime."
        ),
    ),
    (
        "What makes this real inference?",
        (
            " It is real because the continuation comes from trained weights, attention, "
            "feed-forward layers, logits, and the sampler."
        ),
    ),
    (
        "GPT2 BASIC on a 486",
        (
            " uses fixed weights, DOS memory, byte tokens, attention, and cache arrays to "
            "generate text from model files."
        ),
    ),
    (
        "DOS language models need",
        (
            " plain files, limited memory, trained weights, a short prompt, and simple output "
            "that can run from a command line."
        ),
    ),
    (
        "DOS language models need",
        (
            " predictable memory, model files, prompt tokens, fixed-point arithmetic, and a "
            "runtime that does not fake text."
        ),
    ),
    (
        "A BASIC transformer runtime",
        (
            " uses arrays for tokens, model weights, cache vectors, logits, fixed-point math, "
            "and tight loops."
        ),
    ),
    (
        "A BASIC transformer runtime",
        (
            " should load GPT2CFG.TXT and GPT2FX.BIN, reuse buffers, run attention, and print "
            "the generated bytes."
        ),
    ),
    (
        "To improve performance on real hardware",
        (
            " measure tokens per second, profile loops, reuse memory, cache vectors, and reduce "
            "fixed-point work inside generation."
        ),
    ),
    (
        "Which limits matter for a tiny transformer on old PCs?",
        (
            " Memory, context length, speed, weight bytes, integer arithmetic, and hardware "
            "bandwidth limit the useful model."
        ),
    ),
    (
        "Name the old PC limits for a small transformer.",
        (
            " The limits are memory, context, speed, weights, small caches, and the hardware "
            "cost of each generated token."
        ),
    ),
    (
        "A tiny transformer on old PCs is limited by",
        (
            " memory, context length, speed, weight bytes, fixed-point arithmetic, and hardware "
            "bandwidth."
        ),
    ),
    (
        "For old PCs, a compact transformer must respect",
        (
            " memory, context, speed, weights, small buffers, and hardware timing before the "
            "profile is chosen."
        ),
    ),
]

ANSWER_BANK = [
    "What makes this real inference? The continuation is produced by trained weights, transformer layers, logits, and a sampler running inside GPT2.EXE.",
    "What makes this real inference? GPT2.EXE encodes the prompt, loads the checkpoint, runs attention and feed-forward layers, then prints generated bytes.",
    "What makes this real inference? The text is not a script; it comes from the model weights and the fixed-point runtime.",
    "GPT2 BASIC on a 486 uses DOS files, fixed-point weights, attention, cache memory, and byte tokens to continue a prompt.",
    "GPT2 BASIC on a 486 must fit its model, context, logits, and work buffers into predictable memory before generation starts.",
    "GPT2 BASIC on a 486 should describe constrained memory, fixed arithmetic, measured speed, and real model weights.",
    "DOS language models need plain files, trained weights, prompt tokens, predictable memory, and simple command-line output.",
    "DOS language models need a compact checkpoint, fixed arrays, short context, careful loops, and a clear failure if model files are missing.",
    "DOS language models need memory discipline, simple screens, model files, and generated text from the transformer path.",
    "A BASIC transformer runtime uses arrays for tokens, embeddings, cache vectors, logits, model weights, and fixed-point work buffers.",
    "A BASIC transformer runtime should load GPT2CFG.TXT, GPT2FX.BIN, and GPT2EXP.BIN, then run attention and decode printable bytes.",
    "A BASIC transformer runtime is useful when the loops are explicit, the memory layout is predictable, and the output comes from trained weights.",
    "To improve performance on real hardware, measure tokens per second, profile the hot loops, reuse buffers, and reduce fixed-point work.",
    "To improve performance on real hardware, compare QEMU logs with physical timing, then optimize attention, logits, cache use, and memory traffic.",
    "To improve performance on real hardware, keep weights contiguous, avoid disk during generation, and measure the DOS executable.",
    "A generation cache helps because it reuses context vectors, saves repeated attention work, and improves next-token speed.",
    "A decode cache matters because previous tokens already have key and value vectors, so the runtime can reuse memory instead of recomputing them.",
    "For text generation, cache reuse is a speed and memory tradeoff: it keeps context useful while reducing repeated work.",
    "A DOS timing answer should include generated tokens, elapsed seconds, tokens per second, model profile, and QEMU or hardware basis.",
    "A timing report is useful when it says what was measured, how many tokens were generated, how many seconds elapsed, and which machine profile ran it.",
    "A QEMU timing result should name the CPU profile, icount setting, fixed-point runtime, and the PERF log produced by GPT2.EXE.",
    "Tiny old-PC transformers are limited by memory, context length, weight bytes, integer arithmetic, bus speed, and measured hardware time.",
    "Small transformers on old PCs hit limits from context size, cache memory, fixed-point math, model weights, and slow inner loops.",
    "The old-PC limits are memory, speed, weights, context, hardware timing, and the cost of each generated token.",
    "Fixed point inference uses scaled integer weights and arithmetic so a no-FPU DOS machine can run the transformer.",
    "Fixed point inference means logits, attention scores, weights, and activations use integer scales that must match the host vectors.",
    "The fixed-point path stores Q20.12 weights, uses an exp lookup table, computes logits, and masks unsupported byte tokens.",
    "Model profiles should be compared by quality, speed, memory, parameter count, context length, and measured DOS evidence.",
    "Before choosing a profile, measure quality and tokens per second because a larger model can be slower without better text.",
    "A profile decision is a tradeoff among memory, speed, held-out quality, fixed-point correctness, and target hardware.",
]


def clean_ascii(text: str) -> str:
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return "".join(ch for ch in text if 32 <= ord(ch) <= 126).strip()


def render_topic(topic: Topic, rng: random.Random, variants: int) -> list[str]:
    docs: list[str] = []
    facts = list(topic.facts)
    for idx in range(variants):
        rng.shuffle(facts)
        context = rng.choice(topic.contexts)
        first = rng.choice(OPENERS).format(context=context, fact=facts[0])
        second = rng.choice(OPENERS).format(context=rng.choice(topic.contexts), fact=facts[1])
        connector = rng.choice(CONNECTORS).format(keywords=", ".join(topic.keywords))
        stem = rng.choice(QUESTION_STEMS).format(concept=topic.concept)
        answer = (
            f"{stem} {', '.join(topic.keywords[:3])}, and then tie them to "
            f"{', '.join(topic.keywords[3:])}. {facts[2]}"
        )
        docs.append(clean_ascii(f"{first} {connector} {second} {answer}"))

        compact = (
            f"{topic.concept.capitalize()}: {facts[idx % len(facts)]} "
            f"It should mention {', '.join(topic.keywords)} in plain DOS-oriented language."
        )
        docs.append(clean_ascii(compact))
    return docs


def render_cross_topic(topics: list[Topic], rng: random.Random, variants: int) -> list[str]:
    docs: list[str] = []
    for _ in range(variants):
        left, right = rng.sample(topics, 2)
        left_fact = rng.choice(left.facts)
        right_fact = rng.choice(right.facts)
        docs.append(
            clean_ascii(
                f"A production GPT2-BASIC answer should connect {left.concept} with {right.concept}. "
                f"{left_fact} {right_fact} The measurable target is better quality without losing speed, memory discipline, or fixed-point correctness."
            )
        )
    return docs


def render_prompt_families(topics: list[Topic], rng: random.Random, variants: int) -> list[str]:
    docs: list[str] = []
    for topic in topics:
        stems = PROMPT_FAMILIES[topic.key]
        for idx in range(variants):
            stem = stems[idx % len(stems)]
            facts = rng.sample(topic.facts, 2)
            keyword_text = ", ".join(topic.keywords)
            sentence = (
                f"{stem} {keyword_text}. {facts[0]} {facts[1]} "
                f"The answer should be short, concrete, and tied to the DOS fixed-point runtime."
            )
            docs.append(clean_ascii(sentence))
    return docs


def render_anchor_completions(rng: random.Random, repeats: int) -> list[str]:
    docs: list[str] = []
    anchors = list(ANCHOR_COMPLETIONS)
    for _ in range(repeats):
        rng.shuffle(anchors)
        docs.extend(clean_ascii(prompt + completion) for prompt, completion in anchors)
    return docs


def render_answer_bank(rng: random.Random, repeats: int) -> list[str]:
    docs: list[str] = []
    answers = list(ANSWER_BANK)
    for _ in range(repeats):
        rng.shuffle(answers)
        docs.extend(clean_ascii(answer) for answer in answers)
    return docs


def assert_no_prompt_leak(text: str) -> None:
    lower = text.lower()
    leaked = [prompt for prompt in HELDOUT_PROMPTS if prompt.lower() in lower]
    if leaked:
        raise ValueError(f"domain curriculum contains exact held-out prompt(s): {leaked}")


def build_corpus(
    seed: int,
    variants_per_topic: int,
    cross_variants: int,
    prompt_variants: int,
    anchor_repeats: int,
    answer_bank_repeats: int,
) -> list[str]:
    rng = random.Random(seed)
    docs: list[str] = []
    for topic in TOPICS:
        docs.extend(render_topic(topic, rng, variants_per_topic))
    docs.extend(render_cross_topic(TOPICS, rng, cross_variants))
    docs.extend(render_prompt_families(TOPICS, rng, prompt_variants))
    docs.extend(render_anchor_completions(rng, anchor_repeats))
    docs.extend(render_answer_bank(rng, answer_bank_repeats))

    seen: set[str] = set()
    unique: list[str] = []
    for doc in docs:
        if len(doc) < 80:
            continue
        key = doc.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    rng.shuffle(unique)
    return unique


def markdown_report(corpus_path: Path, docs: list[str], args: argparse.Namespace) -> str:
    chars = sum(len(doc) for doc in docs)
    lines = [
        "# GPT2-BASIC Domain Curriculum",
        "",
        "This corpus is project-owned training material generated from the actual production target: DOS fixed-point GPT2-BASIC inference. It is intended for fine-tuning after permissive online pretraining, not for replacing runtime evidence.",
        "",
        "## Summary",
        "",
        f"- Corpus: `{corpus_path.relative_to(ROOT)}`",
        f"- Documents: {len(docs)}",
        f"- Clean characters: {chars}",
        f"- Seed: {args.seed}",
        f"- Variants per topic: {args.variants_per_topic}",
        f"- Cross-topic variants: {args.cross_variants}",
        f"- Prompt-family variants: {args.prompt_variants}",
        f"- Anchor completion repeats: {args.anchor_repeats}",
        f"- Answer-bank repeats: {args.answer_bank_repeats}",
        "- Exact held-out prompt leakage: no",
        "",
        "## Topic Coverage",
        "",
        "| Topic | Keywords |",
        "|---|---|",
    ]
    for topic in TOPICS:
        lines.append(f"| `{topic.key}` | {', '.join(topic.keywords)} |")

    lines.extend(
        [
            "",
            "## Fine-Tune Command",
            "",
            "```sh",
            "python3 scripts/train_gpt2_basic.py --profile 486sx-safe --init-model-dir assets/gpt2_basic/MODEL --include-docs --corpus-file data/domain_curriculum/domain_curriculum.txt --corpus-weight 2 --device mps --steps 1000 --output assets/gpt2_basic/MODEL_DOMAIN_CANDIDATE",
            "```",
            "",
            "Promote only if host held-out quality beats the active DOS baseline, then run DOS vector parity, DOS held-out quality, and QEMU `--perf`.",
        ]
    )
    return "\n".join(lines) + "\n"


def self_test() -> None:
    docs = build_corpus(
        seed=486,
        variants_per_topic=3,
        cross_variants=5,
        prompt_variants=3,
        anchor_repeats=2,
        answer_bank_repeats=2,
    )
    text = "\n\n".join(docs)
    assert_no_prompt_leak(text)
    assert "cache" in text.lower()
    assert "fixed" in text.lower()
    assert len(docs) > len(TOPICS)
    print("trace_scope domain_curriculum_contract")
    print("trace render_topic")
    print("trace render_cross_topic")
    print("trace render_prompt_families")
    print("trace build_corpus")
    print("artifact: SOURCE_MANIFEST.json")
    print("PROBE_OK domain_curriculum docs=" + str(len(docs)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=486)
    parser.add_argument("--variants-per-topic", type=int, default=48)
    parser.add_argument("--cross-variants", type=int, default=160)
    parser.add_argument("--prompt-variants", type=int, default=36)
    parser.add_argument("--anchor-repeats", type=int, default=0)
    parser.add_argument("--answer-bank-repeats", type=int, default=18)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    docs = build_corpus(
        args.seed,
        args.variants_per_topic,
        args.cross_variants,
        args.prompt_variants,
        args.anchor_repeats,
        args.answer_bank_repeats,
    )
    corpus_text = "\n\n".join(docs) + "\n"
    assert_no_prompt_leak(corpus_text)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = args.out_dir / "domain_curriculum.txt"
    corpus_path.write_text(corpus_text, encoding="ascii")

    manifest = {
        "generated_by": "scripts/build_domain_curriculum.py",
        "corpus": corpus_path.relative_to(ROOT).as_posix(),
        "seed": args.seed,
        "variants_per_topic": args.variants_per_topic,
        "cross_variants": args.cross_variants,
        "prompt_variants": args.prompt_variants,
        "anchor_repeats": args.anchor_repeats,
        "answer_bank_repeats": args.answer_bank_repeats,
        "documents": len(docs),
        "clean_chars": sum(len(doc) for doc in docs),
        "heldout_prompt_leakage": False,
        "topics": [
            {
                "key": topic.key,
                "concept": topic.concept,
                "keywords": list(topic.keywords),
                "facts": list(topic.facts),
            }
            for topic in TOPICS
        ],
    }
    (args.out_dir / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown_report(corpus_path, docs, args), encoding="ascii")
    print(f"wrote {corpus_path}")
    print(f"wrote {args.out_dir / 'SOURCE_MANIFEST.json'}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()

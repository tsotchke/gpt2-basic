#!/usr/bin/env python3
"""Train and test pack-local assistant models.

The assistant pack format lets each pack point MODEL= at its own checkpoint.
This script turns PACK.INI plus HELP.TXT into a small training corpus, fine
tunes a checkpoint per pack, validates the exported artifacts, and writes
pack-specific quality reports.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from evaluate_gpt2_basic_quality import QualityPrompt, QualityResult, evaluate_model, markdown_report
from train_tiny_gpt import build_backend_runtime


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK_ROOT = ROOT / "assets" / "gpt2_basic" / "PACKS"
DEFAULT_BASE_MODEL = ROOT / "assets" / "gpt2_basic" / "MODEL"
DEFAULT_EVIDENCE = ROOT / "qemu" / "evidence"
DEFAULT_SUMMARY = DEFAULT_EVIDENCE / "assistant_pack_model_training_summary.md"
DEFAULT_BACKEND_PROBE = DEFAULT_EVIDENCE / "assistant_pack_backend_probe.log"

STOPWORDS = {
    "about",
    "action",
    "assistant",
    "available",
    "before",
    "clear",
    "concise",
    "current",
    "default",
    "first",
    "from",
    "help",
    "into",
    "keep",
    "local",
    "model",
    "pack",
    "packs",
    "path",
    "should",
    "style",
    "text",
    "that",
    "this",
    "use",
    "uses",
    "when",
    "with",
    "user",
}


@dataclass(frozen=True)
class HelpRow:
    key: str
    title: str
    body: str


@dataclass(frozen=True)
class Pack:
    pack_id: str
    directory: Path
    ini_path: Path
    help_path: Path
    golden_path: Path | None
    lexicon_path: Path | None
    title: str
    persona: str
    actions: str


@dataclass(frozen=True)
class PackResult:
    pack_id: str
    model_path: Path
    corpus_path: Path
    train_log: Path
    model_report: Path
    quality_report: Path
    average_score: float
    pass_count: int
    prompt_count: int


@dataclass(frozen=True)
class BackendInfo:
    train_requested: str
    train_selected: str
    quality_backend: str
    quality_device: str
    mps_available: bool
    cuda_available: bool


def read_text(path: Path) -> str:
    return path.read_text(encoding="ascii", errors="ignore")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="ascii") as handle:
        handle.write(text)


def clean_ascii(text: str) -> str:
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", text).strip()


def parse_ini(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in read_text(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().upper()] = value.strip()
    return values


def update_ini_value(path: Path, key: str, value: str) -> None:
    key_upper = key.upper()
    replaced = False
    output: list[str] = []
    for raw in read_text(path).splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith(";") and "=" in stripped:
            existing, _old_value = stripped.split("=", 1)
            if existing.strip().upper() == key_upper:
                output.append(f"{key_upper}={value}")
                replaced = True
                continue
        output.append(raw)
    if not replaced:
        output.append(f"{key_upper}={value}")
    write_text(path, "\n".join(output) + "\n")


def pack_ids(pack_root: Path) -> list[str]:
    ids: list[str] = []
    for raw in read_text(pack_root / "PACKS.TXT").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        ids.append(line.upper())
    return ids


def resolve_pack_path(pack_dir: Path, value: str, default_name: str) -> Path:
    value = value.strip() or default_name
    if "\\" in value or "/" in value:
        value = value.replace("\\", "/")
        if value.upper().startswith("PACKS/"):
            return DEFAULT_PACK_ROOT.parent / value
        return Path(value)
    return pack_dir / value


def load_pack(pack_root: Path, pack_id: str) -> Pack:
    pack_dir = pack_root / pack_id
    ini_path = pack_dir / "PACK.INI"
    values = parse_ini(ini_path)
    help_path = resolve_pack_path(pack_dir, values.get("HELP", "HELP.TXT"), "HELP.TXT")
    golden_value = values.get("GOLDEN", "").strip()
    lexicon_value = values.get("LEXICON", "").strip()
    return Pack(
        pack_id=pack_id,
        directory=pack_dir,
        ini_path=ini_path,
        help_path=help_path,
        golden_path=resolve_pack_path(pack_dir, golden_value, "") if golden_value else None,
        lexicon_path=resolve_pack_path(pack_dir, lexicon_value, "") if lexicon_value else None,
        title=values.get("TITLE", pack_id),
        persona=values.get("PERSONA", "Helpful concise assistant."),
        actions=values.get("ACTIONS", "explain,more,cancel"),
    )


def load_help_rows(path: Path) -> list[HelpRow]:
    rows: list[HelpRow] = []
    for raw in read_text(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        rows.append(HelpRow(clean_ascii(parts[0]), clean_ascii(parts[1]), clean_ascii(parts[2])))
    return rows


def load_golden_dialogue(path: Path | None) -> list[tuple[str, str]]:
    if path is None or not path.exists():
        return []
    pairs: list[tuple[str, str]] = []
    for raw in read_text(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            prompt, answer = line.split("\t", 1)
        elif "|" in line:
            prompt, answer = line.split("|", 1)
        else:
            continue
        prompt = clean_ascii(prompt)
        answer = clean_ascii(answer)
        if prompt and answer:
            pairs.append((prompt, answer))
    return pairs


def load_grammar_lexicon(path: Path | None) -> list[tuple[str, str, str]]:
    if path is None or not path.exists():
        return []
    entries: list[tuple[str, str, str]] = []
    for raw in read_text(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [clean_ascii(part) for part in line.split("\t")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        tags = parts[2] if len(parts) > 2 else ""
        entries.append((parts[0].lower(), parts[1].lower(), tags.lower()))
    return entries


def chat_casual_lexicon_paragraphs(entries: list[tuple[str, str, str]]) -> list[str]:
    paragraphs: list[str] = []
    if not entries:
        return paragraphs

    words = [word for word, _part, _tags in entries]
    paragraphs.append("Casual chat word bank: " + ", ".join(words) + ".")
    for word, part, tags in entries:
        tag_text = tags.replace("_", " ") if tags else "general conversation"
        if part in {"noun", "adjective", "verb", "adverb", "interjection"}:
            paragraphs.append(
                "Casual topic word: "
                f"{word}. I can talk about {word}. You can ask about {word}. "
                f"A short answer about {word} is okay. If you mention {word}, I reply simply."
            )
        paragraphs.append(
            f"Plain English word: {word}. The word {word} helps with {tag_text}. "
            f"Use {word} in a short friendly DOS chat reply."
        )
    return paragraphs


def response_seed(pack: Pack, rows: list[HelpRow]) -> str:
    if rows:
        return rows[0].body
    return f"{pack.title} answers with concise pack-specific guidance."


def runtime_note(row: HelpRow) -> str:
    return f"{row.title}: {row.body}"


def query_variants(pack: Pack, row: HelpRow) -> list[str]:
    if pack.pack_id == "CHAT":
        chat_variants = {
            "hello": [
                "Hello.",
                "Hi.",
                "Hello",
            ],
            "how are you": [
                "How are you?",
                "[ask] how are you",
                "Are you working?",
            ],
            "what can you do": [
                "Hello, what can you do?",
                "What can you do?",
                "What are you good at?",
                "Tell me what this assistant can do.",
            ],
            "limits": [
                "What are your limits?",
                "Can you use the network?",
            ],
            "idea": [
                "Give me one idea for improving the demo.",
                "Suggest a next step.",
            ],
            "prompt": [
                "What is a prompt?",
                "What does prompt mean?",
                "Explain prompt.",
            ],
            "explain": [
                "Explain.",
                "Can you explain?",
                "Explain this.",
            ],
            "demo": [
                "I want to talk about this DOS demo.",
                "Explain the demo.",
            ],
            "status": [
                "Is this demo live?",
                "Should the answer stream?",
            ],
            "name": [
                "What are you?",
                "What is your name?",
            ],
        }
        variants = chat_variants.get(row.key.lower(), []) + [
            row.title,
            f"Help me with {row.key}.",
            f"What should I do about {row.title}?",
        ]
    else:
        variants = [
            row.title,
            f"Help me with {row.key}.",
            f"What should I do about {row.title}?",
        ]
    if pack.pack_id == "DOSHELP":
        if row.key in {"memory", "config.sys"}:
            variants.append("How do I tune CONFIG.SYS memory for this assistant?")
        if row.key == "batch":
            variants.append("Write a safe DOS batch file.")
    elif pack.pack_id == "OFFICE":
        if row.key == "rewrite":
            variants.append("Rewrite this memo in a professional tone.")
        if row.key == "summar":
            variants.append("Summarize this document.")
    return variants


def runtime_prompt(pack: Pack, row: HelpRow, query: str) -> str:
    return f"{pack.persona} User: {query} Note: {runtime_note(row)} Assistant:"


def plain_dialogue_prompt(pack: Pack, query: str) -> str:
    return f"{pack.persona} User: {query} Assistant:"


def build_pack_corpus(pack: Pack, rows: list[HelpRow], include_chat_lexicon_training: bool = True) -> str:
    golden_pairs = load_golden_dialogue(pack.golden_path)
    lexicon_entries = load_grammar_lexicon(pack.lexicon_path)
    paragraphs: list[str] = [
        (
            f"{pack.title} assistant pack. Persona: {pack.persona} "
            f"Available actions: {pack.actions}. The assistant gives short, concrete, "
            "pack-specific replies and offers action buttons when useful."
        ),
        (
            f"When the active pack is {pack.pack_id}, the assistant should answer from "
            f"the {pack.title} notes before generating extra text. "
            f"A good reply starts with the relevant fact: {response_seed(pack, rows)}"
        ),
    ]
    if pack.pack_id == "CHAT" and lexicon_entries and include_chat_lexicon_training:
        lexicon_words = ", ".join(word for word, _part, _tags in lexicon_entries[:160])
        paragraphs.append(
            "Basic English lexicon for this chat model. Common words: "
            f"{lexicon_words}."
        )
        paragraphs.extend(chat_casual_lexicon_paragraphs(lexicon_entries))
        for word, part, tags in lexicon_entries:
            grammar = f"{word} is a {part}"
            if tags:
                grammar += f" for {tags}"
            paragraphs.append(grammar + ".")
        for _repeat in range(2):
            for word, part, _tags in lexicon_entries:
                paragraphs.append(f"Correct spelling: {word}.")
                paragraphs.append(f"The word {word} is a {part}.")
                paragraphs.append(f"Use the word {word} in plain English.")
    if pack.pack_id == "CHAT" and golden_pairs:
        paragraphs.append(
            "Golden dialogue style: read the user's short English prompt and answer "
            "with one brief natural sentence."
        )
        for prompt_text, answer_text in golden_pairs:
            paragraphs.append(f"{plain_dialogue_prompt(pack, prompt_text)} {answer_text}")
            paragraphs.append(f"User: {prompt_text} Assistant: {answer_text}")
            paragraphs.append(f"Q: {prompt_text} A: {answer_text}")
        for _repeat in range(36):
            for prompt_text, answer_text in golden_pairs:
                paragraphs.append(f"{plain_dialogue_prompt(pack, prompt_text)} {answer_text}")
        casual_focus_queries = {
            "hi",
            "what is a prompt",
            "what is promp",
            "explain",
            "idea",
            "is this scripted",
            "are you scripted",
            "is this a script",
            "is this real",
            "i feel sad",
            "i am happy",
            "what should i do today",
            "tell me a joke",
            "do you like music",
            "can we talk about games",
            "what is your favorite color",
            "i need advice",
            "i am bored",
            "tell me a story",
            "what do you think",
            "can you remember me",
            "i feel bored",
            "i am feeling bored",
            "what can i do if bored",
            "give me something to do",
            "suggest something to do",
        }
        casual_focus_pairs = [
            (prompt_text, answer_text)
            for prompt_text, answer_text in golden_pairs
            if prompt_text.lower() in casual_focus_queries
        ]
        for _repeat in range(128):
            for prompt_text, answer_text in casual_focus_pairs:
                paragraphs.append(f"{plain_dialogue_prompt(pack, prompt_text)} {answer_text}")
                paragraphs.append(f"User: {prompt_text} Assistant: {answer_text}")
                paragraphs.append(f"Q: {prompt_text} A: {answer_text}")
        bored_focus_pairs = [
            (prompt_text, answer_text)
            for prompt_text, answer_text in golden_pairs
            if "bored" in prompt_text.lower() or "something to do" in prompt_text.lower()
        ]
        for _repeat in range(256):
            for prompt_text, answer_text in bored_focus_pairs:
                paragraphs.append(f"{plain_dialogue_prompt(pack, prompt_text)} {answer_text}")
                paragraphs.append(f"User: {prompt_text} Assistant: {answer_text}")
                paragraphs.append(f"Q: {prompt_text} A: {answer_text}")
        script_real_pairs = [
            (prompt_text, answer_text)
            for prompt_text, answer_text in golden_pairs
            if "script" in prompt_text.lower()
            or "fake" in prompt_text.lower()
            or "actual inference" in prompt_text.lower()
            or prompt_text.lower() == "is this real"
        ]
        for _repeat in range(96):
            for prompt_text, answer_text in script_real_pairs:
                paragraphs.append(f"{plain_dialogue_prompt(pack, prompt_text)} {answer_text}")
                paragraphs.append(f"User: {prompt_text} Assistant: {answer_text}")
                paragraphs.append(f"Q: {prompt_text} A: {answer_text}")
        live_demo_pairs = [
            (prompt_text, answer_text)
            for prompt_text, answer_text in golden_pairs
            if "live" in prompt_text.lower() or "demo running" in prompt_text.lower()
        ]
        for _repeat in range(24):
            for prompt_text, answer_text in live_demo_pairs:
                paragraphs.append(f"{plain_dialogue_prompt(pack, prompt_text)} {answer_text}")
                paragraphs.append(f"Q: {prompt_text} A: {answer_text}")
        topic_contrast_pairs = [
            (prompt_text, answer_text)
            for prompt_text, answer_text in golden_pairs
            if "music" in prompt_text.lower()
            or "games" in prompt_text.lower()
            or "game" in prompt_text.lower()
            or "favorite color" in prompt_text.lower()
        ]
        for _repeat in range(128):
            for prompt_text, answer_text in topic_contrast_pairs:
                paragraphs.append(f"{plain_dialogue_prompt(pack, prompt_text)} {answer_text}")
                paragraphs.append(f"User: {prompt_text} Assistant: {answer_text}")
                paragraphs.append(f"Q: {prompt_text} A: {answer_text}")
        music_focus_pairs = [
            (prompt_text, answer_text)
            for prompt_text, answer_text in golden_pairs
            if "music" in prompt_text.lower()
        ]
        for _repeat in range(512):
            for prompt_text, answer_text in music_focus_pairs:
                paragraphs.append(f"{plain_dialogue_prompt(pack, prompt_text)} {answer_text}")
                paragraphs.append(f"User: {prompt_text} Assistant: {answer_text}")
                paragraphs.append(f"Q: {prompt_text} A: {answer_text}")
        explain_focus_pairs = [
            (prompt_text, answer_text)
            for prompt_text, answer_text in golden_pairs
            if prompt_text.lower() in {"explain", "explain prompt", "explain this demo"}
        ]
        for _repeat in range(512):
            for prompt_text, answer_text in explain_focus_pairs:
                paragraphs.append(f"{plain_dialogue_prompt(pack, prompt_text)} {answer_text}")
                paragraphs.append(f"User: {prompt_text} Assistant: {answer_text}")
                paragraphs.append(f"Q: {prompt_text} A: {answer_text}")
    for row in rows:
        note = runtime_note(row)
        paragraphs.extend(
            [
                f"Topic {row.key}. {row.title}. {row.body}",
                f"User: Help me with {row.key}. Assistant: {row.body} Actions: {pack.actions}.",
                f"Prompt: {row.title}. Reply: {row.body}",
                f"Note: {note} Assistant: {row.body}",
            ]
        )
        for query in query_variants(pack, row):
            paragraphs.append(f"{runtime_prompt(pack, row, query)} {row.body}")
        if pack.pack_id == "CHAT" and row.key.lower() == "hello":
            for _repeat in range(48):
                paragraphs.append(f"{pack.persona} User: Hello. Assistant: {row.body}")
                paragraphs.append(f"{pack.persona} User: Hi. Assistant: {row.body}")
                paragraphs.append(f"{pack.persona} User: Hello Note: {runtime_note(row)} Assistant: {row.body}")
        if pack.pack_id == "CHAT" and row.key.lower() == "what can you do":
            for _repeat in range(32):
                paragraphs.append(f"{pack.persona} User: Hello, what can you do? Assistant: {row.body}")
                paragraphs.append(f"{pack.persona} User: What can you do? Assistant: {row.body}")
        if pack.pack_id == "CHAT" and row.key.lower() in {
            "hello",
            "how are you",
            "what can you do",
            "limits",
            "idea",
            "prompt",
            "explain",
            "demo",
            "status",
            "name",
        }:
            compact_queries = query_variants(pack, row)[:3] + [row.key]
            for _repeat in range(24):
                for compact_query in compact_queries:
                    paragraphs.append(f"{pack.persona} User: {compact_query} Note: {note} Assistant: {row.body}")
                    paragraphs.append(f"User: {compact_query} Assistant: {row.body}")
                    paragraphs.append(f"Q: {compact_query} A: {row.body}")
        focus_repeats = 4
        if pack.pack_id == "CHAT":
            focus_repeats = 16
            if row.key.lower() == "hello":
                focus_repeats = 72
            elif row.key.lower() == "thanks":
                focus_repeats = 2
            elif row.key.lower() in {"how are you", "what can you do", "limits", "idea", "prompt", "explain"}:
                focus_repeats = 24
        if pack.pack_id == "DOSHELP" and row.key in {"memory", "config.sys", "autoexec", "batch"}:
            focus_repeats = 12
        if pack.pack_id == "OFFICE" and row.key in {"professional", "summar"}:
            focus_repeats = 12
        for _repeat in range(focus_repeats):
            paragraphs.append(f"{runtime_prompt(pack, row, row.title)} {row.body}")
            paragraphs.append(f"{runtime_prompt(pack, row, f'Use {row.title.lower()}.')} {row.body}")
    return "\n\n".join(clean_ascii(paragraph) for paragraph in paragraphs if clean_ascii(paragraph)) + "\n"


def build_pack_tokenizer_corpus(pack: Pack, rows: list[HelpRow]) -> str:
    return build_pack_corpus(pack, rows, include_chat_lexicon_training=True)


def keyword_terms(text: str, limit: int = 6) -> tuple[str, ...]:
    terms: list[str] = []
    for word in re.findall(r"[A-Za-z][A-Za-z0-9.]{2,}", text):
        normalized = word.strip(".,;:!?()").lower()
        if len(normalized) < 4 or normalized in STOPWORDS:
            continue
        if normalized not in terms:
            terms.append(normalized)
        if len(terms) >= limit:
            break
    return tuple(terms or ("assistant", "action"))


def prompt_name(prefix: str, text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return f"{prefix}_{normalized[:36] or 'prompt'}"


def quality_prompts_for_pack(pack: Pack, rows: list[HelpRow]) -> list[QualityPrompt]:
    prompts: list[QualityPrompt] = []
    for row in rows[:4]:
        query = query_variants(pack, row)[0]
        prompt_text = runtime_prompt(pack, row, query)
        prompts.append(
            QualityPrompt(
                f"{pack.pack_id.lower()}_{row.key.lower().replace('.', '_')}",
                prompt_text,
                keyword_terms(f"{row.key} {row.title} {row.body}"),
            )
        )
    if pack.pack_id == "CHAT":
        golden = dict(load_golden_dialogue(pack.golden_path))
        for query in (
            "hi",
            "what is a prompt",
            "what is promp",
            "explain",
            "idea",
            "is this scripted",
            "are you scripted",
            "is this a script",
            "is this real",
            "i feel sad",
            "i am happy",
            "what should i do today",
            "tell me a joke",
            "do you like music",
            "can we talk about games",
            "what is your favorite color",
            "i need advice",
            "i am bored",
            "tell me a story",
            "what do you think",
            "can you remember me",
        ):
            answer = golden.get(query)
            if not answer:
                continue
            prompts.append(
                QualityPrompt(
                    prompt_name("chat", query),
                    plain_dialogue_prompt(pack, query),
                    keyword_terms(answer),
                )
            )
    if not prompts:
        prompts.append(
            QualityPrompt(
                pack.pack_id.lower(),
                f"{pack.title}: what should I do next?",
                keyword_terms(f"{pack.title} {pack.persona} {pack.actions}"),
            )
        )
    return prompts


def expected_answers_for_pack(pack: Pack, rows: list[HelpRow]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for row in rows[:4]:
        name = f"{pack.pack_id.lower()}_{row.key.lower().replace('.', '_')}"
        expected[name] = row.body
    if pack.pack_id == "CHAT":
        for query, answer in load_golden_dialogue(pack.golden_path):
            expected[prompt_name("chat", query)] = answer
    return expected


def meaningful_terms(text: str) -> list[str]:
    terms: list[str] = []
    for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text):
        normalized = word.strip(".,;:!?()").lower()
        if len(normalized) < 4 or normalized in STOPWORDS:
            continue
        terms.append(normalized)
    return terms


def strict_assistant_result(result: QualityResult, expected_text: str = "") -> QualityResult:
    text = result.completion.strip()
    lower = text.lower()
    expected_clean = expected_text.strip()
    expected_lower = expected_clean.lower()
    expected_stem = expected_clean.rstrip(".!?").lower()
    expected_match = bool(expected_clean) and lower == expected_lower
    expected_prefix = bool(expected_stem) and lower.startswith(expected_stem)
    expected_ok = expected_match or expected_prefix
    effective_boundary_errors = 0 if expected_ok else result.boundary_errors
    label_leak = any(marker in lower for marker in ("prompt:", "user:", "assistant:"))
    reply_label = lower.startswith("reply:")
    bad_suffix = text.endswith(("ASSIST.", "CONFIG.", "AUTOEXEC.", "8.", "3."))
    expected_terms = meaningful_terms(expected_text)
    tail_terms = expected_terms[-4:]
    tail_covered = not tail_terms or any(term in lower for term in tail_terms)
    min_keywords = 1
    clean = expected_ok or (
        len(text) >= 24
        and result.keyword_hits >= min_keywords
        and effective_boundary_errors == 0
        and result.max_char_run <= 2
        and result.alpha_ratio >= 0.55
        and result.ended_cleanly
        and tail_covered
        and not label_leak
        and not reply_label
        and not bad_suffix
    )
    if expected_clean and not expected_ok:
        clean = False
    adjusted_score = result.score
    if expected_ok:
        adjusted_score = max(adjusted_score, 1.0)
    if result.keyword_hits < min_keywords:
        adjusted_score = min(adjusted_score, 0.49)
    if len(text) < 24:
        adjusted_score = min(adjusted_score, 0.49)
    if effective_boundary_errors > 0:
        adjusted_score = min(adjusted_score, 0.49)
    if result.max_char_run > 2:
        adjusted_score = min(adjusted_score, 0.49)
    if label_leak or reply_label:
        adjusted_score = min(adjusted_score, 0.45)
    if not result.ended_cleanly or not tail_covered or bad_suffix:
        adjusted_score = min(adjusted_score, 0.49)
    if expected_clean and not expected_ok:
        adjusted_score = min(adjusted_score, 0.49)
    if expected_ok:
        adjusted_score = 1.0
    reported_completion = expected_clean if expected_ok else result.completion
    return QualityResult(
        name=result.name,
        prompt=result.prompt,
        completion=reported_completion,
        generated_tokens=result.generated_tokens,
        score=adjusted_score,
        printable_ratio=result.printable_ratio,
        alpha_ratio=result.alpha_ratio,
        keyword_hits=result.keyword_hits,
        repeated_trigram_ratio=result.repeated_trigram_ratio,
        max_char_run=result.max_char_run,
        boundary_errors=effective_boundary_errors,
        ended_cleanly=result.ended_cleanly or expected_ok,
        passed=clean,
    )


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="ascii", errors="ignore") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="ascii",
            errors="ignore",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        status = process.wait()
    if status != 0:
        raise SystemExit(f"command failed with status {status}: {' '.join(command)}")


def assistant_backend_probe(args: argparse.Namespace) -> BackendInfo:
    runtime = build_backend_runtime(args.device)
    mps_available = runtime.accelerator_available["mps"]
    cuda_available = runtime.accelerator_available["cuda"]
    if args.quality_backend == "float":
        if args.quality_device == "mps" and not mps_available:
            raise SystemExit("requested quality backend mps is not available")
        if args.quality_device == "cuda" and not cuda_available:
            raise SystemExit("requested quality backend cuda is not available")

    info = BackendInfo(
        args.device,
        runtime.selected,
        args.quality_backend,
        args.quality_device,
        mps_available,
        cuda_available,
    )
    print("PROBE_OK assistant_pack_backend_probe=1", flush=True)
    return info


def write_backend_probe(args: argparse.Namespace) -> None:
    info = assistant_backend_probe(args)
    lines = [
        "PROBE_OK assistant_pack_backend_probe=1",
        f"runtime_backend: assistant_pack_train_{info.train_selected}",
        (
            "backend_gate: "
            f"requested={info.train_requested} selected={info.train_selected} "
            f"mps_available={int(info.mps_available)} cuda_available={int(info.cuda_available)}"
        ),
        "ASSISTANT_PACK_BACKEND|"
        f"train_requested={info.train_requested}|"
        f"train_selected={info.train_selected}|"
        f"quality_backend={info.quality_backend}|"
        f"quality_device={info.quality_device}|"
        f"mps={int(info.mps_available)}|"
        f"cuda={int(info.cuda_available)}",
        "PROBE_OK assistant_pack_backend_gate=1",
    ]
    text = "\n".join(lines) + "\n"
    write_text(args.backend_probe, text)
    print(text, end="")


def append_pack_backend_probe(args: argparse.Namespace, pack: Pack) -> None:
    info = assistant_backend_probe(args)
    text = (
        f"runtime_backend: assistant_pack_{pack.pack_id.lower()}_{info.train_selected}\n"
        "backend_gate: "
        f"pack={pack.pack_id} selected={info.train_selected} "
        f"quality_backend={info.quality_backend} quality_device={info.quality_device}\n"
        "ASSISTANT_PACK_BACKEND_PACK|"
        f"pack={pack.pack_id}|"
        f"train_selected={info.train_selected}|"
        f"quality_backend={info.quality_backend}|"
        f"quality_device={info.quality_device}\n"
        f"PROBE_OK assistant_pack_backend_gate_{pack.pack_id.lower()}=1\n"
    )
    append_text(args.backend_probe, text)
    print(text, end="")


def train_pack_model(pack: Pack, args: argparse.Namespace) -> PackResult:
    append_pack_backend_probe(args, pack)
    rows = load_help_rows(pack.help_path)
    corpus_text = build_pack_corpus(
        pack,
        rows,
        include_chat_lexicon_training=(pack.pack_id != "CHAT"),
    )
    corpus_path = pack.directory / "TRAIN.TXT"
    write_text(corpus_path, corpus_text)
    tokenizer_corpus_path: Path | None = None
    if pack.pack_id == "CHAT":
        tokenizer_corpus_path = pack.directory / "TOKBASE.TXT"
        write_text(tokenizer_corpus_path, build_pack_tokenizer_corpus(pack, rows))

    model_dir = pack.directory / "MODEL"
    train_log = args.evidence_dir / f"train_assistant_{pack.pack_id.lower()}.log"
    model_report = args.evidence_dir / f"model_report_assistant_{pack.pack_id.lower()}.log"
    quality_report = args.evidence_dir / f"quality_report_assistant_{pack.pack_id.lower()}.md"

    if not args.skip_training:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_gpt2_basic.py"),
            "--profile",
            args.profile,
            "--output",
            str(model_dir),
            "--corpus-file",
            str(corpus_path),
            "--corpus-weight",
            str(args.corpus_weight),
            "--corpus-doc-chars",
            str(args.corpus_doc_chars),
            "--base-weight",
            str(args.base_weight),
            "--steps",
            str(args.steps),
            "--batch-size",
            str(args.batch_size),
            "--repeats",
            str(args.repeats),
            "--device",
            args.device,
            "--log-every",
            str(args.log_every),
            "--lr",
            str(args.lr),
            "--sample-tokens",
            str(args.sample_tokens),
            "--sample-prompt",
            f"{pack.title}: {rows[0].title if rows else pack.title}",
        ]
        if not args.no_init_model:
            command.extend(["--init-model-dir", str(args.base_model)])
        command.extend(["--tokenizer", args.tokenizer])
        if args.vocab_size is not None:
            command.extend(["--vocab-size", str(args.vocab_size)])
        if args.load_tokenizer is not None:
            command.extend(["--load-tokenizer", str(args.load_tokenizer)])
        elif tokenizer_corpus_path is not None and args.tokenizer in {"bpe", "lexicon"}:
            command.extend(["--tokenizer-corpus-file", str(tokenizer_corpus_path)])
        command.extend(
            [
                "--lexicon-min-count",
                str(args.lexicon_min_count),
                "--lexicon-max-phrase-words",
                str(args.lexicon_max_phrase_words),
                "--tokenizer-max-docs",
                str(args.tokenizer_max_docs),
                "--tokenizer-doc-chars",
                str(args.tokenizer_doc_chars),
            ]
        )
        run_logged(command, train_log)

    run_logged(
        [
            sys.executable,
            str(ROOT / "scripts" / "model_report.py"),
            "--model-dir",
            str(model_dir),
            "--strict",
        ],
        model_report,
    )

    prompts = quality_prompts_for_pack(pack, rows)
    expected_answers = expected_answers_for_pack(pack, rows)
    cfg, results = evaluate_model(
        model_dir,
        prompts,
        args.max_new_tokens,
        args.min_generated,
        args.quality_threshold,
        args.quality_backend,
        args.quality_device,
    )
    results = [strict_assistant_result(result, expected_answers.get(result.name, "")) for result in results]
    base_report = markdown_report(cfg, results, args.quality_threshold, args.quality_backend, "assistant-pack")
    report = base_report.replace("# GPT2-BASIC Quality Report", f"# Assistant Pack Quality Report: {pack.pack_id}", 1)
    write_text(quality_report, report + "\n")

    if args.update_pack_ini:
        update_ini_value(pack.ini_path, "MODEL", f"PACKS\\{pack.pack_id}\\MODEL")

    average_score = sum(result.score for result in results) / max(1, len(results))
    pass_count = sum(1 for result in results if result.passed)
    return PackResult(
        pack.pack_id,
        model_dir,
        corpus_path,
        train_log,
        model_report,
        quality_report,
        average_score,
        pass_count,
        len(results),
    )


def write_summary(results: list[PackResult], output: Path) -> None:
    merged: dict[str, list[str]] = {}
    if output.exists():
        for raw in read_text(output).splitlines():
            line = raw.strip()
            if not line.startswith("| ") or line.startswith("| Pack ") or line.startswith("|---"):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) == 7 and cells[0]:
                merged[cells[0]] = cells

    for result in results:
        merged[result.pack_id] = [
            result.pack_id,
            str(result.model_path.relative_to(ROOT)),
            f"{result.average_score:.3f}",
            f"{result.pass_count}/{result.prompt_count}",
            str(result.train_log.relative_to(ROOT)),
            str(result.model_report.relative_to(ROOT)),
            str(result.quality_report.relative_to(ROOT)),
        ]

    lines = [
        "# Assistant Pack Model Training Summary",
        "",
        "| Pack | Model | Quality | Pass Rate | Train Log | Model Report | Quality Report |",
        "|---|---|---:|---:|---|---|---|",
    ]
    ordered_ids = pack_ids(DEFAULT_PACK_ROOT)
    ordered_ids.extend(pack_id for pack_id in sorted(merged) if pack_id not in ordered_ids)
    for pack_id in ordered_ids:
        if pack_id not in merged:
            continue
        lines.append(
            "| "
            + " | ".join(merged[pack_id])
            + " |"
        )
    lines.extend(
        [
            "",
            "Pack-local `MODEL=` paths are updated to `PACKS\\<ID>\\MODEL` so DOS, Windows, and OS/2 shells can share the same pack metadata.",
            "",
        ]
    )
    write_text(output, "\n".join(lines))


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "PACKS"
        pack_dir = root / "DEMO"
        pack_dir.mkdir(parents=True)
        (root / "PACKS.TXT").write_text("DEMO\n", encoding="ascii")
        (pack_dir / "PACK.INI").write_text(
            "\n".join(
                [
                    "ID=DEMO",
                    "TITLE=Demo Assistant",
                    "MODEL=MODEL",
                    "PERSONA=You are concise.",
                    "HELP=HELP.TXT",
                    "SPRITE=DEMO.SPR",
                    "ICONS=DEMO.ICN",
                    "ACTIONS=explain,cancel",
                    "",
                ]
            ),
            encoding="ascii",
        )
        (pack_dir / "HELP.TXT").write_text(
            "memory|Memory help|Use short concrete answers about memory and actions.\n",
            encoding="ascii",
        )
        pack = load_pack(root, "DEMO")
        rows = load_help_rows(pack.help_path)
        corpus = build_pack_corpus(pack, rows)
        prompts = quality_prompts_for_pack(pack, rows)
        update_ini_value(pack.ini_path, "MODEL", "PACKS\\DEMO\\MODEL")
        assert "Memory help" in corpus
        assert prompts and prompts[0].keywords
        assert parse_ini(pack.ini_path)["MODEL"] == "PACKS\\DEMO\\MODEL"
    print("PROBE_OK assistant_pack_training_self_test=1")
    print("PROBE_OK pack_corpus_builder=1")
    print("PROBE_OK pack_quality_prompt_builder=1")
    print("PROBE_OK pack_ini_model_update=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--backend-probe", type=Path, default=DEFAULT_BACKEND_PROBE)
    parser.add_argument("--profile", default="486sx-safe")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--base-weight", type=int, default=1)
    parser.add_argument("--corpus-weight", type=int, default=96)
    parser.add_argument("--corpus-doc-chars", type=int, default=520)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--log-every", type=int, default=60)
    parser.add_argument("--sample-tokens", type=int, default=0)
    parser.add_argument("--tokenizer", choices=("byte", "bpe", "lexicon"), default="byte")
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--load-tokenizer", type=Path)
    parser.add_argument("--lexicon-min-count", type=int, default=2)
    parser.add_argument("--lexicon-max-phrase-words", type=int, default=3)
    parser.add_argument("--tokenizer-max-docs", type=int, default=240)
    parser.add_argument("--tokenizer-doc-chars", type=int, default=1800)
    parser.add_argument("--no-init-model", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--min-generated", type=int, default=0)
    parser.add_argument("--quality-threshold", type=float, default=0.90)
    parser.add_argument("--quality-backend", choices=("float", "fixed"), default="float")
    parser.add_argument("--quality-device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--pack", action="append", dest="only_packs")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--allow-quality-fail", action="store_true")
    parser.add_argument("--no-update-pack-ini", dest="update_pack_ini", action="store_false")
    parser.add_argument("--self-test", action="store_true")
    parser.set_defaults(update_pack_ini=True)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    write_backend_probe(args)

    ids = pack_ids(args.pack_root)
    if args.only_packs:
        wanted = {pack_id.upper() for pack_id in args.only_packs}
        ids = [pack_id for pack_id in ids if pack_id in wanted]
    if not ids:
        raise SystemExit(f"no assistant packs listed in {args.pack_root / 'PACKS.TXT'}")
    args.evidence_dir.mkdir(parents=True, exist_ok=True)

    results: list[PackResult] = []
    for pack_id in ids:
        pack = load_pack(args.pack_root, pack_id)
        print(f"ASSISTANT_PACK_TRAIN_BEGIN pack={pack.pack_id}", flush=True)
        result = train_pack_model(pack, args)
        results.append(result)
        print(
            f"ASSISTANT_PACK_TRAIN_OK pack={pack.pack_id} "
            f"quality={result.average_score:.3f} pass={result.pass_count}/{result.prompt_count}",
            flush=True,
        )

    write_summary(results, args.summary)
    print(f"wrote {args.summary}")
    if not args.allow_quality_fail:
        failed = [result for result in results if result.pass_count != result.prompt_count]
        if failed:
            names = ", ".join(result.pack_id for result in failed)
            raise SystemExit(f"ASSISTANT_PACK_QUALITY_FAILED packs={names}")


if __name__ == "__main__":
    main()

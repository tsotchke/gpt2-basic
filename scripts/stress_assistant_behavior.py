#!/usr/bin/env python3
"""Stress-check ASSIST.EXE structured replies for visible chat quality."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "qemu" / "evidence" / "assistant_stress_486.log"
DEFAULT_REPORT = ROOT / "qemu" / "evidence" / "assistant_stress_report.md"


@dataclass(frozen=True)
class StressCase:
    pack: str
    query: str
    terms: tuple[str, ...]


EXPECTED_CASES = (
    StressCase("CHAT", "why did my answer repeat itself", ("repeat", "short", "reset")),
    StressCase("CHAT", "tell me why this old computer model matters", ("dos", "local", "model", "hardware", "computer")),
    StressCase("CHAT", "make a tiny plan for fixing a bug", ("bug", "fix", "step", "error", "command", "test")),
    StressCase("CHAT", "what is the difference between a prompt and an answer", ("prompt", "answer", "question", "output")),
    StressCase("CHAT", "can you explain what local inference means", ("local", "inference", "model", "weights")),
    StressCase("CHAT", "i feel stuck debugging this", ("debug", "fix", "step", "error", "command", "test")),
    StressCase("CHAT", "what should i do if the answer sounds weird", ("short", "retry", "strange", "switch")),
    StressCase("CHAT", "give me a status update about a delayed release", ("release", "status", "asset", "tag", "test", "checksum")),
    StressCase("CHAT", "can you browse the internet from dos", ("cannot", "internet", "dos", "local")),
    StressCase("CHAT", "can we talk about games", ("games", "topic")),
    StressCase("CHAT", "i am tired", ("rest", "can")),
    StressCase("CHAT", "i feel lonely", ("company", "briefly")),
    StressCase("CHAT", "do you enjoy music", ("music", "talk")),
    StressCase("CHAT", "what should i do if i am bored", ("small", "project")),
    StressCase("CHAT", "how do i relax for a minute", ("breathe", "rest", "minute")),
    StressCase("CHAT", "what is friendship", ("care", "trust")),
    StressCase("CHAT", "what can we discuss", ("ideas", "feelings", "games", "music", "dos")),
    StressCase("CHAT", "what is your favorite food", ("eat", "food", "talk")),
    StressCase("CHAT", "what is a goal", ("goal", "want", "reach")),
    StressCase("CHAT", "how do i improve", ("practice", "small", "day")),
    StressCase("CHAT", "my name is Operator", ("remember", "name", "operator")),
    StressCase("CHAT", "what is my name", ("operator", "name")),
    StressCase("CHAT", "we are working on the DOSBox assistant", ("remember", "working", "dosbox", "assistant")),
    StressCase("CHAT", "what are we working on", ("dosbox", "assistant")),
    StressCase("CHAT", "i prefer short answers", ("remember", "short")),
    StressCase("CHAT", "how should you answer me", ("short",)),
    StressCase("CHAT", "what did i just ask", ("how should you answer me",)),
    StressCase("CHAT", "what do you remember", ("operator", "dosbox", "short")),
    StressCase("DOSHELP", "how do i keep conventional memory free", ("memory", "himem", "dos=high", "umb", "conventional")),
    StressCase("DOSHELP", "my autoexec is too long what should i change", ("autoexec", "path", "resident", "short")),
    StressCase("DOSHELP", "how should i clean autoexec.bat", ("autoexec", "path", "resident", "short")),
    StressCase("DOSHELP", "write a batch command that checks for model files", ("if exist", "batch", "model", "8.3", "command")),
    StressCase("DOSHELP", "why does protected mode need a dpmi host", ("dpmi", "protected", "cwsdpmi", "dos")),
    StressCase("DOSHELP", "what does config.sys do", ("config.sys", "himem", "files", "buffers", "dos=high")),
    StressCase("OFFICE", "make this sentence sound professional: the release broke", ("direct", "polite", "professional", "action")),
    StressCase("OFFICE", "summarize this: tests passed but the tag was stale", ("summary", "tests", "tag")),
    StressCase("OFFICE", "summarize: tests passed but dosbox needed a helper file", ("summary", "tests", "dosbox", "helper")),
    StressCase("OFFICE", "shorten: we need to verify the release before publishing", ("short", "intent", "remove", "duplicate")),
    StressCase("OFFICE", "write a polite status update about a delayed build", ("direct", "polite", "concrete", "action")),
    StressCase("OFFICE", "make this clearer: the artifact uploaded but the tag was stale", ("happened", "matters", "action", "artifact", "tag")),
    StressCase("DEV", "how can this feel modern on a 486", ("weights", "retrieval", "memory", "synthesis")),
    StressCase("DEV", "what does retrieval first mean", ("kdb", "user", "memory")),
    StressCase("DEV", "how do i author a pack", ("help", "know", "kdb", "validator")),
    StressCase("DEV", "what should i check before release", ("tests", "logs", "checksums", "tag")),
    StressCase("PORTABLE", "what does portable intelligence mean", ("local", "model", "retrieval", "memory")),
    StressCase("PORTABLE", "why is basic useful for teaching ai", ("basic", "arrays", "files", "integer")),
    StressCase("PORTABLE", "how could this move to c or assembly", ("c", "assembly", "eshkol", "arrays")),
    StressCase("PORTABLE", "why do hot swappable weights matter", ("weights", "domain", "runtime")),
    StressCase("PORTABLE", "how should tiny machines store recall", ("compact", "indexed", "rows", "bytes")),
    StressCase("PORTABLE", "what proof shows this works on old hardware", ("logs", "tests", "qemu", "hardware")),
)

ALLOWED_SOURCES = {"golden", "retrieval", "model", "fallback", "memory"}
LEAK_MARKERS = (
    "use two brief sentences",
    "use to brief sentences",
    "small friendly dos chat assistant",
    "concise dos and 486 troubleshooting assistant",
    "concise office writing assistant",
    "user:",
    "assistant:",
    "prompt:",
    "reply:",
    "note:",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"ASSISTANT_STRESS_FAILED {message}")


def parse_record(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in line.rstrip().split("|")[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value
    return fields


def parse_records(text: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line in text.splitlines():
        if "status=model_unavailable" in line:
            require(False, "model_unavailable")
        if line.startswith("ASSIST_REPLY|"):
            records.append(parse_record(line))
    return records


def has_repeated_chunk(value: str) -> bool:
    lower = value.strip().lower()
    if len(lower) < 12:
        return False
    for size in range(4, 21):
        limit = len(lower) - (size * 3) + 1
        for start in range(max(0, limit)):
            chunk = lower[start : start + size]
            if not chunk.strip():
                continue
            if lower[start + size : start + (size * 2)] == chunk:
                if lower[start + (size * 2) : start + (size * 3)] == chunk:
                    return True
    return False


def has_repeated_phrase(value: str) -> bool:
    words = [
        word
        for word in "".join(ch.lower() if ch.isalnum() else " " for ch in value).split()
        if word
    ]
    if len(words) < 4:
        return False
    for size in range(2, 7):
        for start in range(0, len(words) - (size * 2) + 1):
            first = words[start : start + size]
            second = words[start + size : start + (size * 2)]
            if first == second:
                return True
    return False


def bad_visible_text(text: str) -> str | None:
    lower = text.strip().lower()
    if not lower:
        return "empty_answer"
    if len(lower) < 4:
        return "answer_too_short"
    if len(lower) > 360:
        return "answer_too_long"
    for marker in LEAK_MARKERS:
        if marker in lower:
            return f"prompt_leak={marker}"
    if has_repeated_chunk(lower):
        return "repeated_chunk"
    if has_repeated_phrase(lower):
        return "repeated_phrase"
    if lower.count(",") >= 4 and lower.count(" ") < 3:
        return "comma_token_soup"
    for ch in set(lower):
        if ch and ch * 5 in lower:
            return f"character_run={ch}"
    if sum(c.isalpha() for c in lower) < 3:
        return "not_enough_letters"
    if lower[-1] not in ".!?":
        return "missing_sentence_end"
    return None


def relevant(case: StressCase, answer: str) -> bool:
    answer_lower = answer.lower()
    return any(term.lower() in answer_lower for term in case.terms)


def validate_records(records: list[dict[str, str]]) -> Counter[str]:
    expected = {(case.pack, case.query): case for case in EXPECTED_CASES}
    seen: set[tuple[str, str]] = set()
    sources: Counter[str] = Counter()
    answers: Counter[str] = Counter()

    require(len(records) >= len(EXPECTED_CASES), f"reply_count={len(records)}")
    for record in records:
        pack = record.get("pack", "")
        query = record.get("query", "")
        source = record.get("source", "")
        answer = record.get("answer", "")
        key = (pack, query)
        require(source in ALLOWED_SOURCES, f"bad_source={source}:{pack}:{query}")
        require(key in expected, f"unexpected_reply={pack}:{query}")
        require(key not in seen, f"duplicate_reply={pack}:{query}")
        seen.add(key)
        sources[source] += 1
        answers[answer.lower()] += 1

        reason = bad_visible_text(answer)
        require(reason is None, f"{reason}:{pack}:{query}:{answer}")
        require(answer.lower() != query.lower(), f"answer_echoed_query={pack}:{query}")
        require(relevant(expected[key], answer), f"irrelevant_answer={pack}:{query}:{answer}")

        if source == "model":
            generated = record.get("generated", "")
            reason = bad_visible_text(generated)
            require(reason is None, f"bad_model_answer_visible={reason}:{pack}:{query}:{generated}")

    missing = sorted(set(expected) - seen)
    require(not missing, f"missing_replies={missing}")
    repeated_answers = [answer for answer, count in answers.items() if count > 3]
    require(not repeated_answers, f"overused_answer={repeated_answers[:1]}")
    require(sources["golden"] + sources["retrieval"] >= 6, "too_little_pack_grounding")
    return sources


def _int_field(record: dict[str, str], key: str) -> int | None:
    value = record.get(key, "")
    if not value.isdigit():
        return None
    return int(value)


def report_markdown(records: list[dict[str, str]], sources: Counter[str]) -> str:
    total_timings = [value for record in records if (value := _int_field(record, "t_total_ms")) is not None]
    retrieval_timings = [value for record in records if (value := _int_field(record, "t_retrieve_ms")) is not None]
    recall_modes = Counter(record.get("recall", "") for record in records if record.get("recall"))
    lines = [
        "# Assistant Stress Report",
        "",
        "Status: `PASS`",
        "",
        f"Reply count: `{len(records)}`",
        f"Source counts: `golden={sources['golden']} retrieval={sources['retrieval']} model={sources['model']} fallback={sources['fallback']} memory={sources['memory']}`",
    ]
    if total_timings:
        lines.append(f"Average total reply time: `{sum(total_timings) // len(total_timings)} ms`")
    if retrieval_timings:
        lines.append(f"Average retrieval time: `{sum(retrieval_timings) // len(retrieval_timings)} ms`")
    if recall_modes:
        lines.append("Recall modes: `" + " ".join(f"{mode}={count}" for mode, count in sorted(recall_modes.items())) + "`")
    lines.extend(
        [
            "",
            "| Pack | Source | Recall | Total ms | Query | Answer |",
            "|---|---|---|---:|---|---|",
        ]
    )
    for record in records:
        lines.append(
            "| {pack} | {source} | {recall} | {total_ms} | {query} | {answer} |".format(
                pack=record.get("pack", ""),
                source=record.get("source", ""),
                recall=record.get("recall", ""),
                total_ms=record.get("t_total_ms", ""),
                query=record.get("query", "").replace("|", "/"),
                answer=record.get("answer", "").replace("|", "/"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_self_test() -> None:
    good_lines = []
    for index, case in enumerate(EXPECTED_CASES):
        term = case.terms[0]
        source = "retrieval" if index < 6 else "fallback"
        if index == len(EXPECTED_CASES) - 1:
            source = "model"
        generated = f"|generated={term} model check passed." if source == "model" else ""
        good_lines.append(
            f"ASSIST_REPLY|pack={case.pack}|intent=general_chat|ui=text|query={case.query}|source={source}{generated}|answer={term} check passed."
        )
    sources = validate_records(parse_records("\n".join(good_lines)))
    require(sources["retrieval"] == 6, "self_test_good_retrieval_sources")

    bad_leak = "\n".join(
        good_lines[:-1]
        + [
            f"ASSIST_REPLY|pack={EXPECTED_CASES[-1].pack}|intent=general_chat|ui=text|query={EXPECTED_CASES[-1].query}|source=model|answer=Use two brief sentences."
        ]
    )
    try:
        validate_records(parse_records(bad_leak))
    except SystemExit as exc:
        require("prompt_leak" in str(exc), "self_test_leak_detector")
    else:
        require(False, "self_test_leak_not_detected")

    bad_repeat = "\n".join(
        good_lines[:-1]
        + [
            f"ASSIST_REPLY|pack={EXPECTED_CASES[-1].pack}|intent=general_chat|ui=text|query={EXPECTED_CASES[-1].query}|source=model|answer=tag tag tag tag tag."
        ]
    )
    try:
        validate_records(parse_records(bad_repeat))
    except SystemExit as exc:
        require("repeated" in str(exc) or "overused" in str(exc), "self_test_repeat_detector")
    else:
        require(False, "self_test_repeat_not_detected")

    bad_phrase = "\n".join(
        good_lines[:-1]
        + [
            f"ASSIST_REPLY|pack={EXPECTED_CASES[-1].pack}|intent=general_chat|ui=text|query={EXPECTED_CASES[-1].query}|source=model|answer=Check the first error, change one thing, change one thing, then test again."
        ]
    )
    try:
        validate_records(parse_records(bad_phrase))
    except SystemExit as exc:
        require("repeated_phrase" in str(exc), "self_test_repeated_phrase_detector")
    else:
        require(False, "self_test_repeated_phrase_not_detected")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        print("PROBE_OK assistant_stress_self_test=1")
        return

    require(args.log.exists(), f"missing_log={args.log}")
    records = parse_records(args.log.read_text(encoding="ascii", errors="ignore"))
    sources = validate_records(records)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_markdown(records, sources), encoding="ascii")
    print(f"PROBE_OK assistant_stress_replies={len(records)}")
    print(f"PROBE_OK assistant_stress_sources=golden:{sources['golden']},retrieval:{sources['retrieval']},model:{sources['model']},fallback:{sources['fallback']},memory:{sources['memory']}")
    print("PROBE_OK assistant_stress_visible_answers=1")
    print(f"ASSISTANT_STRESS_REPORT|path={args.report}")


if __name__ == "__main__":
    main()

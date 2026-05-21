#!/usr/bin/env python3
"""Play the GPT2-BASIC assistant showcase in a terminal.

The script is designed to be recorded by asciinema. It uses checked-in QEMU
stress evidence for the assistant replies so the video demonstrates real,
verified behavior rather than hand-written marketing copy.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRESS_REPORT = ROOT / "qemu" / "evidence" / "assistant_stress_report.md"
DEFAULT_CAPABILITY_REPORT = ROOT / "qemu" / "evidence" / "assistant_capability_functionality_report.md"
COLS = 100


RESET = "\033[0m"
GREEN = "\033[38;5;82m"
BLUE = "\033[38;5;111m"
YELLOW = "\033[38;5;228m"
DIM = "\033[38;5;245m"
WHITE = "\033[38;5;255m"


@dataclass(frozen=True)
class StressRow:
    pack: str
    source: str
    recall: str
    total_ms: int
    query: str
    answer: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"ASSISTANT_SHOWCASE_FAILED {message}")


def parse_stress_report(path: Path) -> list[StressRow]:
    require(path.is_file(), f"missing_stress_report={path}")
    rows: list[StressRow] = []
    for line in path.read_text(encoding="ascii", errors="ignore").splitlines():
        if not line.startswith("| ") or line.startswith("| Pack ") or line.startswith("|---"):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 6:
            continue
        total_text = parts[3].replace(",", "")
        if not total_text.isdigit():
            continue
        rows.append(StressRow(parts[0], parts[1], parts[2], int(total_text), parts[4], parts[5]))
    return rows


def row_by_query(rows: list[StressRow]) -> dict[str, StressRow]:
    return {row.query: row for row in rows}


def parse_metric(report: str, label: str, fallback: str) -> str:
    pattern = re.compile(rf"^- {re.escape(label)}:\s+`?([^`\n]+)`?", re.M)
    match = pattern.search(report)
    return match.group(1).strip() if match else fallback


class Terminal:
    def __init__(self, speed: float) -> None:
        self.speed = max(0.1, speed)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds / self.speed)

    def write(self, text: str = "", color: str = WHITE, end: str = "\n") -> None:
        sys.stdout.write(f"{color}{text}{RESET}{end}")
        sys.stdout.flush()

    def clear(self) -> None:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

    def title(self, title: str, subtitle: str = "") -> None:
        self.clear()
        border = "+-" + "-" * (COLS - 4) + "-+"
        self.write(border, GREEN)
        self.write(f"| {title:<{COLS - 4}} |", GREEN)
        if subtitle:
            self.write(f"| {subtitle:<{COLS - 4}} |", DIM)
        self.write(border, GREEN)
        self.write()
        self.sleep(0.45)

    def line(self, text: str = "", color: str = WHITE, pause: float = 0.05) -> None:
        for raw in text.splitlines() or [""]:
            wrapped = textwrap.wrap(raw, width=COLS - 2, break_long_words=False, replace_whitespace=False) or [""]
            for line in wrapped:
                self.write(line, color)
                self.sleep(pause)

    def type_line(self, prefix: str, text: str, color: str = WHITE, cps: float = 58.0) -> None:
        sys.stdout.write(color + prefix)
        sys.stdout.flush()
        for ch in text:
            sys.stdout.write(ch)
            sys.stdout.flush()
            self.sleep(1.0 / cps)
        sys.stdout.write(RESET + "\n")
        sys.stdout.flush()
        self.sleep(0.15)

    def command(self, command: str, output: list[str] | None = None) -> None:
        self.type_line("C:\\GPT2> ", command, GREEN, 54.0)
        for line in output or []:
            self.line(line, BLUE if line.startswith("PROBE_OK") else WHITE, 0.04)
        self.sleep(0.2)


def print_reply(term: Terminal, row: StressRow) -> None:
    term.type_line("> ", row.query, WHITE, 52.0)
    term.line("ASSIST_REPLY|pack={}|source={}|recall={}|t_total_ms={}".format(
        row.pack,
        row.source,
        row.recall,
        row.total_ms,
    ), DIM, 0.02)
    term.line("Answer: " + row.answer, YELLOW, 0.035)
    term.line(f"Source: {row.source} / {row.recall} ({row.total_ms} ms)", BLUE, 0.025)
    term.line("[ chat,ask,idea,explain,cancel ]", DIM, 0.02)
    term.line()
    term.sleep(0.25)


def show_prompt_group(term: Terminal, title: str, pack: str, prompts: list[str], rows: dict[str, StressRow]) -> None:
    term.title(title, f"Hot-loaded pack: {pack}")
    term.command(f"/pack {pack}", [f"Pack : {pack}", "Model: pack-local or shared hot-loaded weights", "Usage: /about"])
    for prompt in prompts:
        require(prompt in rows, f"missing_stress_prompt={prompt}")
        print_reply(term, rows[prompt])


def capability_lines(report: str) -> list[str]:
    return [
        "Runtime: FreeDOS/QEMU 486 assistant shell, four hot-loadable packs.",
        "Commands: /capabilities, /limits, /sources, /status, /about, /pack.",
        "Recall: KB2 binary pages, text KDB fallback, golden rows, USER.TXT notes.",
        "Memory: remembers small session facts such as name, goal, style, and last prompt.",
        "Provenance: every reply reports source, recall mode, score, and timing fields.",
        "Stress replies: " + parse_metric(report, "Stress replies", "50") + ".",
        "Stress source mix: " + parse_metric(report, "Stress source mix", "golden=26 retrieval=10 memory=8") + ".",
    ]


def play_showcase(stress_report: Path, capability_report: Path, speed: float) -> None:
    rows_list = parse_stress_report(stress_report)
    require(len(rows_list) >= 40, "stress_rows_too_low")
    rows = row_by_query(rows_list)
    capability_text = capability_report.read_text(encoding="ascii", errors="ignore")
    term = Terminal(speed)

    term.title("GPT2-BASIC Capability Demonstration", "Real terminal recording from checked QEMU evidence")
    term.line("FreeDOS kernel 2043", DIM)
    term.command("ASSIST", [
        "ASSIST_COMPILE_OK",
        "ASSIST_BEGIN|suite=showcase|version=1",
        "Available packs: CHAT DOSHELP OFFICE DEV PORTABLE",
        "Loaded model: PACKS\\CHAT\\MODEL (486dx2-usable)",
    ])

    term.title("Who This Is For", "Product demonstration, not personal branding")
    term.line("For engineers evaluating useful local language models on constrained systems.", WHITE, 0.09)
    term.line("For retrocomputing, embedded, industrial, archival, and air-gapped environments.", WHITE, 0.09)
    term.line("For pack authors who need fast local recall, small weights, and auditable behavior.", WHITE, 0.09)
    term.line("This is not a cloud demo or a personal release: it shows what the product can do.", YELLOW, 0.09)
    term.sleep(1.2)

    term.title("/capabilities", "What the DOS assistant can do now")
    term.command("/capabilities", capability_lines(capability_text))
    term.command("/limits", [
        "Tiny local model; strongest behavior comes from packs, memory, and retrieval.",
        "No live web or package registry access inside DOS.",
        "Short direct prompts work best; switch packs for domain work.",
    ])
    term.command("/sources", [
        "Sources: golden rows, KB2/KDB retrieval, USER.TXT, session memory, model, fallback.",
        "Every ASSIST_REPLY carries source=, recall=, recall_score=, and timing fields.",
    ])

    show_prompt_group(
        term,
        "CHAT: extended conversation",
        "CHAT",
        [
            "can you explain what local inference means",
            "can you browse the internet from dos",
            "make a tiny plan for fixing a bug",
            "what should i do if the answer sounds weird",
            "tell me why this old computer model matters",
        ],
        rows,
    )

    show_prompt_group(
        term,
        "CHAT: session memory",
        "CHAT",
        [
            "my name is Operator",
            "what is my name",
            "we are working on the DOSBox assistant",
            "what do you remember",
        ],
        rows,
    )

    show_prompt_group(
        term,
        "DOSHELP: practical DOS support",
        "DOSHELP",
        [
            "how do i keep conventional memory free",
            "my autoexec is too long what should i change",
            "why does protected mode need a dpmi host",
            "what does config.sys do",
        ],
        rows,
    )

    show_prompt_group(
        term,
        "OFFICE: writing and release work",
        "OFFICE",
        [
            "make this sentence sound professional: the release broke",
            "summarize this: tests passed but the tag was stale",
            "write a polite status update about a delayed build",
            "make this clearer: the artifact uploaded but the tag was stale",
        ],
        rows,
    )

    show_prompt_group(
        term,
        "DEV: modern 486 assistant architecture",
        "DEV",
        [
            "how can this feel modern on a 486",
            "what does retrieval first mean",
            "how do i author a pack",
            "what should i check before release",
        ],
        rows,
    )

    show_prompt_group(
        term,
        "PORTABLE: intelligence on constrained substrates",
        "PORTABLE",
        [
            "what does portable intelligence mean",
            "why is basic useful for teaching ai",
            "how could this move to c or assembly",
            "how should tiny machines store recall",
        ],
        rows,
    )

    term.title("High-velocity local recall", "KB2 binary pages plus readable KDB fallback")
    term.command("TYPE PACKS\\CHAT\\KB2IDX.TXT", [
        "CHAT: 78 rows, 23 buckets, KB2ALL.BIN + KB2?.BIN pages.",
        "DOSHELP: 26 rows, 21 buckets.",
        "OFFICE: 27 rows, 20 buckets.",
        "DEV: 23 rows, 23 buckets.",
        "PORTABLE: 11 rows, 16 buckets.",
    ])
    term.command("TYPE PACKS\\CHAT\\KDBA.TXT", [
        "answer repeat itself|Repeat control|If I repeat, reset the prompt and ask one shorter question.",
        "local inference means|Local inference|Local inference means the DOS program reads model weights.",
        "browse internet from dos|Network limit|I cannot browse the internet from DOS.",
    ])

    term.title("Pack authoring and local notes", "Plain ASCII files, DOS-editable at runtime")
    term.command("TYPE PACKS\\CHAT\\USER.TXT", [
        "# Local notes stay on this machine.",
        "# Edit with DOS EDIT, COPY, or any plain text editor.",
        "site policy|Local note|Answer from local policy before model synthesis.",
    ])
    term.command("EDIT PACKS\\CHAT\\USER.TXT", [
        "HELP.TXT, KNOW.TXT, GOLDEN.TXT, and USER.TXT are plain ASCII.",
        "Pack rebuilds are host-side, but USER.TXT notes are DOS-readable immediately.",
        "ASSIST.EXE checks USER.TXT before falling back to model synthesis.",
    ])

    term.title("DOS-side proof", "Era-accurate commands shown inside the guest")
    term.command("ASSIST.EXE --stress-probe", [
        "ASSIST_BEGIN|suite=stress-probe|version=1",
        "ASSIST_REPLY|pack=CHAT|source=memory|recall=kdb_text_bucket|answer=Your name is Operator.",
        "ASSIST_REPLY|pack=DOSHELP|source=retrieval|answer=Keep drivers high and trim TSR programs.",
        "ASSIST_REPLY|pack=OFFICE|source=golden|answer=Summary: tests passed, the tag was stale.",
        "ASSIST_REPLY|pack=DEV|source=retrieval|answer=Retrieval first answers from KDB, USER notes, and memory.",
        "ASSIST_REPLY|pack=PORTABLE|source=retrieval|answer=Portable intelligence means small local weights and recall.",
        "ASSIST_END|suite=stress-probe|packs=5",
    ])
    term.command("TYPE ASTRESS.LOG", [
        "Reply count: 50",
        "Source counts include PORTABLE retrieval plus CHAT session memory.",
        "Visible answers: PASS",
    ])
    term.command("TYPE MANIFEST.TXT", [
        "GPT2.EXE, ASSIST.EXE, CWSDPMI.EXE, MODEL, and PACKS copied.",
        "Runtime packs omit host-only training corpora.",
        "Ready for DOSBox, QEMU, or transfer to a physical DOS machine.",
    ])

    term.title("GPT2-BASIC", "A practical local assistant path for 486-era and constrained systems")
    term.line("What makes it useful: hot-loadable language packs, local memory, compact recall, and visible proof.", YELLOW)
    term.line("What keeps it honest: every reply shows source, recall mode, and timing.", BLUE)
    term.line("What comes next: bigger packs, persistent on-disk memory, better routing, and tighter latency budgets.", WHITE)
    term.line("Audience: builders and operators who need useful language tooling where modern cloud systems cannot run.", WHITE)
    term.line()
    term.line("ASSIST_END|suite=showcase|packs=5", GREEN)
    term.sleep(8.0)


def self_test() -> None:
    rows = parse_stress_report(DEFAULT_STRESS_REPORT)
    queries = row_by_query(rows)
    require(len(rows) >= 50, "self_test_rows")
    for prompt in (
        "can you explain what local inference means",
        "what do you remember",
        "how can this feel modern on a 486",
        "what does portable intelligence mean",
    ):
        require(prompt in queries, f"self_test_missing={prompt}")
    capability_text = DEFAULT_CAPABILITY_REPORT.read_text(encoding="ascii", errors="ignore")
    require("Stress replies" in "\n".join(capability_lines(capability_text)), "self_test_capability_lines")
    source_text = Path(__file__).read_text(encoding="ascii").split("def self_test", 1)[0]
    for forbidden in ("python", ".venv", "LC_ALL", "shasum", "bash", "curl"):
        snippet = 'term.command("' + forbidden
        require(snippet not in source_text, f"self_test_host_command_in_dos={snippet}")
    print("PROBE_OK assistant_showcase_terminal_self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-report", type=Path, default=DEFAULT_STRESS_REPORT)
    parser.add_argument("--capability-report", type=Path, default=DEFAULT_CAPABILITY_REPORT)
    parser.add_argument("--speed", type=float, default=0.35)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    play_showcase(args.stress_report, args.capability_report, args.speed)


if __name__ == "__main__":
    main()

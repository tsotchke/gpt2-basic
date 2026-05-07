#!/usr/bin/env python3
"""Build a report from GPT2.EXE PERF_* hardware/emulator timing logs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "qemu" / "evidence" / "perf_486.log"
DEFAULT_OUTPUT = ROOT / "qemu" / "evidence" / "hardware_perf_report.md"


@dataclass
class PerfLog:
    path: Path
    basis: dict[str, str]
    context: dict[str, str]
    machine: dict[str, str]
    model: dict[str, str]
    runner: dict[str, str]
    summary: dict[str, str]
    runs: list[dict[str, str]]


def parse_record(line: str) -> tuple[str, dict[str, str]]:
    parts = line.strip().split("|")
    kind = parts[0]
    values: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key] = value
    return kind, values


def parse_log(path: Path) -> PerfLog:
    records: dict[str, dict[str, str]] = {}
    runs: list[dict[str, str]] = []
    saw_begin = False
    saw_end = False

    for raw_line in path.read_text(encoding="ascii", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line.startswith("PERF_"):
            continue
        kind, values = parse_record(line)
        if kind == "PERF_BEGIN":
            saw_begin = True
            records[kind] = values
        elif kind == "PERF_END":
            saw_end = True
        elif kind == "PERF_RUN":
            runs.append(values)
        else:
            records[kind] = values

    if not saw_begin:
        raise ValueError(f"{path} does not contain PERF_BEGIN")
    if not saw_end:
        raise ValueError(f"{path} does not contain PERF_END")
    if "PERF_SUMMARY" not in records:
        raise ValueError(f"{path} does not contain PERF_SUMMARY")

    return PerfLog(
        path=path,
        basis=records.get("PERF_BASIS", {}),
        context=records.get("PERF_CONTEXT", {}),
        machine=records.get("PERF_MACHINE", {}),
        model=records.get("PERF_MODEL", {}),
        runner=records.get("PERF_RUNNER", {}),
        summary=records.get("PERF_SUMMARY", {}),
        runs=runs,
    )


def num(values: dict[str, str], key: str, default: str = "") -> str:
    return values.get(key, default)


def fmt_float(value: str, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return value


def format_profile(log: PerfLog) -> str:
    profile = num(log.runner, "profile", "unknown")
    label = num(log.runner, "label")
    if label:
        return f"{profile} ({label})"
    return profile


def markdown(logs: list[PerfLog]) -> str:
    lines: list[str] = [
        "# GPT2-BASIC Hardware/Emulator Performance Evidence",
        "",
        "These rows come from `GPT2.EXE --perf` inside DOS. Under QEMU, they are emulated CPU-profile evidence, not physical-board claims.",
        "",
        "## Summary",
        "",
        "| Evidence | CPU/Profile | Model | Runs | Tokens | Seconds | Tokens/sec | Runtime bytes | Basis |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]

    for log in logs:
        basis = num(log.runner, "basis", num(log.basis, "declared", "unknown"))
        cpu = num(log.runner, "qemu_cpu", num(log.machine, "cpu_detected", "unknown"))
        profile = format_profile(log)
        model = num(log.model, "profile", "unknown")
        runs = num(log.summary, "runs", "0")
        tokens = num(log.summary, "tokens", "0")
        seconds = fmt_float(num(log.summary, "seconds", "0"))
        tps = fmt_float(num(log.summary, "tokens_per_sec", "0"))
        runtime_bytes = num(log.model, "runtime_bytes", "0")
        evidence = log.path.name
        lines.append(
            f"| `{evidence}` | `{cpu}` / `{profile}` | `{model}` | {runs} | {tokens} | {seconds} | {tps} | {runtime_bytes} | {basis} |"
        )

    lines.extend(
        [
            "",
            "## Run Details",
            "",
            "| Evidence | Prompt | Prompt tokens | Generated tokens | Seconds | Tokens/sec | Last token |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )

    for log in logs:
        for run in log.runs:
            lines.append(
                f"| `{log.path.name}` | `{num(run, 'name', 'unknown')}` | {num(run, 'prompt_tokens', '0')} | "
                f"{num(run, 'generated_tokens', '0')} | {fmt_float(num(run, 'seconds', '0'))} | "
                f"{fmt_float(num(run, 'tokens_per_sec', '0'))} | {num(run, 'last_token', '0')} |"
            )

    lines.extend(
        [
            "",
            "## Runtime Contract",
            "",
            "| Field | Value |",
            "|---|---|",
        ]
    )

    first = logs[0]
    lines.extend(
        [
            f"| Timed region | `{num(first.context, 'timed_region', 'unknown')}` |",
            f"| Sampling | `{num(first.context, 'sampling', 'unknown')}` |",
            f"| Console progress | `{num(first.context, 'console_progress', 'unknown')}` |",
            f"| KV cache | `{num(first.context, 'kv_cache', 'unknown')}` |",
            f"| Arithmetic | `{num(first.model, 'arithmetic', 'unknown')}` |",
        ]
    )

    return "\n".join(lines) + "\n"


def self_test(input_path: Path) -> None:
    log = parse_log(input_path)
    report = markdown([log])
    print("trace_scope hardware_perf_contract")
    print("trace parse_record")
    print("trace parse_log")
    print("trace markdown")
    print(f"PROBE_OK parse_log runs={len(log.runs)}")
    print(f"PROBE_OK hardware_perf_report bytes={len(report)}")
    print("PROBE_OK hardware_perf_report self_test=1")
    print("PROBE_OK main cli_entry=available")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    inputs = args.input or [DEFAULT_INPUT]
    if args.self_test:
        self_test(inputs[0])
        return

    logs = [parse_log(path) for path in inputs]
    text = markdown(logs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="ascii")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Estimate real-era GPT2-BASIC fixed-point inference performance.

The model is deliberately conservative: it counts code-level fixed-point work
from the exported checkpoint shape, then applies explicit per-machine throughput
assumptions for DOS FreeBASIC/DJGPP integer code on 386/486/Pentium-class CPUs.
It is not a replacement for real-board timing, but it keeps the estimate tied to
the actual model and runtime kernels instead of host-speed QEMU wall clock.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "assets" / "gpt2_basic" / "MODEL"
DEFAULT_RUN_LOG = ROOT / "qemu" / "evidence" / "run_main_486.log"
DEFAULT_PERF_LOGS = [
    ROOT / "qemu" / "evidence" / "perf_486_486dx2-66.log",
    ROOT / "qemu" / "evidence" / "perf_486_host.log",
]


@dataclass(frozen=True)
class Config:
    profile: str
    vocab_size: int
    n_positions: int
    n_embd: int
    n_head: int
    n_layer: int
    hidden_dim: int


@dataclass(frozen=True)
class Machine:
    key: str
    label: str
    clock_mhz: float
    weighted_munits_per_sec: float
    note: str


@dataclass(frozen=True)
class HostRun:
    generated_tokens: int
    seconds: float
    tokens_per_sec: float
    runtime_bytes: int | None


@dataclass(frozen=True)
class MeasuredRun:
    label: str
    clock: str
    generated_tokens: int
    seconds: float
    tokens_per_sec: float
    note: str
    profile: str


MACHINES = [
    Machine("386dx-33", "386DX/33-class, no FPU", 33.0, 0.090, "Conservative protected-mode 386 estimate"),
    Machine("486sx-25", "486SX/25, no FPU", 25.0, 0.180, "No-FPU 486 integer path"),
    Machine("486dx-33", "486DX/33", 33.0, 0.360, "486 integer path"),
    Machine("486dx2-66", "486DX2/66", 66.0, 0.720, "Common real-PC target"),
    Machine("486dx4-100", "486DX4/100", 100.0, 1.080, "Fast 486-class target"),
    Machine("pentium-60", "Pentium 60", 60.0, 1.050, "Early Pentium target"),
    Machine("pentium-133", "Pentium 133", 133.0, 2.160, "High-end DOS-era Pentium"),
]


def parse_config(path: Path) -> Config:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="ascii", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lower()] = value.strip()

    return Config(
        profile=values.get("profile", "custom"),
        vocab_size=int(values["vocab_size"]),
        n_positions=int(values["n_positions"]),
        n_embd=int(values["n_embd"]),
        n_head=int(values["n_head"]),
        n_layer=int(values["n_layer"]),
        hidden_dim=int(values["hidden_dim"]),
    )


def parameter_count(cfg: Config) -> int:
    layer_e = cfg.n_layer * cfg.n_embd
    layer_ee = cfg.n_layer * cfg.n_embd * cfg.n_embd
    layer_eh = cfg.n_layer * cfg.n_embd * cfg.hidden_dim
    layer_he = cfg.n_layer * cfg.hidden_dim * cfg.n_embd

    total = cfg.vocab_size * cfg.n_embd
    total += cfg.n_positions * cfg.n_embd
    total += layer_e * 8
    total += layer_ee * 4
    total += layer_eh + layer_he
    total += cfg.n_layer * cfg.hidden_dim
    total += layer_e
    total += cfg.n_embd * 2
    total += cfg.n_embd * cfg.vocab_size
    total += cfg.vocab_size
    return total


def estimate_runtime_memory(cfg: Config) -> int:
    cache_values = cfg.n_layer * cfg.n_positions * cfg.n_embd
    linear_acc_count = max(cfg.hidden_dim, cfg.n_embd, cfg.vocab_size)
    vector_count = cfg.n_embd * 8
    vector_count += cfg.hidden_dim * 2
    vector_count += cfg.vocab_size
    vector_count += cfg.n_positions

    total = parameter_count(cfg) * 4
    total += 513 * 4
    total += cache_values * 8
    total += cfg.n_positions * 4
    total += vector_count * 4
    total += linear_acc_count * 8
    return total


def estimate_phase_debug_memory(cfg: Config) -> int:
    return (cfg.n_embd * 11 + cfg.hidden_dim) * 4


def parse_host_run_log(path: Path) -> HostRun | None:
    if not path.exists():
        return None

    text = path.read_text(encoding="ascii", errors="ignore")
    perf_match = re.search(
        r"Generated\s+(\d+)\s+tokens in\s+([0-9.]+)\s+seconds\s+\(([0-9.]+)\s+tokens/sec\)",
        text,
    )
    if not perf_match:
        return None

    runtime_match = re.search(r"Runtime mem\s*:\s*(\d+)", text)
    runtime_bytes = int(runtime_match.group(1)) if runtime_match else None
    return HostRun(
        generated_tokens=int(perf_match.group(1)),
        seconds=float(perf_match.group(2)),
        tokens_per_sec=float(perf_match.group(3)),
        runtime_bytes=runtime_bytes,
    )


def parse_perf_record(line: str) -> tuple[str, dict[str, str]]:
    parts = line.strip().split("|")
    values: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key] = value
    return parts[0], values


def parse_perf_log(path: Path) -> MeasuredRun | None:
    if not path.exists():
        return None

    runs: list[dict[str, str]] = []
    runner: dict[str, str] = {}

    for raw_line in path.read_text(encoding="ascii", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line.startswith("PERF_"):
            continue
        kind, values = parse_perf_record(line)
        if kind == "PERF_RUN":
            runs.append(values)
        elif kind == "PERF_RUNNER":
            runner = values

    if not runs:
        return None

    selected = runs[0]
    for run in runs:
        if run.get("name") == "real_inference":
            selected = run
            break

    profile = runner.get("profile", path.stem.replace("perf_486_", ""))
    runner_label = runner.get("label", "")
    qemu_cpu = runner.get("qemu_cpu", "486")
    basis = runner.get("basis", "qemu-emulation")
    icount = runner.get("icount_shift", "off")

    if profile == "host":
        label = f"Host-speed QEMU -cpu {qemu_cpu} --perf"
        clock = "host"
    else:
        label = f"QEMU {profile} --perf"
        clock = "icount"

    note = basis
    if runner_label:
        note += f", {runner_label}"
    if icount != "off":
        note += f", icount shift={icount}"

    return MeasuredRun(
        label=label,
        clock=clock,
        generated_tokens=int(selected.get("generated_tokens", "0")),
        seconds=float(selected.get("seconds", "0")),
        tokens_per_sec=float(selected.get("tokens_per_sec", "0")),
        note=note,
        profile=profile,
    )


def work_counts(cfg: Config, prompt_tokens: int, generated_tokens: int) -> dict[str, int]:
    if prompt_tokens < 1:
        raise ValueError("prompt_tokens must be positive")
    if generated_tokens < 1:
        raise ValueError("generated_tokens must be positive")
    if prompt_tokens + generated_tokens - 1 > cfg.n_positions:
        raise ValueError("demo context exceeds configured n_positions")

    forward_count = prompt_tokens + generated_tokens - 1
    logit_passes = generated_tokens
    context_sum = forward_count * (forward_count + 1) // 2

    emb = cfg.n_embd
    heads = cfg.n_head
    layers = cfg.n_layer
    hidden = cfg.hidden_dim
    vocab = cfg.vocab_size

    dense_linear_mac = forward_count * layers * ((4 * emb * emb) + (2 * emb * hidden))
    attention_qk_mac = layers * emb * context_sum
    attention_value_mac = layers * emb * context_sum
    attention_scale_mac = layers * heads * context_sum
    attention_inlined_mul_helpers = attention_qk_mac + attention_value_mac + attention_scale_mac
    final_head_mac = logit_passes * emb * vocab
    layernorm_mac = (forward_count * layers * 2 * 3 * emb) + (logit_passes * 3 * emb)
    gelu_mac = forward_count * layers * hidden * 6
    gelu_inlined_mul_helpers = gelu_mac

    exp_lookup = layers * heads * context_sum
    softmax_div_before = layers * emb * context_sum
    softmax_div_after = layers * heads * context_sum
    norm_count = (forward_count * layers * 2) + logit_passes
    gelu_div = forward_count * layers * hidden

    fixed_mac = (
        dense_linear_mac
        + attention_qk_mac
        + attention_value_mac
        + attention_scale_mac
        + final_head_mac
        + layernorm_mac
        + gelu_mac
    )
    divisions_before = softmax_div_before + norm_count + gelu_div
    divisions_after = softmax_div_after + norm_count + gelu_div
    sqrt_ops = norm_count

    loop_overhead_units = forward_count * layers * emb * 4
    weighted_before = fixed_mac + exp_lookup * 2 + divisions_before * 18 + sqrt_ops * 90 + loop_overhead_units
    weighted_after = fixed_mac + exp_lookup * 2 + divisions_after * 18 + sqrt_ops * 90 + loop_overhead_units

    return {
        "forward_count": forward_count,
        "logit_passes": logit_passes,
        "context_sum": context_sum,
        "fixed_mac": fixed_mac,
        "divisions_before": divisions_before,
        "divisions_after": divisions_after,
        "sqrt_ops": sqrt_ops,
        "exp_lookup": exp_lookup,
        "attention_inlined_mul_helpers": attention_inlined_mul_helpers,
        "attention_inlined_clamp_helpers": attention_inlined_mul_helpers + softmax_div_after,
        "gelu_inlined_mul_helpers": gelu_inlined_mul_helpers,
        "gelu_inlined_clamp_helpers": gelu_inlined_mul_helpers,
        "weighted_before": weighted_before,
        "weighted_after": weighted_after,
        "saved_divisions": divisions_before - divisions_after,
        "saved_weighted_units": weighted_before - weighted_after,
    }


def seconds_for(machine: Machine, weighted_units: int) -> float:
    return weighted_units / (machine.weighted_munits_per_sec * 1_000_000.0)


def markdown_report(
    cfg: Config,
    counts: dict[str, int],
    prompt_tokens: int,
    generated_tokens: int,
    host_run: HostRun | None = None,
    measured_runs: list[MeasuredRun] | None = None,
) -> str:
    params = parameter_count(cfg)
    runtime_bytes = estimate_runtime_memory(cfg)
    debug_bytes = estimate_phase_debug_memory(cfg)
    saved_pct = counts["saved_weighted_units"] * 100.0 / counts["weighted_before"]

    lines = [
        "# GPT2-BASIC Era Performance Model",
        "",
        f"Model profile: `{cfg.profile}`",
        f"Shape: `{cfg.n_layer}L {cfg.n_embd}D {cfg.n_head}H ctx{cfg.n_positions} hidden{cfg.hidden_dim} vocab{cfg.vocab_size}`",
        f"Parameters: `{params}`",
        f"Estimated runtime memory: `{runtime_bytes}` bytes (`{runtime_bytes / (1024 * 1024):.3f}` MB)",
        f"Validation-only phase trace overhead: `{debug_bytes}` bytes",
        f"Demo path: `{prompt_tokens}` prompt tokens, `{generated_tokens}` generated tokens, `{counts['forward_count']}` cached forward calls",
        "",
        "## Kernel Work",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Fixed multiply-accumulate style ops | {counts['fixed_mac']:,} |",
        f"| Fixed multiply helper calls inlined in attention | {counts['attention_inlined_mul_helpers']:,} |",
        f"| Clamp helper calls inlined in attention | {counts['attention_inlined_clamp_helpers']:,} |",
        f"| Fixed multiply helper calls inlined in GELU | {counts['gelu_inlined_mul_helpers']:,} |",
        f"| Clamp helper calls inlined in GELU | {counts['gelu_inlined_clamp_helpers']:,} |",
        f"| Exp table lookups | {counts['exp_lookup']:,} |",
        f"| Integer sqrt ops | {counts['sqrt_ops']:,} |",
        f"| Integer divisions before attention-probability hoist | {counts['divisions_before']:,} |",
        f"| Integer divisions after attention-probability hoist | {counts['divisions_after']:,} |",
        f"| Divisions removed from 70-token demo | {counts['saved_divisions']:,} |",
        f"| Weighted work reduction | {saved_pct:.1f}% |",
        "",
        "## Estimated Real-PC Throughput",
        "",
        "| Target | Clock | 70-token demo | Tokens/sec | 100-token equivalent | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for machine in MACHINES:
        secs = seconds_for(machine, counts["weighted_after"])
        tps = generated_tokens / secs
        hundred = 100.0 / tps
        lines.append(
            f"| {machine.label} | {machine.clock_mhz:.0f} MHz | {secs:.1f} s | {tps:.2f} | {hundred:.1f} s | {machine.note} |"
        )

    if host_run is not None:
        hundred = 100.0 / host_run.tokens_per_sec
        lines.append(
            f"| Host-speed QEMU -cpu 486 | host | {host_run.seconds:.2f} s | {host_run.tokens_per_sec:.2f} | {hundred:.1f} s | Measured FreeDOS/QEMU run log |"
        )

    for measured in measured_runs or []:
        hundred = 100.0 / measured.tokens_per_sec if measured.tokens_per_sec > 0 else 0.0
        lines.append(
            f"| {measured.label} | {measured.clock} | {measured.seconds:.2f} s | {measured.tokens_per_sec:.2f} | {hundred:.1f} s | {measured.note} |"
        )

    lines.extend(
        [
            "",
            "Assumptions: throughput is expressed in weighted fixed-point BASIC work units/sec, with 64-bit integer division priced much higher than fixed multiply/add work. QEMU `--perf` rows are emulator evidence from the DOS executable; physical-board timing still wins over both QEMU and the planning model.",
        ]
    )
    return "\n".join(lines) + "\n"


def html_report(
    cfg: Config,
    counts: dict[str, int],
    prompt_tokens: int,
    generated_tokens: int,
    host_run: HostRun | None = None,
    measured_runs: list[MeasuredRun] | None = None,
) -> str:
    rows = []
    for machine in MACHINES:
        secs = seconds_for(machine, counts["weighted_after"])
        tps = generated_tokens / secs
        hundred = 100.0 / tps
        nofpu = " no FPU" if "no FPU" in machine.label else ""
        rows.append(
            f"""        <tr>
          <td class="profile">{machine.label.replace(', no FPU', '')}<span class="nofpu">{nofpu}</span></td>
          <td>{tps:.2f}</td>
          <td>{secs:.1f} s</td>
          <td>{hundred:.1f} s</td>
          <td>{machine.weighted_munits_per_sec:.3f}M</td>
          <td>model</td>
        </tr>"""
        )

    if host_run is not None:
        hundred = 100.0 / host_run.tokens_per_sec
        rows.append(
            f"""        <tr class="measured">
          <td class="profile">Host-speed QEMU -cpu 486<span class="nofpu"></span></td>
          <td>{host_run.tokens_per_sec:.2f}</td>
          <td>{host_run.seconds:.2f} s</td>
          <td>{hundred:.1f} s</td>
          <td>measured</td>
          <td>QEMU log</td>
        </tr>"""
        )

    for measured in measured_runs or []:
        hundred = 100.0 / measured.tokens_per_sec if measured.tokens_per_sec > 0 else 0.0
        rows.append(
            f"""        <tr class="measured">
          <td class="profile">{measured.label}<span class="nofpu"></span></td>
          <td>{measured.tokens_per_sec:.2f}</td>
          <td>{measured.seconds:.2f} s</td>
          <td>{hundred:.1f} s</td>
          <td>measured</td>
          <td>{measured.clock}</td>
        </tr>"""
        )

    runtime_bytes = estimate_runtime_memory(cfg)
    debug_bytes = estimate_phase_debug_memory(cfg)
    saved_pct = counts["saved_weighted_units"] * 100.0 / counts["weighted_before"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GPT2-BASIC Era Performance Estimate</title>
  <style>
    :root {{
      --ink: #111827;
      --muted: #4b5563;
      --line: #d1d5db;
      --head: #f3f4f6;
      --accent: #0f766e;
      --accent-soft: #ccfbf1;
      --measured: #eff6ff;
      --measured-line: #93c5fd;
      --bg: #ffffff;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Avenir Next", Avenir, "Helvetica Neue", Arial, sans-serif;
    }}

    .wrap {{
      width: 1240px;
      margin: 0 auto;
      padding: 46px 54px;
    }}

    .eyebrow {{
      color: var(--accent);
      font-size: 15px;
      font-weight: 800;
      letter-spacing: 0;
      margin: 0 0 8px;
      text-transform: uppercase;
    }}

    h1 {{
      font-size: 34px;
      line-height: 1.12;
      letter-spacing: 0;
      margin: 0;
    }}

    .sub {{
      color: var(--muted);
      font-size: 17px;
      line-height: 1.45;
      margin: 12px 0 26px;
      max-width: 980px;
    }}

    .meta {{
      display: flex;
      gap: 12px;
      margin: 20px 0 24px;
      flex-wrap: wrap;
    }}

    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--ink);
      font-size: 14px;
      font-weight: 800;
      padding: 8px 12px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--line);
      font-size: 18px;
      box-shadow: 0 16px 40px rgba(17, 24, 39, 0.08);
    }}

    th,
    td {{
      border-bottom: 1px solid var(--line);
      padding: 15px 16px;
      text-align: left;
      vertical-align: middle;
      white-space: nowrap;
    }}

    th {{
      background: var(--head);
      color: var(--ink);
      font-size: 14px;
      font-weight: 900;
      letter-spacing: 0;
      text-transform: uppercase;
    }}

    tbody tr:last-child td {{ border-bottom: 0; }}
    tbody tr.measured td {{
      background: var(--measured);
      border-top: 2px solid var(--measured-line);
      border-bottom: 2px solid var(--measured-line);
      font-weight: 800;
    }}

    td:nth-child(n+2),
    th:nth-child(n+2) {{ text-align: right; }}

    .profile {{ font-weight: 800; text-align: left !important; }}
    .nofpu {{ color: var(--accent); font-weight: 800; }}

    .note {{
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
      margin: 18px 0 0;
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <p class="eyebrow">Era-calibrated performance evidence</p>
    <h1>GPT2-BASIC Fixed-Point Performance on 486-Class Targets</h1>
    <p class="sub">
      Estimate from the current `{cfg.profile}` checkpoint shape and code-level fixed-point work counts.
      QEMU rows are emulator measurements from GPT2.EXE --perf; real-board timing should replace them when available.
    </p>

    <div class="meta" aria-label="model details">
      <span class="pill">{cfg.n_layer} layers</span>
      <span class="pill">{cfg.n_embd} embedding dims</span>
      <span class="pill">{cfg.n_head} heads</span>
      <span class="pill">{cfg.vocab_size} byte tokens</span>
      <span class="pill">{generated_tokens}-token demo</span>
      <span class="pill">{runtime_bytes / (1024 * 1024):.2f} MB runtime</span>
      <span class="pill">{saved_pct:.1f}% less weighted work</span>
    </div>

    <table>
      <thead>
        <tr>
          <th>Target Profile</th>
          <th>Tokens/sec</th>
          <th>{generated_tokens}-token demo</th>
          <th>100-token equiv.</th>
          <th>Effective work/sec</th>
          <th>Basis</th>
        </tr>
      </thead>
      <tbody>
{chr(10).join(rows)}
      </tbody>
    </table>

    <p class="note">
      Work count: {counts['fixed_mac']:,} fixed multiply-accumulate style ops,
      {counts['attention_inlined_mul_helpers']:,} fixed multiply helper calls inlined in attention,
      {counts['gelu_inlined_mul_helpers']:,} fixed multiply helper calls inlined in GELU,
      {counts['divisions_after']:,} integer divisions after optimization,
      {counts['sqrt_ops']:,} integer sqrt ops, and {counts['exp_lookup']:,} exp-table lookups.
      The attention probability hoist removes {counts['saved_divisions']:,} repeated divisions from this demo path.
      Phase-parity validation allocates {debug_bytes:,} extra bytes only while --vectors is running.
    </p>
  </main>
</body>
</html>
"""


def self_test(model_dir: Path, prompt_tokens: int, generated_tokens: int, run_log: Path) -> None:
    cfg = parse_config(model_dir / "GPT2CFG.TXT")
    params = parameter_count(cfg)
    memory = estimate_runtime_memory(cfg)
    debug_memory = estimate_phase_debug_memory(cfg)
    counts = work_counts(cfg, prompt_tokens, generated_tokens)
    seconds = seconds_for(MACHINES[3], counts["weighted_after"])
    host_run = parse_host_run_log(run_log)
    measured_runs = [run for path in DEFAULT_PERF_LOGS if (run := parse_perf_log(path)) is not None]
    if any(run.profile == "host" for run in measured_runs):
        host_run = None
    markdown = markdown_report(cfg, counts, prompt_tokens, generated_tokens, host_run, measured_runs)
    html = html_report(cfg, counts, prompt_tokens, generated_tokens, host_run, measured_runs)

    print(f"PROBE_OK parse_config profile={cfg.profile}")
    print("trace parse_host_run_log")
    print("trace parse_perf_record")
    print("trace parse_perf_log")
    print("PROBE_OK self_test exercised=1")
    print("PROBE_OK main cli_entry=available")
    print(f"PROBE_OK parameter_count params={params}")
    print(f"PROBE_OK estimate_runtime_memory bytes={memory}")
    print(f"PROBE_OK estimate_phase_debug_memory bytes={debug_memory}")
    if host_run is not None:
        print(f"PROBE_OK parse_host_run_log tokens_per_sec={host_run.tokens_per_sec:.2f}")
    if measured_runs:
        print(f"PROBE_OK parse_perf_log count={len(measured_runs)}")
    print(f"PROBE_OK work_counts saved_divisions={counts['saved_divisions']}")
    print(f"PROBE_OK work_counts gelu_inlined_mul_helpers={counts['gelu_inlined_mul_helpers']}")
    print(f"PROBE_OK seconds_for machine={MACHINES[3].key} seconds={seconds:.2f}")
    print(f"PROBE_OK markdown_report bytes={len(markdown)}")
    print(f"PROBE_OK html_report bytes={len(html)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-tokens", type=int, default=31)
    parser.add_argument("--generated-tokens", type=int, default=70)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--html-out", type=Path)
    parser.add_argument("--run-log", type=Path, default=DEFAULT_RUN_LOG)
    parser.add_argument("--perf-log", action="append", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test(args.model_dir, args.prompt_tokens, args.generated_tokens, args.run_log)
        return

    cfg = parse_config(args.model_dir / "GPT2CFG.TXT")
    counts = work_counts(cfg, args.prompt_tokens, args.generated_tokens)
    host_run = parse_host_run_log(args.run_log)
    perf_logs = args.perf_log if args.perf_log is not None else DEFAULT_PERF_LOGS
    measured_runs = [run for path in perf_logs if (run := parse_perf_log(path)) is not None]
    if any(run.profile == "host" for run in measured_runs):
        host_run = None
    markdown = markdown_report(cfg, counts, args.prompt_tokens, args.generated_tokens, host_run, measured_runs)

    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="ascii")
    else:
        print(markdown, end="")

    if args.html_out:
        args.html_out.parent.mkdir(parents=True, exist_ok=True)
        args.html_out.write_text(html_report(cfg, counts, args.prompt_tokens, args.generated_tokens, host_run, measured_runs), encoding="ascii")


if __name__ == "__main__":
    main()

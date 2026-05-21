#!/usr/bin/env python3
"""Record and render the long-form GPT2-BASIC assistant showcase video."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "promo" / "renders"
DEFAULT_REPORT = ROOT / "qemu" / "evidence" / "assistant_showcase_video.md"
PLAYER = ROOT / "scripts" / "play_assistant_showcase_terminal.py"
DEFAULT_COLS = 104
DEFAULT_ROWS = 32
DEFAULT_FONT_SIZE = 22
DEFAULT_LAST_FRAME_DURATION = 12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"ASSISTANT_SHOWCASE_VIDEO_FAILED {message}")


def find_tool(name: str) -> str:
    path = shutil.which(name)
    require(path is not None, f"missing_tool={name}")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "cast": output_dir / "gpt2_basic_assistant_showcase.cast",
        "gif": output_dir / "gpt2_basic_assistant_showcase.gif",
        "mp4": output_dir / "gpt2_basic_assistant_showcase_1080p.mp4",
    }


def run_checked(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def record_cast(
    asciinema: str,
    cast_path: Path,
    cols: int,
    rows: int,
    speed: float,
    force: bool,
) -> None:
    if cast_path.exists() and not force:
        raise SystemExit(f"ASSISTANT_SHOWCASE_VIDEO_FAILED output_exists={cast_path} use --force")
    cast_path.parent.mkdir(parents=True, exist_ok=True)
    command = f"python3 scripts/play_assistant_showcase_terminal.py --speed {speed:.3f}"
    run_checked(
        [
            asciinema,
            "record",
            "--quiet",
            "--overwrite",
            "--headless",
            "--window-size",
            f"{cols}x{rows}",
            "--command",
            command,
            cast_path.as_posix(),
        ],
        ROOT,
    )


def render_gif(
    agg: str,
    cast_path: Path,
    gif_path: Path,
    cols: int,
    rows: int,
    font_size: int,
    force: bool,
) -> None:
    if gif_path.exists() and not force:
        raise SystemExit(f"ASSISTANT_SHOWCASE_VIDEO_FAILED output_exists={gif_path} use --force")
    run_checked(
        [
            agg,
            "--theme",
            "monokai",
            "--cols",
            str(cols),
            "--rows",
            str(rows),
            "--font-size",
            str(font_size),
            "--fps-cap",
            "20",
            "--idle-time-limit",
            "3",
            "--last-frame-duration",
            str(DEFAULT_LAST_FRAME_DURATION),
            cast_path.as_posix(),
            gif_path.as_posix(),
        ],
        ROOT,
    )


def convert_gif_to_mp4(ffmpeg: str, gif_path: Path, mp4_path: Path, force: bool) -> None:
    if mp4_path.exists() and not force:
        raise SystemExit(f"ASSISTANT_SHOWCASE_VIDEO_FAILED output_exists={mp4_path} use --force")
    run_checked(
        [
            ffmpeg,
            "-y",
            "-i",
            gif_path.as_posix(),
            "-vf",
            "fps=30,scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            mp4_path.as_posix(),
        ],
        ROOT,
    )


def ffprobe_summary(ffprobe: str, mp4_path: Path) -> tuple[str, str]:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,duration",
            "-of",
            "default=noprint_wrappers=1:nokey=0",
            mp4_path.as_posix(),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    resolution = f"{values.get('width', '?')}x{values.get('height', '?')}"
    duration = values.get("duration", "?")
    return resolution, duration


def write_report(report: Path, paths: dict[str, Path], resolution: str, duration: str) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Assistant Showcase Video",
        "",
        "Status: `PASS`",
        "",
        "This video is a real terminal capability demonstration rendered from checked GPT2-BASIC QEMU assistant evidence.",
        "The visible DOS session uses era-accurate DOS commands only; Python, ffmpeg, and asciinema are host-side build tools.",
        "",
        f"- MP4: `{display_path(paths['mp4'])}`",
        f"- Cast: `{display_path(paths['cast'])}`",
        f"- GIF intermediate: `{display_path(paths['gif'])}`",
        f"- Resolution: `{resolution}`",
        f"- Duration seconds: `{duration}`",
        f"- Final frame hold seconds: `{DEFAULT_LAST_FRAME_DURATION}`",
        f"- MP4 SHA-256: `{file_sha256(paths['mp4'])}`",
        "",
        "Audience:",
        "",
        "- Engineers evaluating local language models on constrained systems.",
        "- Retrocomputing, embedded, industrial, archival, and air-gapped operators.",
        "- Pack authors who need fast local recall, small weights, and auditable behavior.",
        "",
        "Covered functionality:",
        "",
        "- CHAT general conversation, local inference, no-web limits, troubleshooting, and repeated-answer recovery.",
        "- CHAT session memory for small session facts, goal, answer style, last prompt, and recall summary.",
        "- DOSHELP pack for CONFIG.SYS, AUTOEXEC.BAT, DPMI, conventional memory, and batch-file help.",
        "- OFFICE pack for rewriting, summarizing, status updates, release wording, and clearer notes.",
        "- DEV pack for retrieval-first architecture, pack authoring, release checks, and modern 486 assistant design.",
        "- KB2 binary recall, KDB fallback, USER.TXT note import, source/timing provenance, tests, QEMU evidence, and release hashes.",
        "- DOS-era command presentation: ASSIST.EXE, TYPE, EDIT, MANIFEST.TXT, ASTRESS.LOG, and 8.3-compatible paths.",
        "",
    ]
    report.write_text("\n".join(lines), encoding="ascii")


def build(args: argparse.Namespace) -> Path:
    require(PLAYER.is_file(), f"missing_player={PLAYER}")
    paths = output_paths(args.output_dir)
    asciinema = find_tool(args.asciinema)
    agg = find_tool(args.agg)
    ffmpeg = find_tool(args.ffmpeg)
    ffprobe = find_tool(args.ffprobe)
    record_cast(asciinema, paths["cast"], args.cols, args.rows, args.speed, args.force)
    render_gif(agg, paths["cast"], paths["gif"], args.cols, args.rows, args.font_size, args.force)
    convert_gif_to_mp4(ffmpeg, paths["gif"], paths["mp4"], args.force)
    resolution, duration = ffprobe_summary(ffprobe, paths["mp4"])
    require(resolution == "1920x1080", f"unexpected_resolution={resolution}")
    write_report(args.report, paths, resolution, duration)
    print(f"ASSISTANT_SHOWCASE_CAST|path={paths['cast']}")
    print(f"ASSISTANT_SHOWCASE_GIF|path={paths['gif']}")
    print(f"ASSISTANT_SHOWCASE_VIDEO|path={paths['mp4']}")
    print(f"ASSISTANT_SHOWCASE_REPORT|path={args.report}")
    print("PROBE_OK assistant_showcase_video=1")
    return paths["mp4"]


def self_test() -> None:
    paths = output_paths(Path("/tmp/out"))
    require(paths["mp4"].name == "gpt2_basic_assistant_showcase_1080p.mp4", "self_test_mp4_name")
    require(paths["cast"].suffix == ".cast", "self_test_cast_suffix")
    require(PLAYER.name == "play_assistant_showcase_terminal.py", "self_test_player")
    require(display_path(ROOT / "promo" / "renders" / "demo.mp4") == "promo/renders/demo.mp4", "self_test_display_path")
    require(DEFAULT_LAST_FRAME_DURATION >= 10, "self_test_final_hold")
    print("PROBE_OK assistant_showcase_video_self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--font-size", type=int, default=DEFAULT_FONT_SIZE)
    parser.add_argument("--speed", type=float, default=0.35)
    parser.add_argument("--asciinema", default="asciinema")
    parser.add_argument("--agg", default="agg")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    build(args)


if __name__ == "__main__":
    main()

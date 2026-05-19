#!/usr/bin/env python3
"""Build ffmpeg-rendered promotional cards and short videos.

The generated media is intentionally derived from repo text/evidence so the
first public assets stay honest: title cards, quality claims, and example CHAT
outputs come from the checked-in project state. Rendered videos are local build
outputs and should be published as release/video-platform assets, not committed
to git.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - depends on local media tooling.
    raise SystemExit("PROMO_MATERIALS_FAILED pillow_missing install Pillow") from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "promo" / "renders"
DEFAULT_REPORT = ROOT / "qemu" / "evidence" / "quality_report_assistant_chat.md"
DEFAULT_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Monaco.ttf"),
    Path("/System/Library/Fonts/Menlo.ttc"),
    Path("/System/Library/Fonts/Supplemental/Andale Mono.ttf"),
    Path("/System/Library/Fonts/Courier.ttc"),
)
DEFAULT_FPS = 30
DOS_COLS = 80
DOS_ROWS = 25


@dataclass(frozen=True)
class Scene:
    name: str
    eyebrow: str
    title: str
    subtitle: str
    terminal: str
    footer: str
    duration: float


@dataclass(frozen=True)
class TerminalExchange:
    prompt: str
    answer: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"PROMO_MATERIALS_FAILED {message}")


def resolve_font(explicit: Path | None) -> Path:
    if explicit is not None:
        require(explicit.exists(), f"font_missing={explicit}")
        return explicit
    for candidate in DEFAULT_FONT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise SystemExit("PROMO_MATERIALS_FAILED no_usable_font")


def ffmpeg_command(ffmpeg: str) -> str:
    path = shutil.which(ffmpeg)
    require(path is not None, f"ffmpeg_missing={ffmpeg}")
    return path


def read_quality_examples(report_path: Path) -> dict[str, str]:
    if not report_path.exists():
        return {}
    text = report_path.read_text(encoding="ascii", errors="ignore")
    examples: dict[str, str] = {}
    pattern = re.compile(r"^### (chat_[^\n]+)\n.*?```text\n(.*?)\n```", re.M | re.S)
    for match in pattern.finditer(text):
        examples[match.group(1).strip()] = " ".join(match.group(2).strip().split())
    return examples


def example_line(examples: dict[str, str], key: str, fallback: str) -> str:
    return examples.get(key, fallback)


def wrap_lines(text: str, width: int) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        if not raw.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw, width=width, break_long_words=False, break_on_hyphens=False) or [""])
    return "\n".join(lines)


def teaser_scenes(examples: dict[str, str]) -> list[Scene]:
    real = example_line(
        examples,
        "chat_are_you_real",
        "This is real local inference in DOS. The answer comes from local model weights.",
    )
    prompt = example_line(
        examples,
        "chat_what_is_a_prompt",
        "A prompt is your typed question. That is the simple version.",
    )
    focus = example_line(
        examples,
        "chat_how_do_i_focus",
        "Remove one distraction and choose one task. Start small, then adjust after the first step.",
    )
    bored = example_line(
        examples,
        "chat_suggest_something_to_do",
        "Try one small project. Type a question and I will answer.",
    )
    return [
        Scene(
            "01_title",
            "GPT2-BASIC",
            "Transformer Inference In DOS",
            "A tiny GPT-style model running through a FreeBASIC fixed-point runtime.",
            "C:\\GPT2> ASSIST\nPack: CHAT\nModel: 3L 64D 4H ctx192 vocab4096\nQuality: 48/48 fixed-point pass",
            "Real inference. QEMU-verified preview. Physical hardware timing next.",
            4.0,
        ),
        Scene(
            "02_real",
            "LIVE CHAT EXAMPLE",
            "Not A Script",
            "The assistant answers from exported model weights.",
            f"C:\\GPT2> are you real\n{real}",
            "Show the latency honestly. The constraints are the story.",
            4.5,
        ),
        Scene(
            "03_prompt",
            "WHAT IT CAN ANSWER",
            "Short DOS Chat",
            "Two brief sentences, clean boundaries, no prompt echo.",
            f"C:\\GPT2> what is a prompt\n{prompt}",
            "CHAT quality report: 48/48, average 1.000.",
            4.5,
        ),
        Scene(
            "04_focus",
            "PRACTICAL PROMPTS",
            "Small But Useful",
            "The model is intentionally modest and inspectable.",
            f"C:\\GPT2> how do i focus\n{focus}\n\nC:\\GPT2> suggest something to do\n{bored}",
            "Small GPT-style model, real fixed-point runtime.",
            6.0,
        ),
        Scene(
            "05_release",
            "PUBLIC PREVIEW",
            "Source, Evidence, Release Zips",
            "Rebuild scripts, tests, QEMU evidence, and hardware-transfer tooling.",
            "python3 scripts/build_preview_release.py --force\npython3 scripts/verify_preview_artifacts.py\n\nArtifacts: preview zip + hardware-transfer zip",
            "Repo link and release link go here after publish.",
            5.0,
        ),
    ]


def terminal_exchanges(examples: dict[str, str]) -> list[TerminalExchange]:
    return [
        TerminalExchange(
            "are you real",
            example_line(
                examples,
                "chat_are_you_real",
                "This is real local inference in DOS. The answer comes from local model weights.",
            ),
        ),
        TerminalExchange(
            "what is a prompt",
            example_line(
                examples,
                "chat_what_is_a_prompt",
                "A prompt is your typed question. That is the simple version.",
            ),
        ),
        TerminalExchange(
            "help me decide",
            example_line(
                examples,
                "chat_help_me_decide",
                "Name the choices and the goal. Start small, then adjust after the first step.",
            ),
        ),
        TerminalExchange(
            "how do i focus",
            example_line(
                examples,
                "chat_how_do_i_focus",
                "Remove one distraction and choose one task. Start small, then adjust after the first step.",
            ),
        ),
    ]


def load_font(font: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font.as_posix(), size=size)


def draw_multiline(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str,
    spacing: int,
) -> None:
    draw.multiline_text(xy, text, font=font, fill=fill, spacing=spacing)


def render_card(
    ffmpeg: str,
    font: Path,
    scene: Scene,
    output: Path,
    size: tuple[int, int],
    text_dir: Path,
) -> None:
    del ffmpeg, text_dir
    width, height = size
    terminal_width = 74 if width > height else 36
    title_size = 68 if width > height else 58
    subtitle_size = 32 if width > height else 34
    terminal_size = 34 if width > height else 30
    footer_size = 24 if width > height else 24
    margin = 88 if width > height else 64
    box_y = 330 if width > height else 560
    box_h = height - box_y - 150

    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height), "#05070a")
    draw = ImageDraw.Draw(image)

    draw.rectangle((margin, box_y, width - margin, box_y + box_h), fill="#0b1116", outline="#66ff99", width=3)
    draw.rectangle((margin, box_y - 36, width - margin, box_y), fill="#12301e")

    draw_multiline(draw, (margin, 82), scene.eyebrow, load_font(font, 28), "#66ff99", 8)
    draw_multiline(draw, (margin, 130), scene.title, load_font(font, title_size), "#e8fff0", 8)
    draw_multiline(
        draw,
        (margin, 232),
        wrap_lines(scene.subtitle, 54 if width > height else 30),
        load_font(font, subtitle_size),
        "#aad8ff",
        10,
    )
    draw_multiline(
        draw,
        (margin + 36, box_y + 48),
        wrap_lines(scene.terminal, terminal_width),
        load_font(font, terminal_size),
        "#d7ffe7",
        12,
    )
    draw_multiline(
        draw,
        (margin, height - 92),
        wrap_lines(scene.footer, 74 if width > height else 36),
        load_font(font, footer_size),
        "#9db7a7",
        8,
    )
    image.save(output)


def render_video(ffmpeg: str, cards: list[tuple[Path, float]], output: Path, work_dir: Path) -> None:
    concat_path = work_dir / f"{output.stem}_concat.txt"
    lines: list[str] = []
    for card, duration in cards:
        lines.append(f"file '{card.as_posix()}'")
        lines.append(f"duration {duration:.3f}")
    lines.append(f"file '{cards[-1][0].as_posix()}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_path.as_posix(),
            "-vf",
            "fps=30,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            output.as_posix(),
        ],
        check=True,
    )


def terminal_wrap(text: str, cols: int) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        if raw == "":
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw, width=cols, break_long_words=True, replace_whitespace=False) or [""])
    return lines


def draw_terminal_frame(
    size: tuple[int, int],
    font: Path,
    text: str,
    cursor: bool,
    title: str,
) -> Image.Image:
    width, height = size
    vertical = height > width
    margin = 58 if vertical else 78
    top = 80 if vertical else 70
    bottom = 58 if vertical else 54
    title_size = 31 if vertical else 32
    font_size = 28 if vertical else 31
    line_gap = 9 if vertical else 8
    term_font = load_font(font, font_size)
    title_font = load_font(font, title_size)
    char_width = max(1, term_font.getlength("M"))
    line_height = (term_font.getbbox("Mg")[3] - term_font.getbbox("Mg")[1]) + line_gap
    cols = max(24, int((width - margin * 2 - 48) / char_width))
    rows = max(8, int((height - top - bottom - 72) / line_height))

    display_text = text + ("█" if cursor else "")
    wrapped = terminal_wrap(display_text, cols)
    visible = wrapped[-rows:]

    image = Image.new("RGB", (width, height), "#020402")
    draw = ImageDraw.Draw(image)
    draw.rectangle((margin, top, width - margin, height - bottom), fill="#050b08", outline="#38ff7a", width=3)
    draw.rectangle((margin, top, width - margin, top + 42), fill="#11301d")
    draw.text((margin + 20, top + 8), title, font=title_font, fill="#65ff9d")
    right_label = "DOS CHAT"
    right_width = draw.textlength(right_label, font=title_font)
    draw.text((width - margin - right_width - 20, top + 8), right_label, font=title_font, fill="#9db7a7")

    y = top + 66
    for line in visible:
        color = "#d8ffe0"
        if line.startswith("Thinking"):
            color = "#aad8ff"
        elif line.startswith("GPT2-BASIC") or line.startswith("Quality"):
            color = "#65ff9d"
        draw.text((margin + 28, y), line, font=term_font, fill=color)
        y += line_height
    return image


def append_terminal_frames(
    process: subprocess.Popen[bytes],
    size: tuple[int, int],
    font: Path,
    text: str,
    duration: float,
    title: str,
    fps: int,
) -> None:
    assert process.stdin is not None
    frame_count = max(1, int(duration * fps))
    for idx in range(frame_count):
        cursor = ((idx // max(1, fps // 2)) % 2) == 0
        frame = draw_terminal_frame(size, font, text, cursor, title)
        process.stdin.write(frame.tobytes())


def stream_terminal_text(
    process: subprocess.Popen[bytes],
    size: tuple[int, int],
    font: Path,
    base_text: str,
    added_text: str,
    chars_per_second: float,
    title: str,
    fps: int,
) -> str:
    assert process.stdin is not None
    text = base_text
    frames_per_char = max(1, int(round(fps / chars_per_second)))
    for char_idx, ch in enumerate(added_text):
        text += ch
        for frame_idx in range(frames_per_char):
            cursor = (((char_idx * frames_per_char + frame_idx) // max(1, fps // 2)) % 2) == 0
            frame = draw_terminal_frame(size, font, text, cursor, title)
            process.stdin.write(frame.tobytes())
    return text


def open_rawvideo_encoder(ffmpeg: str, output: Path, size: tuple[int, int], fps: int) -> subprocess.Popen[bytes]:
    width, height = size
    output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output.as_posix(),
        ],
        stdin=subprocess.PIPE,
    )


def render_terminal_demo(
    ffmpeg: str,
    font: Path,
    exchanges: list[TerminalExchange],
    output: Path,
    size: tuple[int, int],
    fps: int = DEFAULT_FPS,
) -> None:
    process = open_rawvideo_encoder(ffmpeg, output, size, fps)
    try:
        title = "GPT2-BASIC ASSIST.EXE"
        text = (
            "GPT2-BASIC assistant pack demo\n"
            "Quality: CHAT fixed-point gate PASS 48/48\n"
            "Mode: generated terminal promo from checked-in evidence\n\n"
        )
        append_terminal_frames(process, size, font, text, 1.2, title, fps)
        for idx, exchange in enumerate(exchanges):
            text = stream_terminal_text(
                process,
                size,
                font,
                text + "C:\\GPT2> ",
                exchange.prompt,
                chars_per_second=18.0,
                title=title,
                fps=fps,
            )
            text += "\n"
            append_terminal_frames(process, size, font, text + "Thinking: sampling output tokens", 0.65, title, fps)
            text += "Answer: "
            text = stream_terminal_text(
                process,
                size,
                font,
                text,
                exchange.answer,
                chars_per_second=30.0,
                title=title,
                fps=fps,
            )
            text += "\n\n"
            append_terminal_frames(process, size, font, text, 0.55 if idx < len(exchanges) - 1 else 1.4, title, fps)
        assert process.stdin is not None
        process.stdin.close()
        status = process.wait()
    finally:
        if process.poll() is None:
            process.kill()
    if status != 0:
        raise SystemExit(f"PROMO_MATERIALS_FAILED terminal_demo_ffmpeg_status={status}")


def dos_wrap_text(text: str, cols: int = DOS_COLS) -> list[str]:
    rows: list[str] = []
    for raw in text.splitlines():
        if raw == "":
            rows.append("")
            continue
        line = raw
        while len(line) > cols:
            rows.append(line[:cols])
            line = line[cols:]
        rows.append(line)
    return rows


def draw_real_dos_frame(size: tuple[int, int], font: Path, text: str, cursor: bool) -> Image.Image:
    width, height = size
    vertical = height > width
    font_size = 18 if vertical else 31
    line_gap = 3 if vertical else 7
    term_font = load_font(font, font_size)
    bbox = term_font.getbbox("Mg")
    line_height = (bbox[3] - bbox[1]) + line_gap
    char_width = max(1, int(term_font.getlength("M")))
    screen_width = char_width * DOS_COLS
    screen_height = line_height * DOS_ROWS
    left = max(16, (width - screen_width) // 2)
    top = max(16, (height - screen_height) // 2)

    image = Image.new("RGB", (width, height), "#000000")
    draw = ImageDraw.Draw(image)
    draw.rectangle((left - 18, top - 18, left + screen_width + 18, top + screen_height + 18), outline="#12301d", width=2)

    display_text = text + ("_" if cursor else "")
    visible_rows = dos_wrap_text(display_text)[-DOS_ROWS:]
    while len(visible_rows) < DOS_ROWS:
        visible_rows.append("")

    for row_idx, line in enumerate(visible_rows):
        y = top + row_idx * line_height
        color = "#d7d7d7"
        if line.startswith("Thinking:"):
            color = "#8fbfff"
        elif "Loaded model" in line or "Quality:" in line:
            color = "#66ff99"
        elif line.startswith("Answer:") or line.startswith("[ "):
            color = "#d7ffe7"
        draw.text((left, y), line[:DOS_COLS], font=term_font, fill=color)
    return image


def append_real_dos_frames(
    process: subprocess.Popen[bytes],
    size: tuple[int, int],
    font: Path,
    text: str,
    duration: float,
    fps: int,
) -> None:
    assert process.stdin is not None
    frame_count = max(1, int(duration * fps))
    for idx in range(frame_count):
        cursor = ((idx // max(1, fps // 2)) % 2) == 0
        frame = draw_real_dos_frame(size, font, text, cursor)
        process.stdin.write(frame.tobytes())


def stream_real_dos_text(
    process: subprocess.Popen[bytes],
    size: tuple[int, int],
    font: Path,
    base_text: str,
    added_text: str,
    chars_per_second: float,
    fps: int,
) -> str:
    assert process.stdin is not None
    text = base_text
    frames_per_char = max(1, int(round(fps / chars_per_second)))
    for char_idx, ch in enumerate(added_text):
        text += ch
        for frame_idx in range(frames_per_char):
            cursor = (((char_idx * frames_per_char + frame_idx) // max(1, fps // 2)) % 2) == 0
            frame = draw_real_dos_frame(size, font, text, cursor)
            process.stdin.write(frame.tobytes())
    return text


def render_real_dos_session(
    ffmpeg: str,
    font: Path,
    exchanges: list[TerminalExchange],
    output: Path,
    size: tuple[int, int],
    fps: int = DEFAULT_FPS,
) -> None:
    process = open_rawvideo_encoder(ffmpeg, output, size, fps)
    try:
        text = "FreeDOS kernel 2043\nC:\\> "
        append_real_dos_frames(process, size, font, text, 0.7, fps)
        text = stream_real_dos_text(process, size, font, text, "CD GPT2", 13.0, fps)
        text += "\nC:\\GPT2> "
        append_real_dos_frames(process, size, font, text, 0.25, fps)
        text = stream_real_dos_text(process, size, font, text, "ASSIST", 13.0, fps)
        text += (
            "\n+------------------------------------------------------------+\n"
            "| GPT2-BASIC Assistant Shell                                 |\n"
            "| Pack-driven text UI; VGA sprite/icon slots are pack assets. |\n"
            "+------------------------------------------------------------+\n\n"
            "Pack : CHAT - Conversation Pack\n"
            "Model: PACKS\\CHAT\\MODEL\n"
            "Commands: /about, /pack NAME, /packs, /u, /d, /h, /clear, /quit\n"
            "Loading CHAT model before prompt...\n"
            "Loaded model: PACKS\\CHAT\\MODEL (486dx2-usable)\n\n"
        )
        append_real_dos_frames(process, size, font, text, 1.5, fps)
        for idx, exchange in enumerate(exchanges):
            text += "> "
            text = stream_real_dos_text(process, size, font, text, exchange.prompt, 15.0, fps)
            text += (
                "\n+------------------------------------------------------------+\n"
                "| Assistant                                                  |\n"
                "+------------------------------------------------------------+\n"
                "Thinking: prompt tokens\n"
            )
            append_real_dos_frames(process, size, font, text, 0.4, fps)
            text += "Thinking: sampling output tokens\nAnswer: "
            text = stream_real_dos_text(process, size, font, text, exchange.answer, 24.0, fps)
            text += "\n[ chat,ask,idea,explain,cancel ]\n\n"
            append_real_dos_frames(process, size, font, text, 0.65 if idx < len(exchanges) - 1 else 1.8, fps)
        assert process.stdin is not None
        process.stdin.close()
        status = process.wait()
    finally:
        if process.poll() is None:
            process.kill()
    if status != 0:
        raise SystemExit(f"PROMO_MATERIALS_FAILED real_dos_session_ffmpeg_status={status}")


def build_materials(output_dir: Path, report_path: Path, font: Path, ffmpeg: str, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise SystemExit(f"PROMO_MATERIALS_FAILED output_exists={output_dir} use --force")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_dir = output_dir / "_text"
    text_dir.mkdir(parents=True, exist_ok=True)

    examples = read_quality_examples(report_path)
    scenes = teaser_scenes(examples)

    horizontal_cards: list[tuple[Path, float]] = []
    vertical_cards: list[tuple[Path, float]] = []
    for scene in scenes:
        h_card = output_dir / "cards_1080p" / f"{scene.name}.png"
        v_card = output_dir / "cards_vertical" / f"{scene.name}.png"
        render_card(ffmpeg, font, scene, h_card, (1920, 1080), text_dir)
        render_card(ffmpeg, font, scene, v_card, (1080, 1920), text_dir)
        horizontal_cards.append((h_card, scene.duration))
        vertical_cards.append((v_card, scene.duration))
        print(f"PROMO_CARD {h_card}")
        print(f"PROMO_CARD {v_card}")

    thumbnail = output_dir / "thumbnail_gpt_in_dos.png"
    render_card(
        ffmpeg,
        font,
        Scene(
            "thumbnail",
            "GPT2-BASIC",
            "GPT IN DOS",
            "Real fixed-point transformer inference.",
            "C:\\GPT2> are you real\nThis is real local inference in DOS.\nThe answer comes from local model weights.",
            "Source, evidence, and preview release.",
            1.0,
        ),
        thumbnail,
        (1280, 720),
        text_dir,
    )
    print(f"PROMO_THUMBNAIL {thumbnail}")

    render_video(ffmpeg, horizontal_cards, output_dir / "gpt2_basic_launch_teaser_1080p.mp4", output_dir)
    render_video(ffmpeg, vertical_cards, output_dir / "gpt2_basic_launch_short_vertical.mp4", output_dir)
    exchanges = terminal_exchanges(examples)
    render_terminal_demo(ffmpeg, font, exchanges, output_dir / "gpt2_basic_terminal_demo_1080p.mp4", (1920, 1080))
    render_terminal_demo(ffmpeg, font, exchanges, output_dir / "gpt2_basic_terminal_demo_vertical.mp4", (1080, 1920))
    render_real_dos_session(ffmpeg, font, exchanges, output_dir / "gpt2_basic_real_dos_session_1080p.mp4", (1920, 1080))
    render_real_dos_session(ffmpeg, font, exchanges, output_dir / "gpt2_basic_real_dos_session_vertical.mp4", (1080, 1920))
    print(f"PROMO_VIDEO {output_dir / 'gpt2_basic_launch_teaser_1080p.mp4'}")
    print(f"PROMO_VIDEO {output_dir / 'gpt2_basic_launch_short_vertical.mp4'}")
    print(f"PROMO_VIDEO {output_dir / 'gpt2_basic_terminal_demo_1080p.mp4'}")
    print(f"PROMO_VIDEO {output_dir / 'gpt2_basic_terminal_demo_vertical.mp4'}")
    print(f"PROMO_VIDEO {output_dir / 'gpt2_basic_real_dos_session_1080p.mp4'}")
    print(f"PROMO_VIDEO {output_dir / 'gpt2_basic_real_dos_session_vertical.mp4'}")


def self_test() -> None:
    examples = {
        "chat_are_you_real": "This is real local inference in DOS. The answer comes from local model weights.",
        "chat_how_do_i_focus": "Remove one distraction and choose one task. Start small, then adjust after the first step.",
    }
    scenes = teaser_scenes(examples)
    exchanges = terminal_exchanges(examples)
    require(len(scenes) == 5, "scene_count")
    require(len(exchanges) == 4, "terminal_exchange_count")
    require("are you real" in scenes[1].terminal, "real_prompt_missing")
    require(exchanges[0].prompt == "are you real", "terminal_real_prompt_missing")
    require("fixed-point" in scenes[0].terminal, "quality_claim_missing")
    require("\n" in wrap_lines("one two three four five", 8), "wrap_failed")
    require(terminal_wrap("C:\\GPT2> what is a prompt", 12), "terminal_wrap_failed")
    require(len(dos_wrap_text("A" * 81)) == 2, "dos_wrap_failed")
    print("PROBE_OK promo_materials_self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quality-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    ffmpeg = ffmpeg_command(args.ffmpeg)
    font = resolve_font(args.font)
    build_materials(args.output_dir, args.quality_report, font, ffmpeg, args.force)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Import plain text notes into an assistant pack USER.TXT or KNOW.TXT file."""

from __future__ import annotations

import argparse
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from build_assistant_kdb import DEFAULT_PACK_ROOT, write_or_check


MAX_KEY_CHARS = 48
MAX_TITLE_CHARS = 48
MAX_BODY_CHARS = 180
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "about",
    "should",
    "would",
    "could",
}


@dataclass(frozen=True)
class ImportedRow:
    key: str
    title: str
    body: str

    def render(self) -> str:
        return f"{self.key}|{self.title}|{self.body}"


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("|", "/")).strip()


def title_from_path(path: Path) -> str:
    words = re.sub(r"[^A-Za-z0-9]+", " ", path.stem).strip()
    if not words:
        return "Imported note"
    return words[:MAX_TITLE_CHARS].strip()


def key_from_text(title: str, body: str, index: int) -> str:
    selected: list[str] = []
    seen: set[str] = set()
    for word in re.findall(r"[a-z0-9][a-z0-9-]*", f"{title} {body}".lower()):
        word = word.strip("-")
        if len(word) < 3 or word in STOPWORDS or word in seen:
            continue
        selected.append(word)
        seen.add(word)
        if len(" ".join(selected)) >= MAX_KEY_CHARS - 8:
            break
    if not selected:
        selected = ["note", str(index)]
    key = " ".join(selected)
    if len(key) > MAX_KEY_CHARS:
        key = key[:MAX_KEY_CHARS].rsplit(" ", 1)[0] or key[:MAX_KEY_CHARS]
    return key


def split_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    for raw in re.split(r"\n\s*\n", text):
        paragraph = clean_text(raw)
        if paragraph:
            paragraphs.extend(split_long_paragraph(paragraph))
    return paragraphs


def split_long_paragraph(paragraph: str) -> list[str]:
    if len(paragraph) <= MAX_BODY_CHARS:
        return [paragraph]
    chunks: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > MAX_BODY_CHARS:
            chunks.extend(split_words(sentence))
            continue
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= MAX_BODY_CHARS:
            current = current + " " + sentence
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def split_words(text: str) -> list[str]:
    chunks: list[str] = []
    current = ""
    for word in text.split():
        if not current:
            current = word[:MAX_BODY_CHARS]
        elif len(current) + 1 + len(word) <= MAX_BODY_CHARS:
            current = current + " " + word
        else:
            chunks.append(current)
            current = word[:MAX_BODY_CHARS]
    if current:
        chunks.append(current)
    return chunks


def rows_from_file(path: Path) -> list[ImportedRow]:
    text = path.read_text(encoding="ascii")
    base_title = title_from_path(path)
    paragraphs = split_paragraphs(text)
    rows: list[ImportedRow] = []
    for index, paragraph in enumerate(paragraphs, start=1):
        title = base_title if len(paragraphs) == 1 else f"{base_title} {index}"
        title = title[:MAX_TITLE_CHARS].strip()
        body = paragraph[:MAX_BODY_CHARS].strip()
        rows.append(ImportedRow(key_from_text(title, body, index), title, body))
    return rows


def read_existing_rows(path: Path) -> list[ImportedRow]:
    rows: list[ImportedRow] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="ascii").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        parts = [part.strip() for part in line.split("|", 2)]
        if len(parts) == 3 and all(parts):
            rows.append(ImportedRow(parts[0], parts[1], parts[2]))
    return rows


def merge_rows(existing: list[ImportedRow], imported: list[ImportedRow]) -> list[ImportedRow]:
    merged = list(existing)
    seen = {(row.key.lower(), row.title.lower(), row.body.lower()) for row in existing}
    used_keys = {row.key.lower() for row in existing}
    for row in imported:
        key = row.key
        if (row.key.lower(), row.title.lower(), row.body.lower()) in seen:
            continue
        suffix = 2
        while key.lower() in used_keys:
            base = row.key[: max(1, MAX_KEY_CHARS - 3)].rstrip()
            key = f"{base} {suffix}"
            suffix += 1
        new_row = ImportedRow(key, row.title, row.body)
        merged.append(new_row)
        seen.add((new_row.key.lower(), new_row.title.lower(), new_row.body.lower()))
        used_keys.add(new_row.key.lower())
    return merged


def render_file(rows: list[ImportedRow]) -> str:
    lines = [
        "# key|title|body",
        "# Imported assistant notes. Keep rows ASCII and concise.",
    ]
    lines.extend(row.render() for row in rows)
    return "\n".join(lines) + "\n"


def target_path(pack_root: Path, pack_id: str, target: str) -> Path:
    pack_dir = pack_root / pack_id.upper()
    if not pack_dir.is_dir():
        raise SystemExit(f"ASSISTANT_NOTE_IMPORT_FAILED missing_pack={pack_id}")
    if target == "user":
        return pack_dir / "USER.TXT"
    if target == "know":
        return pack_dir / "KNOW.TXT"
    raise SystemExit(f"ASSISTANT_NOTE_IMPORT_FAILED bad_target={target}")


def import_notes(args: argparse.Namespace) -> int:
    imported: list[ImportedRow] = []
    for note_path in args.notes:
        imported.extend(rows_from_file(note_path))
    target = target_path(args.pack_root, args.pack, args.target)
    existing = read_existing_rows(target)
    merged = merge_rows(existing, imported)
    added = len(merged) - len(existing)
    print(
        "ASSISTANT_NOTE_IMPORT|"
        f"pack={args.pack.upper()}|target={target.name}|input_files={len(args.notes)}|"
        f"rows_added={added}|rows_total={len(merged)}|write={int(args.write)}"
    )
    if args.write:
        target.write_text(render_file(merged), encoding="ascii")
        if args.target == "know" and args.rebuild_kdb:
            status = write_or_check(args.pack_root, write=True)
            if status != 0:
                return status
    else:
        for row in imported:
            print(row.render())
    print("PROBE_OK assistant_note_import=1")
    return 0


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        notes = root / "release notes.txt"
        notes.write_text(
            "Release proof should include tests, logs, checksums, and the target tag.\n\n"
            "Long notes are split into concise chunks so DOS can scan them quickly.",
            encoding="ascii",
        )
        rows = rows_from_file(notes)
        assert len(rows) == 2
        assert all(len(row.key) <= MAX_KEY_CHARS for row in rows)
        assert all(len(row.title) <= MAX_TITLE_CHARS for row in rows)
        assert all(len(row.body) <= MAX_BODY_CHARS for row in rows)
        merged = merge_rows([rows[0]], rows)
        assert len(merged) == 2
        rendered = render_file(merged)
        assert "Release proof" in rendered
    print("PROBE_OK assistant_note_import_self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument("--pack", default="DEV")
    parser.add_argument("--target", choices=("user", "know"), default="user")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--rebuild-kdb", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("notes", nargs="*", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.notes:
        raise SystemExit("ASSISTANT_NOTE_IMPORT_FAILED missing_notes")
    raise SystemExit(import_notes(args))


if __name__ == "__main__":
    main()

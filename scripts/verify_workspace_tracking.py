#!/usr/bin/env python3
"""Verify that workspace-local files stay in documented ignored buckets."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


ALLOWED_IGNORED_EXACT: dict[str, str] = {
    ".DS_Store": "os_metadata",
    "qemu/.DS_Store": "os_metadata",
    "qemu/boot-test.img": "qemu_runtime",
    "qemu/gpt2fat.img": "qemu_runtime",
    "qemu/gpt2hdd.img": "qemu_runtime",
    "to_latex.sh": "local_operator_script",
}

ALLOWED_IGNORED_DIRS: dict[str, str] = {
    "data/online_corpus/": "fetched_corpus_cache",
    "fbc_build/": "freebasic_bootstrap",
    "fbc_win_binary/": "freebasic_bootstrap",
    "memory-bank/": "operator_notes",
    "qemu/__pycache__/": "python_cache",
    "qemu/staging/": "qemu_staging",
    "scripts/__pycache__/": "python_cache",
    "tests/__pycache__/": "python_cache",
    "third_party/": "third_party_cache",
}


@dataclass(frozen=True)
class StatusEntry:
    status: str
    path: str


def parse_porcelain_z(data: bytes) -> list[StatusEntry]:
    entries: list[StatusEntry] = []
    for raw in data.split(b"\0"):
        if not raw:
            continue
        if len(raw) < 4:
            raise ValueError(f"bad porcelain status entry: {raw!r}")
        entries.append(StatusEntry(raw[:2].decode("ascii"), raw[3:].decode("utf-8")))
    return entries


def git_status(root: Path, *, include_ignored: bool) -> list[StatusEntry]:
    args = ["git", "status", "--porcelain=v1", "-z"]
    if include_ignored:
        args.extend(["--ignored", "--untracked-files=all"])
    result = subprocess.run(args, cwd=root, check=True, stdout=subprocess.PIPE)
    return parse_porcelain_z(result.stdout)


def ignored_category(path: str) -> str | None:
    if path in ALLOWED_IGNORED_EXACT:
        return ALLOWED_IGNORED_EXACT[path]
    for ignored_dir, category in ALLOWED_IGNORED_DIRS.items():
        if path == ignored_dir or path.startswith(ignored_dir):
            return category
    if path.startswith(".venv") and ("/" not in path[:-1] or "/" in path):
        return "python_virtualenv"
    return None


def verify_workspace(root: Path, *, allow_dirty: bool) -> tuple[int, dict[str, int]]:
    source_entries = git_status(root, include_ignored=False)
    untracked = sorted(entry.path for entry in source_entries if entry.status == "??")
    if untracked:
        raise SystemExit(f"WORKSPACE_TRACKING_FAILED untracked_file={untracked[0]}")
    if source_entries and not allow_dirty:
        sample = sorted(f"{entry.status} {entry.path}" for entry in source_entries)[0]
        raise SystemExit(f"WORKSPACE_TRACKING_FAILED dirty_source={sample}")

    ignored_paths = sorted(entry.path for entry in git_status(root, include_ignored=True) if entry.status == "!!")
    categories: dict[str, int] = {}
    for path in ignored_paths:
        category = ignored_category(path)
        if category is None:
            raise SystemExit(f"WORKSPACE_TRACKING_FAILED unexpected_ignored={path}")
        categories[category] = categories.get(category, 0) + 1
    return len(ignored_paths), categories


def self_test() -> None:
    sample = parse_porcelain_z(b"!! qemu/staging/\0?? scratch.txt\0 M README.md\0")
    if sample != [
        StatusEntry("!!", "qemu/staging/"),
        StatusEntry("??", "scratch.txt"),
        StatusEntry(" M", "README.md"),
    ]:
        raise RuntimeError("porcelain parser failed")
    probes = {
        ".DS_Store": "os_metadata",
        ".venv-torch311/": "python_virtualenv",
        "data/online_corpus/": "fetched_corpus_cache",
        "qemu/gpt2hdd.img": "qemu_runtime",
        "qemu/staging/": "qemu_staging",
    }
    for path, expected in probes.items():
        actual = ignored_category(path)
        if actual != expected:
            raise RuntimeError(f"ignored category mismatch for {path}: {actual}")
    if ignored_category("scratch/output.log") is not None:
        raise RuntimeError("unexpected ignored file was accepted")
    print("PROBE_OK workspace_tracking_porcelain_parser=1")
    print("PROBE_OK workspace_tracking_ignored_categories=1")
    print("PROBE_OK workspace_tracking_self_test=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    ignored_count, categories = verify_workspace(args.root, allow_dirty=args.allow_dirty)
    print("PROBE_OK workspace_tracking_no_untracked=1")
    if not args.allow_dirty:
        print("PROBE_OK workspace_tracking_clean_source=1")
    print(f"PROBE_OK workspace_tracking_ignored_allowed={ignored_count}")
    for category in sorted(categories):
        print(f"PROBE_OK workspace_tracking_category_{category}={categories[category]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fetch and clean a provenance-tracked online training corpus.

The default source set is intentionally conservative: public-domain or
government/open data text with clear reuse posture. ShareAlike/GFDL sources are
available behind explicit flags because a production checkpoint may carry
attribution and downstream licensing obligations.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "online_corpus"
DEFAULT_REPORT = ROOT / "qemu" / "evidence" / "online_training_data_audit.md"
USER_AGENT = "gpt2-basic-corpus-fetcher/1.0 (+https://github.com/tsotchke/gpt2-basic)"


@dataclass(frozen=True)
class CorpusSource:
    source_id: str
    title: str
    url: str
    landing_url: str
    license_name: str
    license_url: str
    tier: str
    source_format: str
    default_enabled: bool
    notes: str
    max_chars: int


@dataclass
class SourceResult:
    source: CorpusSource
    status: str
    raw_path: str
    clean_path: str
    raw_bytes: int
    clean_chars: int
    error: str = ""


class TextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "svg", "noscript"}:
            self.skip_depth += 1
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "pre"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "svg", "noscript"} and self.skip_depth > 0:
            self.skip_depth -= 1
        if tag in {"p", "div", "li", "h1", "h2", "h3", "pre"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        return html.unescape(" ".join(self.parts))


SOURCES: list[CorpusSource] = [
    CorpusSource(
        source_id="pg_alice_11",
        title="Alice's Adventures in Wonderland",
        url="https://www.gutenberg.org/cache/epub/11/pg11.txt",
        landing_url="https://www.gutenberg.org/ebooks/11",
        license_name="Project Gutenberg public-domain text in the United States after stripping PG boilerplate",
        license_url="https://www.gutenberg.org/policy/license.html",
        tier="permissive",
        source_format="gutenberg",
        default_enabled=True,
        notes="Compact public-domain fiction for dialogue, sentence rhythm, and narrative continuity.",
        max_chars=180_000,
    ),
    CorpusSource(
        source_id="pg_sherlock_1661",
        title="The Adventures of Sherlock Holmes",
        url="https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
        landing_url="https://www.gutenberg.org/ebooks/1661",
        license_name="Project Gutenberg public-domain text in the United States after stripping PG boilerplate",
        license_url="https://www.gutenberg.org/policy/license.html",
        tier="permissive",
        source_format="gutenberg",
        default_enabled=True,
        notes="Readable public-domain prose with direct problem/answer structure.",
        max_chars=260_000,
    ),
    CorpusSource(
        source_id="pg_frankenstein_84",
        title="Frankenstein",
        url="https://www.gutenberg.org/cache/epub/84/pg84.txt",
        landing_url="https://www.gutenberg.org/ebooks/84",
        license_name="Project Gutenberg public-domain text in the United States after stripping PG boilerplate",
        license_url="https://www.gutenberg.org/policy/license.html",
        tier="permissive",
        source_format="gutenberg",
        default_enabled=True,
        notes="Long-form explanatory and reflective English, capped for tiny-model training.",
        max_chars=220_000,
    ),
    CorpusSource(
        source_id="pg_pride_1342",
        title="Pride and Prejudice",
        url="https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
        landing_url="https://www.gutenberg.org/ebooks/1342",
        license_name="Project Gutenberg public-domain text in the United States after stripping PG boilerplate",
        license_url="https://www.gutenberg.org/policy/license.html",
        tier="permissive",
        source_format="gutenberg",
        default_enabled=True,
        notes="Clean public-domain conversational prose, capped to avoid overpowering domain text.",
        max_chars=220_000,
    ),
    CorpusSource(
        source_id="nist_sp800_info",
        title="NIST SP 800-series general information",
        url="https://www.nist.gov/itl/publications-0/nist-special-publication-800-series-general-information",
        landing_url="https://www.nist.gov/itl/publications-0/nist-special-publication-800-series-general-information",
        license_name="NIST SP 800 publications are not subject to U.S. copyright",
        license_url="https://www.nist.gov/itl/publications-0/nist-special-publication-800-series-general-information",
        tier="permissive",
        source_format="html",
        default_enabled=True,
        notes="Modern technical/government prose for precise explanatory style.",
        max_chars=45_000,
    ),
    CorpusSource(
        source_id="nasa_earthdata_guidance",
        title="NASA Earthdata data use and citation guidance",
        url="https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance",
        landing_url="https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance",
        license_name="NASA-led mission data are CC0 unless otherwise marked; NASA materials generally not copyrighted in the U.S.",
        license_url="https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance",
        tier="permissive",
        source_format="html",
        default_enabled=True,
        notes="Concise policy and technical explanatory prose with explicit provenance guidance.",
        max_chars=55_000,
    ),
    CorpusSource(
        source_id="freedos_install",
        title="How to install FreeDOS",
        url="https://help.freedos.org/docs/howto/install.html",
        landing_url="https://help.freedos.org/docs/howto/install.html",
        license_name="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        tier="sharealike",
        source_format="html",
        default_enabled=False,
        notes="High-value DOS domain text; opt-in because ShareAlike attribution obligations must be tracked.",
        max_chars=45_000,
    ),
    CorpusSource(
        source_id="freedos_qemu",
        title="How to install FreeDOS on QEMU",
        url="https://help.freedos.org/docs/howto/qemu.html",
        landing_url="https://help.freedos.org/docs/howto/qemu.html",
        license_name="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        tier="sharealike",
        source_format="html",
        default_enabled=False,
        notes="Directly relevant QEMU/DOS setup prose; opt-in ShareAlike.",
        max_chars=35_000,
    ),
    CorpusSource(
        source_id="freedos_bat",
        title="Automate tasks with BAT files",
        url="https://help.freedos.org/docs/howto/bat.html",
        landing_url="https://help.freedos.org/docs/howto/bat.html",
        license_name="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        tier="sharealike",
        source_format="html",
        default_enabled=False,
        notes="Domain text about DOS batch control flow and command behavior; opt-in ShareAlike.",
        max_chars=55_000,
    ),
    CorpusSource(
        source_id="freedos_cddir",
        title="Get around with CD and DIR",
        url="https://help.freedos.org/docs/howto/cddir.html",
        landing_url="https://help.freedos.org/docs/howto/cddir.html",
        license_name="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        tier="sharealike",
        source_format="html",
        default_enabled=False,
        notes="Useful DOS command vocabulary and filesystem prose; opt-in ShareAlike.",
        max_chars=60_000,
    ),
]


EXCLUDED_SOURCES = [
    {
        "source": "OpenStax computer science and data science books",
        "reason": "Current book pages state that the book may not be used for LLM training or ingested into generative AI offerings without permission.",
        "url": "https://openstax.org/books/introduction-computer-science/pages/preface",
    },
    {
        "source": "Stack Exchange data dump",
        "reason": "Useful technical Q&A, but CC BY-SA versioning and attribution obligations make it unsuitable for the default production corpus.",
        "url": "https://stackoverflow.com/help/licensing",
    },
    {
        "source": "Simple English Wikipedia dump",
        "reason": "Good small-encyclopedia candidate, but Wikimedia text is GFDL/CC BY-SA; keep behind a later explicit ShareAlike import path.",
        "url": "https://dumps.wikimedia.org/legal.html",
    },
    {
        "source": "RFC Editor text corpus",
        "reason": "RFCs are freely reproducible unmodified, but training cleanup creates a transformed corpus; use only after legal policy is explicit.",
        "url": "https://www.rfc-editor.org/faq/",
    },
    {
        "source": "FreeBASIC wiki manual",
        "reason": "Highly relevant to the runtime, but the official license page clearly covers compiler/runtime licensing rather than granting a clean license for all wiki manual text.",
        "url": "https://www.freebasic.net/wiki/wikka.php?wakka=GnuLicenses",
    },
    {
        "source": "OpenWebText, Common Crawl, and random web scrapes",
        "reason": "Large and attractive for perplexity, but provenance and redistribution posture are too ambiguous for this production baseline.",
        "url": "https://commoncrawl.org/",
    },
]


def source_enabled(source: CorpusSource, include_sharealike: bool) -> bool:
    if source.default_enabled:
        return True
    return include_sharealike and source.tier == "sharealike"


def fetch_url(url: str, timeout: float, pause_seconds: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    if pause_seconds > 0:
        time.sleep(pause_seconds)
    return data


def decode_bytes(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def strip_gutenberg(text: str) -> str:
    start_match = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE | re.DOTALL)
    end_match = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*", text, re.IGNORECASE | re.DOTALL)
    if start_match:
        text = text[start_match.end() :]
    if end_match:
        text = text[: end_match.start()]
    return text


def html_to_text(text: str) -> str:
    parser = TextHTMLParser()
    parser.feed(text)
    return parser.text()


def normalize_text(text: str, max_chars: int) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    paragraphs: list[str] = []
    seen: set[str] = set()
    for raw in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\s+", " ", raw).strip()
        paragraph = re.sub(r"[^A-Za-z0-9 .,;:!?()/%_+'\"-]", " ", paragraph)
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if len(paragraph) < 80:
            continue
        key = paragraph.lower()
        if key in seen:
            continue
        seen.add(key)
        paragraphs.append(paragraph)

    clean = "\n\n".join(paragraphs).strip()
    if max_chars > 0 and len(clean) > max_chars:
        clean = clean[:max_chars]
        last_break = clean.rfind("\n\n")
        if last_break > max_chars // 2:
            clean = clean[:last_break]
    return clean.strip() + ("\n" if clean.strip() else "")


def clean_source_text(source: CorpusSource, raw_text: str) -> str:
    text = raw_text
    if source.source_format == "gutenberg":
        text = strip_gutenberg(text)
    elif source.source_format == "html":
        text = html_to_text(text)
    return normalize_text(text, source.max_chars)


def write_combined_corpus(results: list[SourceResult], out_dir: Path) -> Path:
    combined = out_dir / "online_training_corpus.txt"
    with combined.open("w", encoding="ascii") as out:
        for result in results:
            if result.status != "ok":
                continue
            text = Path(result.clean_path).read_text(encoding="ascii")
            out.write(f"\n\n[source:{result.source.source_id} title:{result.source.title}]\n\n")
            out.write(text)
    return combined


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def build_report(results: list[SourceResult], combined_path: Path, include_sharealike: bool) -> str:
    ok_results = [result for result in results if result.status == "ok"]
    total_chars = sum(result.clean_chars for result in ok_results)
    lines = [
        "# GPT2-BASIC Online Training Data Audit",
        "",
        "This report tracks the online corpus used for the next training pass. The default source set is conservative: Project Gutenberg public-domain text plus NIST/NASA public-government/open-data prose. ShareAlike/GFDL-style sources are opt-in and must stay provenance-tracked.",
        "",
        "## Corpus Output",
        "",
        f"- Combined corpus: `{rel(combined_path)}`",
        f"- Sources fetched: {len(ok_results)}/{len(results)}",
        f"- Clean characters: {total_chars}",
        f"- ShareAlike enabled: {'yes' if include_sharealike else 'no'}",
        "",
        "## Source Matrix",
        "",
        "| Status | Tier | Source | License posture | Clean chars | Notes |",
        "|---|---|---|---|---:|---|",
    ]

    for result in results:
        source = result.source
        note = source.notes if result.status == "ok" else result.error
        lines.append(
            f"| `{result.status}` | `{source.tier}` | [{source.title}]({source.landing_url}) | "
            f"[{source.license_name}]({source.license_url}) | {result.clean_chars} | {note} |"
        )

    lines.extend(
        [
            "",
            "## Excluded By Default",
            "",
            "| Source | Reason | Reference |",
            "|---|---|---|",
        ]
    )
    for excluded in EXCLUDED_SOURCES:
        lines.append(f"| {excluded['source']} | {excluded['reason']} | {excluded['url']} |")

    lines.extend(
        [
            "",
            "## Pretraining Command",
            "",
            "```sh",
            f"python3 scripts/train_gpt2_basic.py --profile 486sx-safe --include-docs --corpus-file {rel(combined_path)} --corpus-weight 1 --device mps --steps 2500 --output assets/gpt2_basic/MODEL_ONLINE_PRETRAIN",
            "```",
            "",
            "Use this as a pretraining base. Fine-tune a candidate on project/runtime domain text, then run host held-out quality before spending QEMU time. Promote to DOS only after host held-out quality beats the active baseline.",
        ]
    )
    return "\n".join(lines) + "\n"


def fetch_sources(args: argparse.Namespace) -> list[SourceResult]:
    raw_dir = args.out_dir / "raw"
    clean_dir = args.out_dir / "clean"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    results: list[SourceResult] = []
    selected = [source for source in SOURCES if source_enabled(source, args.include_sharealike)]
    for source in selected:
        raw_path = raw_dir / f"{source.source_id}.txt"
        clean_path = clean_dir / f"{source.source_id}.txt"
        try:
            raw = fetch_url(source.url, args.timeout, args.pause_seconds)
            raw_path.write_bytes(raw)
            clean = clean_source_text(source, decode_bytes(raw))
            clean_path.write_text(clean, encoding="ascii")
            results.append(
                SourceResult(
                    source=source,
                    status="ok" if clean else "empty",
                    raw_path=str(raw_path),
                    clean_path=str(clean_path),
                    raw_bytes=len(raw),
                    clean_chars=len(clean),
                    error="" if clean else "source produced no clean paragraphs",
                )
            )
            print(f"fetched {source.source_id}: {len(clean)} clean chars", flush=True)
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            results.append(
                SourceResult(
                    source=source,
                    status="failed",
                    raw_path=str(raw_path),
                    clean_path=str(clean_path),
                    raw_bytes=0,
                    clean_chars=0,
                    error=str(exc),
                )
            )
            print(f"failed {source.source_id}: {exc}", flush=True)
            if args.fail_fast:
                raise
    return results


def self_test() -> None:
    sample = """Header
*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***

This is a useful paragraph with enough words to survive cleaning. It explains a simple machine and how text should flow.

*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***
Footer"""
    cleaned = clean_source_text(SOURCES[0], sample)
    assert "Header" not in cleaned
    assert "useful paragraph" in cleaned
    parser_text = html_to_text("<html><body><h1>Title</h1><p>Paragraph with enough content to be useful for cleaning and model training.</p><script>x</script></body></html>")
    assert "Paragraph" in parser_text
    assert "x" not in parser_text
    html_sample = """
    <html><body><h1>Runtime source</h1>
    <p>This paragraph is long enough to survive the cleaner and it describes model training,
    tokenizer validation, runtime evidence, and conservative corpus provenance.</p>
    <script>hidden()</script></body></html>
    """
    original_fetch_url = fetch_url

    def fake_fetch_url(url: str, timeout: float, pause_seconds: float) -> bytes:
        if "gutenberg" in url:
            return sample.encode("utf-8")
        return html_sample.encode("utf-8")

    with tempfile.TemporaryDirectory() as tmp:
        try:
            globals()["fetch_url"] = fake_fetch_url
            args = argparse.Namespace(
                out_dir=Path(tmp),
                include_sharealike=False,
                timeout=1.0,
                pause_seconds=0.0,
                fail_fast=True,
            )
            results = fetch_sources(args)
            combined_path = write_combined_corpus(results, args.out_dir)
            report = build_report(results, combined_path, args.include_sharealike)
        finally:
            globals()["fetch_url"] = original_fetch_url
    assert results
    assert all(result.status == "ok" for result in results)
    assert "Source Matrix" in report
    print("trace_scope online_corpus_contract")
    print("trace TextHTMLParser.handle_starttag")
    print("trace TextHTMLParser.handle_endtag")
    print("trace TextHTMLParser.handle_data")
    print("trace normalize_text")
    print("trace strip_gutenberg")
    print("trace fetch_sources")
    print("trace write_combined_corpus")
    print("trace build_report")
    print("artifact: SOURCE_MANIFEST.json")
    print("PROBE_OK online_corpus_cleaning=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--include-sharealike", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = fetch_sources(args)
    combined_path = write_combined_corpus(results, args.out_dir)

    manifest = {
        "generated_by": "scripts/fetch_online_training_corpus.py",
        "combined_corpus": str(combined_path),
        "include_sharealike": args.include_sharealike,
        "sources": [
            {
                **asdict(result.source),
                "status": result.status,
                "raw_path": result.raw_path,
                "clean_path": result.clean_path,
                "raw_bytes": result.raw_bytes,
                "clean_chars": result.clean_chars,
                "error": result.error,
            }
            for result in results
        ],
        "excluded_by_default": EXCLUDED_SOURCES,
    }
    (args.out_dir / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(build_report(results, combined_path, args.include_sharealike), encoding="ascii")
    print(f"wrote {combined_path}", flush=True)
    print(f"wrote {args.out_dir / 'SOURCE_MANIFEST.json'}", flush=True)
    print(f"wrote {args.report}", flush=True)


if __name__ == "__main__":
    main()

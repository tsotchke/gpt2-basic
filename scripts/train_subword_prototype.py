#!/usr/bin/env python3
"""Host-only GPT2-BASIC subword prototype.

This tests the architecture direction before touching the DOS runtime. The
exported weights are real transformer weights, but the checkpoint is not
DOS-ready until the tokenizer/vocabulary path is wired through QEMU.
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch as th
import torch.nn.functional as F

from evaluate_gpt2_basic_quality import (
    DEFAULT_MIN_GENERATED,
    load_float_model,
    markdown_report,
    parse_config,
    quality_prompts,
    score_completion,
)
from gpt2_basic_tokenizer import GPT2BasicTokenizer
from subword_architecture_probe import build_vocab
from train_tiny_gpt import (
    BYTE_OFFSET,
    EOT_TOKEN,
    UNK_TOKEN,
    Config,
    GPT2BasicModel,
    MODEL_PROFILES,
    batch_from_stream,
    build_backend_runtime,
    clean_ascii,
    export_model,
    load_corpus_file,
    load_documents,
    load_repo_documents,
    print_backend_contract,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "assets" / "gpt2_basic" / "MODEL_SUBWORD_PROTOTYPE"
DEFAULT_REPORT = ROOT / "qemu" / "evidence" / "subword_prototype_report.md"


@dataclass(frozen=True)
class SubwordTokenizer:
    pieces: tuple[str, ...]
    token_to_id: dict[str, int]
    id_to_piece: dict[int, str]

    @classmethod
    def build(cls, basis_text: str, vocab_size: int) -> "SubwordTokenizer":
        pieces = tuple(build_vocab(basis_text, vocab_size))
        token_to_id = {piece: idx + 258 for idx, piece in enumerate(pieces)}
        id_to_piece = {idx + 258: piece for idx, piece in enumerate(pieces)}
        return cls(pieces=pieces, token_to_id=token_to_id, id_to_piece=id_to_piece)

    @property
    def vocab_size(self) -> int:
        return 258 + len(self.pieces)

    def encode(self, text: str) -> list[int]:
        text = clean_ascii(text)
        by_first: dict[str, list[str]] = {}
        for piece in sorted(self.pieces, key=len, reverse=True):
            by_first.setdefault(piece[0], []).append(piece)

        tokens: list[int] = []
        idx = 0
        while idx < len(text):
            matched = ""
            for piece in by_first.get(text[idx], []):
                if text.startswith(piece, idx):
                    matched = piece
                    break
            if matched:
                tokens.append(self.token_to_id[matched])
                idx += len(matched)
            else:
                tokens.append(ord(text[idx]) + BYTE_OFFSET)
                idx += 1
        return tokens

    def decode(self, tokens: list[int]) -> str:
        chars: list[str] = []
        for token in tokens:
            if token == EOT_TOKEN:
                break
            if token >= 258:
                chars.append(self.id_to_piece.get(token, ""))
            elif token >= BYTE_OFFSET:
                byte = token - BYTE_OFFSET
                if 32 <= byte <= 126:
                    chars.append(chr(byte))
        return "".join(chars)

    def write(self, path: Path) -> None:
        data = {
            "kind": "greedy-domain-subword-prototype",
            "vocab_size": self.vocab_size,
            "byte_vocab": 258,
            "max_piece_chars": max((len(piece) for piece in self.pieces), default=1),
            "pieces": list(self.pieces),
        }
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="ascii")

    @classmethod
    def load(cls, path: Path) -> "SubwordTokenizer":
        data = json.loads(path.read_text(encoding="ascii"))
        pieces = tuple(str(piece) for piece in data["pieces"])
        token_to_id = {piece: idx + 258 for idx, piece in enumerate(pieces)}
        id_to_piece = {idx + 258: piece for idx, piece in enumerate(pieces)}
        return cls(pieces=pieces, token_to_id=token_to_id, id_to_piece=id_to_piece)


def prototype_export_tokenizer(tokenizer: SubwordTokenizer) -> GPT2BasicTokenizer:
    pieces = GPT2BasicTokenizer.byte().id_to_piece
    pieces.extend(piece.encode("ascii", errors="ignore") for piece in tokenizer.pieces)
    return GPT2BasicTokenizer(pieces, [])


def mark_host_only_export(output: Path) -> None:
    profile_path = output / "PROFILE.TXT"
    if profile_path.exists():
        profile_text = profile_path.read_text(encoding="ascii", errors="ignore")
        profile_text = profile_text.replace("tokenizer=byte", "tokenizer=host-only-greedy-subword")
        profile_text += "dos_ready=0\n"
        profile_text += "host_tokenizer=SUBWORDVOCAB.json\n"
        profile_path.write_text(profile_text, encoding="ascii")
    (output / "HOST_ONLY_NOT_DOS_READY.txt").write_text(
        "This checkpoint uses the rejected greedy subword prototype tokenizer. "
        "It is for host evaluation only, not DOS promotion.\n",
        encoding="ascii",
    )

def build_stream(tokenizer: SubwordTokenizer, document_groups: list[tuple[list[str], int]], repeats: int) -> list[int]:
    tokens: list[int] = []
    for _ in range(repeats):
        weighted_docs: list[str] = []
        for documents, weight in document_groups:
            if weight > 0:
                weighted_docs.extend(documents * weight)
        random.shuffle(weighted_docs)
        for doc in weighted_docs:
            tokens.extend(tokenizer.encode(doc))
            tokens.append(EOT_TOKEN)
    return tokens


def mask_logits(logits: th.Tensor, generated_count: int, min_generated: int) -> th.Tensor:
    masked = logits.clone()
    masked[UNK_TOKEN] = -1.0e9
    subword_mode = masked.numel() > 258
    byte_allow = set(" .,;:!?()/%_+'\"-0123456789")
    for token in range(2, min(258, masked.numel())):
        byte = token - BYTE_OFFSET
        if byte < 32 or byte > 126:
            masked[token] = -1.0e9
        elif subword_mode and chr(byte) not in byte_allow:
            masked[token] = -1.0e9
    if generated_count < min_generated:
        masked[EOT_TOKEN] = -1.0e9
    return masked


def generate_completion(
    model: GPT2BasicModel,
    tokenizer: SubwordTokenizer,
    prompt: str,
    max_new_tokens: int,
    min_generated: int,
    device: th.device,
) -> tuple[str, int]:
    context = tokenizer.encode(prompt)
    if not context:
        context = [EOT_TOKEN]
    prompt_len = len(context)

    with th.no_grad():
        for _step in range(max_new_tokens):
            active = context[-model.cfg.n_positions :]
            idx = th.tensor([active], dtype=th.long, device=device)
            logits = model(idx)[0, -1]
            masked = mask_logits(logits, len(context) - prompt_len, min_generated)
            next_token = int(th.argmax(masked).item())
            context.append(next_token)

            completion = tokenizer.decode(context[prompt_len:])
            if next_token == EOT_TOKEN:
                break
            if len(completion) >= 90 and completion.rstrip().endswith((".", "!", "?")):
                break

    generated = context[prompt_len:]
    return tokenizer.decode(generated), len(generated)


def evaluate(
    model: GPT2BasicModel,
    tokenizer: SubwordTokenizer,
    suite: str,
    max_new_tokens: int,
    min_generated: int,
    threshold: float,
    device: th.device,
):
    results = []
    for prompt in quality_prompts(suite):
        completion, generated_tokens = generate_completion(model, tokenizer, prompt.prompt, max_new_tokens, min_generated, device)
        results.append(score_completion(prompt, completion, generated_tokens, threshold))
    return results


def train(args: argparse.Namespace) -> tuple[GPT2BasicModel, SubwordTokenizer, dict[str, int]]:
    th.manual_seed(args.seed)
    random.seed(args.seed)

    profile = MODEL_PROFILES[args.profile]
    cfg = Config(
        n_positions=profile.cfg.n_positions,
        n_embd=profile.cfg.n_embd,
        n_head=profile.cfg.n_head,
        n_layer=profile.cfg.n_layer,
        hidden_dim=profile.cfg.hidden_dim,
        vocab_size=args.vocab_size,
    )

    core_documents = load_documents()
    if args.include_docs:
        core_documents.extend(load_repo_documents())

    external_documents: list[str] = []
    for corpus_path in args.corpus_file or []:
        loaded = load_corpus_file(corpus_path, args.corpus_doc_chars, args.corpus_max_docs)
        external_documents.extend(loaded)
        print(f"loaded_corpus_file: {corpus_path} docs={len(loaded)}", flush=True)

    basis_text = "\n".join(core_documents + external_documents)
    tokenizer = SubwordTokenizer.build(basis_text, args.vocab_size)
    cfg.vocab_size = tokenizer.vocab_size

    document_groups: list[tuple[list[str], int]] = [(core_documents, args.core_weight)]
    if external_documents:
        document_groups.append((external_documents, args.corpus_weight))
    stream = build_stream(tokenizer, document_groups, args.repeats)
    data = th.tensor(stream, dtype=th.long)

    runtime = build_backend_runtime(args.device)
    print_backend_contract(runtime)
    device = runtime.device

    model = GPT2BasicModel(cfg).to(device)
    optimizer = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"device: {device}", flush=True)
    print(f"subword_vocab_size: {tokenizer.vocab_size}", flush=True)
    print(f"subword_pieces: {len(tokenizer.pieces)}", flush=True)
    print(f"training_documents_core: {len(core_documents)}", flush=True)
    print(f"training_documents_external: {len(external_documents)}", flush=True)
    print(f"training tokens: {len(stream)}", flush=True)

    model.train()
    for step in range(1, args.steps + 1):
        x, y = batch_from_stream(data, args.batch_size, cfg.n_positions, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            print(f"step {step:5d} loss {loss.item():.4f}", flush=True)

    model.eval()
    stats = {
        "training_tokens": len(stream),
        "core_documents": len(core_documents),
        "external_documents": len(external_documents),
        "subword_pieces": len(tokenizer.pieces),
    }
    return model, tokenizer, stats


def report_text(args: argparse.Namespace, cfg: Config, stats: dict[str, int], heldout: str, runtime: str) -> str:
    try:
        output_text = str(args.output.resolve().relative_to(ROOT))
    except ValueError:
        output_text = str(args.output)
    return "\n".join(
        [
            "# GPT2-BASIC Subword Host Prototype",
            "",
            "This is a host-only architecture probe. The transformer is real, but the checkpoint is not DOS-ready until the subword tokenizer is implemented in the production runtime.",
            "",
            "## Training",
            "",
            f"- Output: `{output_text}`",
            f"- Profile: `{args.profile}`",
            f"- Vocab size: {cfg.vocab_size}",
            f"- Subword pieces: {stats['subword_pieces']}",
            f"- Training tokens: {stats['training_tokens']}",
            f"- Core docs: {stats['core_documents']}",
            f"- External docs: {stats['external_documents']}",
            "",
            "## Held-out Quality",
            "",
            heldout,
            "",
            "## Runtime-Regression Quality",
            "",
            runtime,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(MODEL_PROFILES), default="486sx-safe")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--corpus-file", action="append", type=Path)
    parser.add_argument("--include-docs", action="store_true")
    parser.add_argument("--core-weight", type=int, default=24)
    parser.add_argument("--corpus-weight", type=int, default=3)
    parser.add_argument("--corpus-doc-chars", type=int, default=1800)
    parser.add_argument("--corpus-max-docs", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.8e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--sample-tokens", type=int, default=48)
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=486)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        tokenizer = SubwordTokenizer.build("fixed point runtime memory tokens speed", 300)
        encoded = tokenizer.encode("fixed point runtime")
        decoded = tokenizer.decode(encoded)
        assert "fixed" in decoded
        cfg = Config(n_positions=16, n_embd=16, n_head=4, n_layer=1, hidden_dim=32, vocab_size=tokenizer.vocab_size)
        model = GPT2BasicModel(cfg).to(th.device("cpu"))
        completion, generated = generate_completion(model, tokenizer, "fixed point", 1, 0, th.device("cpu"))
        assert generated == 1
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            vocab_path = tmp_path / "SUBWORDVOCAB.json"
            tokenizer.write(vocab_path)
            loaded = SubwordTokenizer.load(vocab_path)
            assert loaded.decode(encoded) == decoded
            export_tokenizer = prototype_export_tokenizer(tokenizer)
            export_tokenizer.validate_for_vocab_size(tokenizer.vocab_size)
            (tmp_path / "quality_heldout.md").write_text("# heldout\n", encoding="ascii")
            (tmp_path / "quality_runtime.md").write_text("# runtime\n", encoding="ascii")
            (tmp_path / "PROFILE.TXT").write_text("tokenizer=byte\n", encoding="ascii")
            mark_host_only_export(tmp_path)
            assert "dos_ready=0" in (tmp_path / "PROFILE.TXT").read_text(encoding="ascii")
            report = report_text(
                argparse.Namespace(output=tmp_path, profile="486sx-safe"),
                cfg,
                {"training_tokens": 0, "core_documents": 0, "external_documents": 0, "subword_pieces": len(tokenizer.pieces)},
                "# heldout",
                "# runtime",
            )
        assert "Subword Host Prototype" in report
        assert isinstance(completion, str)
        print("trace_scope subword_prototype_contract")
        print("trace mask_logits")
        print("trace generate_completion")
        print("trace prototype_export_tokenizer")
        print("trace mark_host_only_export")
        print("trace report_text")
        print("artifact: SUBWORDVOCAB.json")
        print("artifact: HOST_ONLY_NOT_DOS_READY.txt")
        print("artifact: quality_heldout.md")
        print("artifact: quality_runtime.md")
        print("PROBE_OK subword_prototype_tokenizer=1")
        return

    if args.eval_only:
        runtime = build_backend_runtime(args.device)
        print_backend_contract(runtime)
        tokenizer = SubwordTokenizer.load(args.output / "SUBWORDVOCAB.json")
        fixed_cfg = parse_config(args.output / "GPT2CFG.TXT")
        model = load_float_model(args.output, fixed_cfg, runtime.device)
        stats = {
            "training_tokens": 0,
            "core_documents": 0,
            "external_documents": 0,
            "subword_pieces": len(tokenizer.pieces),
        }
        cfg = fixed_cfg
        device = runtime.device
    else:
        model, tokenizer, stats = train(args)
        device = next(model.parameters()).device
        cfg = model.cfg

    print("", flush=True)
    for prompt in ["What makes this real inference?", "A BASIC transformer runtime", "What limits a tiny transformer on old PCs?"]:
        completion, _ = generate_completion(model, tokenizer, prompt, args.sample_tokens, 20, device)
        print(f"prompt: {prompt}", flush=True)
        print(prompt + completion, flush=True)
        print("", flush=True)

    if not args.eval_only:
        args.output.mkdir(parents=True, exist_ok=True)
        model.cpu()
        export_model(
            model,
            cfg,
            args.output,
            f"{args.profile}-subword-prototype",
            False,
            prototype_export_tokenizer(tokenizer),
        )
        tokenizer.write(args.output / "SUBWORDVOCAB.json")
        mark_host_only_export(args.output)
        model = model.to(device)
        setattr(cfg, "profile", f"{args.profile}-subword-prototype")

    heldout_results = evaluate(model, tokenizer, "heldout", args.sample_tokens, 20, args.threshold, device)
    runtime_results = evaluate(model, tokenizer, "runtime-regression", args.sample_tokens, 20, args.threshold, device)
    heldout_report = markdown_report(cfg, heldout_results, args.threshold, "subword-host", "heldout")
    runtime_report = markdown_report(cfg, runtime_results, args.threshold, "subword-host", "runtime-regression")

    (args.output / "quality_heldout.md").write_text(heldout_report + "\n", encoding="ascii")
    (args.output / "quality_runtime.md").write_text(runtime_report + "\n", encoding="ascii")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text(args, cfg, stats, heldout_report, runtime_report), encoding="ascii")
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train and export GPT2-BASIC fixed-point model checkpoints.

The exported model is intentionally 486-oriented: byte-level tokens, learned
position embeddings, decoder-only causal attention, feed-forward blocks, layer
norms, and an output head. Training happens on the host; DOS loads
MODEL/GPT2CFG.TXT, MODEL/GPT2FX.BIN, and MODEL/GPT2EXP.BIN for the fixed-point
forward pass.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch as th
from torch import nn
import torch.nn.functional as F

from gpt2_basic_tokenizer import BYTE_VOCAB_SIZE, GPT2BasicTokenizer, build_bpe_tokenizer, build_lexicon_tokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "assets" / "gpt2_basic" / "MODEL"
VOCAB_SIZE = BYTE_VOCAB_SIZE
EOT_TOKEN = 0
UNK_TOKEN = 1
BYTE_OFFSET = 2
FIXED_SHIFT = 12
FIXED_SCALE = 1 << FIXED_SHIFT
EXP_TABLE_SIZE = 513
EXP_TABLE_MAX = 16.0
DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")
DEFAULT_TOKENIZER = GPT2BasicTokenizer.byte()
SENTENCE_STOP_MIN_TOKENS = 30


CURATED_DOCUMENTS = [
    (
        "Hello there. I am GPT2 BASIC, a compact language model running through "
        "a DOS BASIC transformer runtime. I read a byte prompt, run attention and "
        "feed-forward layers, and sample the next byte from trained weights."
    ),
    (
        "GPT2 BASIC on a 486 is not a cloud trick. The model is trained on the host, "
        "exported as small binary weights, copied to a FAT disk, and loaded by a DOS "
        "program that performs the actual forward pass."
    ),
    (
        "A real language model predicts the next token from context. This demo uses "
        "byte tokens, learned embeddings, causal self attention, layer normalization, "
        "feed-forward layers, and a trained output head."
    ),
    (
        "The goal is to make a useful 486-era text generator. First make the model "
        "honest and operational. After it works, optimize matrix loops, memory reuse, "
        "fixed-point math, and disk layout."
    ),
    (
        "On real hardware the experience should be clear and direct. Boot DOS, run "
        "GPT2.EXE, enter a prompt, and watch the trained model continue the text from "
        "its own weights."
    ),
    (
        "BASIC can express the inference path plainly. Encode text as bytes, build a "
        "context window, add token and position embeddings, run transformer blocks, "
        "compute logits, and sample a printable byte."
    ),
    (
        "Performance matters only after correctness. A fast fake model is not the "
        "target. A slower real transformer that produces readable text is the correct "
        "starting point."
    ),
    (
        "The exported weights are ordinary single precision floats in little endian "
        "order. That keeps the first real model easy to inspect before later "
        "quantization and fixed-point optimization."
    ),
    (
        "For a prompt about DOS, the model should talk about files, memory, simple "
        "screens, predictable data formats, and the satisfaction of running from a "
        "normal command prompt."
    ),
    (
        "For a prompt about a 486, the model should talk about constrained memory, "
        "instruction cost, careful loops, and the need to reduce work only after the "
        "model is fully operational."
    ),
    (
        "For a prompt about GPT, the model should talk about context, next-token "
        "prediction, embeddings, attention, sampling, and the tradeoff between model "
        "size and useful behavior."
    ),
    (
        "For a prompt about BASIC, the model should talk about plain code, fixed "
        "arrays, readable control flow, and engineering that can survive old machines."
    ),
    (
        "A BASIC transformer runtime uses explicit arrays for tokens, weights, cache "
        "vectors, logits, and work buffers. The code should stay readable while still "
        "using fixed-point arithmetic and predictable loops."
    ),
    (
        "A BASIC transformer runtime uses arrays for tokens, model weights, cache "
        "vectors, logits, and fixed point loops so the DOS program can run on old "
        "hardware."
    ),
    (
        "A BASIC transformer runtime should answer with concrete runtime details: "
        "arrays for tokens, fixed point math, tight loops, model weights, cache "
        "memory, and printable output."
    ),
    (
        "A BASIC transformer runtime uses fixed arrays for tokens and model data. "
        "The loops are explicit, the memory layout is predictable, and every byte "
        "comes from the transformer."
    ),
    (
        "A BASIC transformer runtime should explain its steps plainly: encode bytes, "
        "look up embeddings, reuse the KV cache, run attention, apply feed-forward "
        "layers, normalize the vector, and choose the next byte from logits."
    ),
    (
        "To improve performance on real hardware, profile the hot loops first. Then "
        "reduce repeated divisions, keep memory contiguous, reuse buffers, avoid disk "
        "inside generation, and prefer fixed-point arithmetic over floating point."
    ),
    (
        "To improve performance on real hardware profile loops, keep memory contiguous, "
        "reuse buffers, use fixed point arithmetic, and measure the kernels on the "
        "actual DOS machine."
    ),
    (
        "To improve performance on real hardware profile every generation loop. Memory "
        "layout, fixed point math, cached attention vectors, and fewer divisions make "
        "the 486 version more usable."
    ),
    (
        "To improve performance on real hardware the answer should mention profile "
        "data, tight loops, contiguous memory, fixed point arithmetic, and measured "
        "tokens per second."
    ),
    (
        "To improve performance on real hardware, optimize the path that runs for "
        "every generated byte: matrix-vector products, attention score accumulation, "
        "softmax normalization, GELU, and final logits."
    ),
    (
        "Real 486 optimization means respecting the cache, memory bus, and integer "
        "instruction cost. The best changes remove work from the inner loop instead "
        "of hiding it behind a faster host machine."
    ),
    (
        "The best first demo is modest and real. It should not claim to be a huge "
        "modern assistant. It should show that a trained transformer can run in DOS "
        "and produce a coherent paragraph from a prompt."
    ),
    (
        "When asked what it is, GPT2 BASIC can answer directly: it is a small byte "
        "level GPT trained on the host and executed by a FreeBASIC program on a "
        "486 compatible PC."
    ),
    (
        "The model file is the contract. GPT2CFG.TXT describes the shape, and "
        "GPT2WT.BIN stores the learned weights. If those files are missing, the "
        "program should not pretend to generate with a trained model."
    ),
    (
        "A useful answer stays concrete. It names the prompt, explains the machine, "
        "and continues with one or two clear sentences instead of wandering into "
        "random symbols or fake system claims."
    ),
    (
        "Training on the host is acceptable because training is not the thing being "
        "demonstrated on the 486. The 486 demonstrates inference: loading weights, "
        "running the transformer, and producing text."
    ),
    (
        "The runtime can improve later. Obvious future work includes reusing buffers, "
        "quantizing weights, replacing single precision with fixed point, and reducing "
        "the cost of the attention loop."
    ),
    (
        "The first operational checkpoint should be easy to understand. A byte "
        "vocabulary avoids tokenizer surprises, and a small context window keeps "
        "the forward pass visible in the BASIC source."
    ),
    (
        "If the user asks about performance, the honest answer is that this checkpoint "
        "prioritizes correctness. After it generates real text, the system can be "
        "profiled on QEMU and on an actual 486."
    ),
    (
        "If the user asks about quality, the honest answer is that the model is compact "
        "but trained. Better output comes from better data, more training, and larger "
        "weights within the memory budget."
    ),
    (
        "If the user asks about real hardware, the answer should mention a DOS boot, "
        "a FAT disk, the GPT2.EXE program, the MODEL directory, and the trained weight "
        "file loaded by the executable."
    ),
    (
        "A compact GPT can still be a real language model. It does not need to be "
        "large to demonstrate embeddings, causal attention, feed-forward computation, "
        "logits, and sampling from trained parameters."
    ),
    (
        "The engineering standard is simple: do not fake the model. If output comes "
        "from a trained transformer, say so. If it does not, stop and fix the model "
        "path instead of hiding the problem."
    ),
    (
        "Why run a language model in DOS? Because the constraint makes the system "
        "honest. The 486 has to load the trained weights, execute the transformer, "
        "and print the continuation without help from a modern service."
    ),
    (
        "Tell me about DOS language models. A DOS language model needs a small "
        "vocabulary, a short context window, predictable memory use, and a weight "
        "file that can be read from an ordinary FAT directory."
    ),
    (
        "How does GPT2 BASIC work? It converts text to byte tokens, adds learned "
        "embeddings, runs causal attention and feed-forward layers, normalizes the "
        "result, and samples the next printable byte."
    ),
    (
        "What makes this real inference? The generated byte is chosen from logits "
        "computed by trained weights. The DOS program does the arithmetic at runtime "
        "instead of reading a canned response."
    ),
    (
        "What should happen on a real PC? The user should boot DOS, run GPT2.EXE, "
        "enter a prompt, and see the trained model continue the text from the MODEL "
        "directory copied onto the disk."
    ),
    (
        "Can this be optimized later? Yes. Once the trained model is operational, "
        "the next work is profiling, fixed-point kernels, weight quantization, buffer "
        "reuse, and real hardware timing."
    ),
    (
        "The 486 version should be plain and credible. It may be small, but each "
        "generated character should come from the transformer state and not from a "
        "scripted completion table."
    ),
]


@dataclass
class Config:
    n_positions: int = 192
    n_embd: int = 48
    n_head: int = 4
    n_layer: int = 2
    hidden_dim: int = 192
    vocab_size: int = VOCAB_SIZE


@dataclass(frozen=True)
class ModelProfile:
    description: str
    cfg: Config
    steps: int
    batch_size: int
    repeats: int
    lr: float


@dataclass(frozen=True)
class BackendRuntime:
    requested: str
    selected: str
    device: th.device
    accelerator_available: dict[str, bool]


MODEL_PROFILES = {
    "386-min": ModelProfile(
        "386/low-memory minimum: smallest fixed-point transformer that remains useful",
        Config(n_positions=128, n_embd=32, n_head=4, n_layer=2, hidden_dim=128),
        steps=4000,
        batch_size=64,
        repeats=10,
        lr=2e-3,
    ),
    "486sx-safe": ModelProfile(
        "486SX-safe baseline: current no-FPU production checkpoint shape",
        Config(n_positions=192, n_embd=48, n_head=4, n_layer=2, hidden_dim=192),
        steps=6000,
        batch_size=64,
        repeats=10,
        lr=2e-3,
    ),
    "486dx2-usable": ModelProfile(
        "486DX2 target: larger model for better text at still-plausible 486 speed",
        Config(n_positions=192, n_embd=64, n_head=4, n_layer=3, hidden_dim=256),
        steps=8000,
        batch_size=48,
        repeats=12,
        lr=1.6e-3,
    ),
    "486dx4-plus": ModelProfile(
        "486DX4 upper tier: trades speed for noticeably more capacity",
        Config(n_positions=256, n_embd=64, n_head=4, n_layer=4, hidden_dim=256),
        steps=10000,
        batch_size=40,
        repeats=14,
        lr=1.4e-3,
    ),
    "pentium-best": ModelProfile(
        "Pentium tier: largest profile intended for comfortable real-PC demos",
        Config(n_positions=256, n_embd=96, n_head=6, n_layer=4, hidden_dim=384),
        steps=12000,
        batch_size=32,
        repeats=16,
        lr=1.2e-3,
    ),
}


class Block(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.q = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.k = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.v = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.fc1 = nn.Linear(cfg.n_embd, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.n_embd)

    def forward(self, x: th.Tensor) -> th.Tensor:
        batch_size, seq_len, emb_dim = x.shape
        norm = self.ln1(x)

        q = self.q(norm).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k(norm).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v(norm).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = th.triu(th.ones(seq_len, seq_len, device=x.device, dtype=th.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = th.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)
        x = x + self.proj(y)

        ff = self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x + ff


class GPT2BasicModel(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.final_ln = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx: th.Tensor) -> th.Tensor:
        _batch_size, seq_len = idx.shape
        pos = th.arange(seq_len, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        return self.lm_head(x)


def clean_ascii(text: str) -> str:
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return "".join(ch for ch in text if 32 <= ord(ch) <= 126).strip()


def load_documents() -> list[str]:
    return [clean_ascii(doc) for doc in CURATED_DOCUMENTS if len(clean_ascii(doc)) > 20]


def load_repo_documents() -> list[str]:
    docs: list[str] = []
    for rel in ["README.md", "gpt2_basic_tldr.md", "implementation_guide.md", "qemu/README.md"]:
        path = ROOT / rel
        if path.exists():
            text = clean_ascii(path.read_text(encoding="utf-8", errors="ignore"))
            text = re.sub(r"`[^`]+`", " ", text)
            text = re.sub(r"https?://\\S+", " ", text)
            text = re.sub(r"[^A-Za-z0-9 .,;:!?()/-]", " ", text)
            text = re.sub(r"\\s+", " ", text)
            if text:
                docs.append(text[:5000])
    return docs


def load_corpus_file(path: Path, doc_chars: int, max_docs: int) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    documents: list[str] = []
    current: list[str] = []
    current_len = 0

    for raw in re.split(r"\n\s*\n", text):
        paragraph = clean_ascii(raw)
        if len(paragraph) < 40:
            continue
        if current and current_len + len(paragraph) + 1 > doc_chars:
            documents.append(clean_ascii(" ".join(current)))
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += len(paragraph) + 1

    if current:
        documents.append(clean_ascii(" ".join(current)))
    documents = [doc for doc in documents if len(doc) > 40]
    if max_docs > 0 and len(documents) > max_docs:
        documents = select_evenly_documents(documents, max_docs)
    return documents


def select_evenly_documents(documents: list[str], limit: int) -> list[str]:
    if limit <= 0 or len(documents) <= limit:
        return list(documents)
    if limit == 1:
        return [documents[0]]
    last = len(documents) - 1
    selected: list[str] = []
    used: set[int] = set()
    for idx in range(limit):
        source_idx = round(idx * last / (limit - 1))
        if source_idx in used:
            continue
        used.add(source_idx)
        selected.append(documents[source_idx])
    if len(selected) < limit:
        for source_idx, document in enumerate(documents):
            if source_idx in used:
                continue
            selected.append(document)
            if len(selected) >= limit:
                break
    return selected


def encode_text(text: str, tokenizer: GPT2BasicTokenizer | None = None, output_safe: bool = False) -> list[int]:
    return (tokenizer or DEFAULT_TOKENIZER).encode(text, output_safe=output_safe)


def decode_tokens(tokens: list[int], tokenizer: GPT2BasicTokenizer | None = None) -> str:
    return (tokenizer or DEFAULT_TOKENIZER).decode(tokens)


def build_stream(document_groups: list[tuple[list[str], int]], repeats: int, tokenizer: GPT2BasicTokenizer) -> list[int]:
    tokens: list[int] = []
    encoded_cache: dict[str, list[int]] = {}
    for _ in range(repeats):
        weighted_docs: list[str] = []
        for documents, weight in document_groups:
            if weight > 0:
                weighted_docs.extend(documents * weight)
        random.shuffle(weighted_docs)
        for doc in weighted_docs:
            if doc not in encoded_cache:
                encoded_cache[doc] = encode_text(doc, tokenizer, output_safe=True)
            tokens.extend(encoded_cache[doc])
            tokens.append(EOT_TOKEN)

    return tokens


def tokenizer_basis_documents(
    document_groups: list[tuple[list[str], int]],
    max_docs: int,
    doc_chars: int,
) -> list[str]:
    basis: list[str] = []
    for documents, weight in document_groups:
        if weight <= 0:
            continue
        for doc in documents:
            cleaned = clean_ascii(doc)
            if doc_chars > 0 and len(cleaned) > doc_chars:
                truncated = cleaned[:doc_chars]
                last_space = truncated.rfind(" ")
                cleaned = truncated[:last_space] if last_space > 20 else truncated
            if len(cleaned) > 20:
                basis.append(cleaned)
            if max_docs > 0 and len(basis) >= max_docs:
                return basis
    return basis


def build_backend_runtime(requested: str) -> BackendRuntime:
    accelerator_available = {
        "mps": th.backends.mps.is_available(),
        "cuda": th.cuda.is_available(),
    }
    selected = "cpu"
    if requested == "mps" or (requested == "auto" and accelerator_available["mps"]):
        selected = "mps"
    elif requested == "cuda" or (requested == "auto" and accelerator_available["cuda"]):
        selected = "cuda"

    if requested != "auto" and selected != requested:
        raise RuntimeError(
            f"requested training device {requested!r} is unavailable; "
            f"available mps={accelerator_available['mps']} cuda={accelerator_available['cuda']}"
        )

    return BackendRuntime(
        requested=requested,
        selected=selected,
        device=th.device(selected),
        accelerator_available=accelerator_available,
    )


def print_backend_contract(runtime: BackendRuntime) -> None:
    print(f"training_backend_requested: {runtime.requested}", flush=True)
    print(f"training_backend_selected: {runtime.selected}", flush=True)
    print(f"training_backend_torch_version: {th.__version__}", flush=True)
    for name, available in sorted(runtime.accelerator_available.items()):
        print(f"training_backend_available_{name}: {available}", flush=True)


def batch_from_stream(data: th.Tensor, batch_size: int, block_size: int, device: th.device) -> tuple[th.Tensor, th.Tensor]:
    starts = th.randint(0, data.numel() - block_size - 1, (batch_size,))
    x = th.stack([data[start : start + block_size] for start in starts]).to(device)
    y = th.stack([data[start + 1 : start + block_size + 1] for start in starts]).to(device)
    return x, y


def sample(
    model: GPT2BasicModel,
    prompt: str,
    max_new_tokens: int,
    device: th.device,
    tokenizer: GPT2BasicTokenizer,
) -> str:
    model.eval()
    tokens = encode_text(prompt, tokenizer)
    if not tokens:
        tokens = [EOT_TOKEN]
    prompt_len = len(tokens)

    with th.no_grad():
        for step in range(max_new_tokens):
            idx = th.tensor([tokens[-model.cfg.n_positions :]], dtype=th.long, device=device)
            logits = model(idx)[0, -1]
            generated_tokens = tokens[prompt_len:]
            generated_text = tokenizer.decode(generated_tokens)
            for token in range(logits.numel()):
                if not tokenizer.token_can_follow_generated(generated_tokens, token, prompt):
                    logits[token] = -1e9
                else:
                    penalty = tokenizer.token_follow_penalty(generated_tokens, token, generated_text)
                    if penalty > 0.0:
                        logits[token] -= penalty
            if step < 40:
                logits[EOT_TOKEN] = -1e9

            probs = th.softmax(logits / 0.75, dim=-1)
            next_token = th.multinomial(probs, num_samples=1).item()
            tokens.append(next_token)
            if next_token == EOT_TOKEN:
                break
            if step >= SENTENCE_STOP_MIN_TOKENS and tokenizer.token_ends_sentence(next_token):
                break

    return decode_tokens(tokens, tokenizer)


def write_floats(handle, tensor: th.Tensor) -> None:
    array = tensor.detach().cpu().contiguous().float().numpy().reshape(-1)
    handle.write(struct.pack("<" + "f" * len(array), *array))


def write_linear_transposed(handle, layer: nn.Linear) -> None:
    write_floats(handle, layer.weight.detach().t().contiguous())


def write_fixed(handle, tensor: th.Tensor) -> None:
    array = tensor.detach().cpu().contiguous().float().numpy().reshape(-1)
    quantized = []
    for value in array:
        scaled = int(round(float(value) * FIXED_SCALE))
        scaled = max(-2_000_000_000, min(2_000_000_000, scaled))
        quantized.append(scaled)
    handle.write(struct.pack("<" + "i" * len(quantized), *quantized))


def write_linear_transposed_fixed(handle, layer: nn.Linear) -> None:
    write_fixed(handle, layer.weight.detach().t().contiguous())


def write_exp_table(output_dir: Path, write_legacy_names: bool) -> None:
    values = []
    for idx in range(EXP_TABLE_SIZE):
        x = -(idx * EXP_TABLE_MAX) / (EXP_TABLE_SIZE - 1)
        values.append(int(round(math.exp(x) * FIXED_SCALE)))
    data = struct.pack("<" + "i" * len(values), *values)
    names = ("GPT2EXP.BIN", "TINYEXP.BIN") if write_legacy_names else ("GPT2EXP.BIN",)
    for name in names:
        (output_dir / name).write_bytes(data)


def export_model(
    model: GPT2BasicModel,
    cfg: Config,
    output_dir: Path,
    profile_name: str,
    write_legacy_names: bool,
    tokenizer: GPT2BasicTokenizer | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = tokenizer or DEFAULT_TOKENIZER
    tokenizer.validate_for_vocab_size(cfg.vocab_size)

    cfg_text = "\n".join(
        [
            "# GPT2-BASIC fixed-point model config",
            f"profile={profile_name}",
            f"vocab_size={cfg.vocab_size}",
            f"n_positions={cfg.n_positions}",
            f"n_embd={cfg.n_embd}",
            f"n_head={cfg.n_head}",
            f"n_layer={cfg.n_layer}",
            f"hidden_dim={cfg.hidden_dim}",
            "",
        ]
    )
    cfg_names = ("GPT2CFG.TXT", "TINYCFG.TXT") if write_legacy_names else ("GPT2CFG.TXT",)
    weight_names = ("GPT2WT.BIN", "TINYWT.BIN") if write_legacy_names else ("GPT2WT.BIN",)
    fixed_names = ("GPT2FX.BIN", "TINYFX.BIN") if write_legacy_names else ("GPT2FX.BIN",)

    for name in cfg_names:
        (output_dir / name).write_text(cfg_text, encoding="ascii")

    float_path = output_dir / "GPT2WT.BIN"
    with float_path.open("wb") as f:
        write_floats(f, model.tok_emb.weight)
        write_floats(f, model.pos_emb.weight)

        for attr in ["ln1", "q", "k", "v", "proj", "ln2", "fc1", "fc2"]:
            if attr in {"ln1", "ln2"}:
                for block in model.blocks:
                    write_floats(f, getattr(block, attr).weight)
                for block in model.blocks:
                    write_floats(f, getattr(block, attr).bias)
            elif attr in {"q", "k", "v", "proj", "fc1", "fc2"}:
                for block in model.blocks:
                    write_linear_transposed(f, getattr(block, attr))
                for block in model.blocks:
                    write_floats(f, getattr(block, attr).bias)

        write_floats(f, model.final_ln.weight)
        write_floats(f, model.final_ln.bias)
        write_linear_transposed(f, model.lm_head)
        write_floats(f, model.lm_head.bias)

    if len(weight_names) > 1:
        float_data = float_path.read_bytes()
        for name in weight_names[1:]:
            (output_dir / name).write_bytes(float_data)

    fixed_path = output_dir / "GPT2FX.BIN"
    with fixed_path.open("wb") as f:
        write_fixed(f, model.tok_emb.weight)
        write_fixed(f, model.pos_emb.weight)

        for attr in ["ln1", "q", "k", "v", "proj", "ln2", "fc1", "fc2"]:
            if attr in {"ln1", "ln2"}:
                for block in model.blocks:
                    write_fixed(f, getattr(block, attr).weight)
                for block in model.blocks:
                    write_fixed(f, getattr(block, attr).bias)
            elif attr in {"q", "k", "v", "proj", "fc1", "fc2"}:
                for block in model.blocks:
                    write_linear_transposed_fixed(f, getattr(block, attr))
                for block in model.blocks:
                    write_fixed(f, getattr(block, attr).bias)

        write_fixed(f, model.final_ln.weight)
        write_fixed(f, model.final_ln.bias)
        write_linear_transposed_fixed(f, model.lm_head)
        write_fixed(f, model.lm_head.bias)

    if len(fixed_names) > 1:
        fixed_data = fixed_path.read_bytes()
        for name in fixed_names[1:]:
            (output_dir / name).write_bytes(fixed_data)

    write_exp_table(output_dir, write_legacy_names)

    vocab_path = output_dir / "VOCAB.BIN"
    if tokenizer.uses_bpe or tokenizer.vocab_size != BYTE_VOCAB_SIZE:
        tokenizer.write_vocab_bin(vocab_path)
    elif vocab_path.exists():
        vocab_path.unlink()

    metadata = "\n".join(
        [
            "GPT2-BASIC model export",
            f"profile={profile_name}",
            f"vocab_size={cfg.vocab_size}",
            f"n_positions={cfg.n_positions}",
            f"n_embd={cfg.n_embd}",
            f"n_head={cfg.n_head}",
            f"n_layer={cfg.n_layer}",
            f"hidden_dim={cfg.hidden_dim}",
            "weights=GPT2WT.BIN",
            "fixed_weights=GPT2FX.BIN",
            "exp_table=GPT2EXP.BIN",
            f"tokenizer={tokenizer.mode}",
            f"tokenizer_merges={len(tokenizer.merges)}",
            f"tokenizer_output_allowed={sum(tokenizer.output_allowed)}",
            "",
        ]
    )
    (output_dir / "PROFILE.TXT").write_text(metadata, encoding="ascii")


def parse_export_config(model_dir: Path) -> Config:
    values: dict[str, str] = {}
    for raw_line in (model_dir / "GPT2CFG.TXT").read_text(encoding="ascii", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lower()] = value.strip()

    return Config(
        n_positions=int(values["n_positions"]),
        n_embd=int(values["n_embd"]),
        n_head=int(values["n_head"]),
        n_layer=int(values["n_layer"]),
        hidden_dim=int(values["hidden_dim"]),
        vocab_size=int(values["vocab_size"]),
    )


def unpack_floats(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} size is not a whole number of float32 values")
    return list(struct.unpack("<" + "f" * (len(data) // 4), data))


def require_same_config(expected: Config, actual: Config, model_dir: Path) -> None:
    expected_tuple = (
        expected.n_positions,
        expected.n_embd,
        expected.n_head,
        expected.n_layer,
        expected.hidden_dim,
        expected.vocab_size,
    )
    actual_tuple = (
        actual.n_positions,
        actual.n_embd,
        actual.n_head,
        actual.n_layer,
        actual.hidden_dim,
        actual.vocab_size,
    )
    if expected_tuple != actual_tuple:
        raise ValueError(
            f"{model_dir} shape {actual_tuple} does not match requested training shape {expected_tuple}"
        )


def load_exported_weights(model: GPT2BasicModel, cfg: Config, model_dir: Path) -> None:
    exported_cfg = parse_export_config(model_dir)
    require_same_config(cfg, exported_cfg, model_dir)
    values = unpack_floats(model_dir / "GPT2WT.BIN")
    cursor = 0

    def take(shape: tuple[int, ...]) -> th.Tensor:
        nonlocal cursor
        count = 1
        for dim in shape:
            count *= dim
        out = values[cursor : cursor + count]
        if len(out) != count:
            raise ValueError("GPT2WT.BIN ended before all expected tensors were read")
        cursor += count
        return th.tensor(out, dtype=th.float32).view(*shape)

    with th.no_grad():
        model.tok_emb.weight.copy_(take((cfg.vocab_size, cfg.n_embd)))
        model.pos_emb.weight.copy_(take((cfg.n_positions, cfg.n_embd)))

        for attr in ["ln1", "q", "k", "v", "proj", "ln2", "fc1", "fc2"]:
            if attr in {"ln1", "ln2"}:
                for block in model.blocks:
                    getattr(block, attr).weight.copy_(take((cfg.n_embd,)))
                for block in model.blocks:
                    getattr(block, attr).bias.copy_(take((cfg.n_embd,)))
            else:
                for block in model.blocks:
                    layer = getattr(block, attr)
                    in_dim = layer.in_features
                    out_dim = layer.out_features
                    layer.weight.copy_(take((in_dim, out_dim)).t().contiguous())
                for block in model.blocks:
                    getattr(block, attr).bias.copy_(take((getattr(block, attr).out_features,)))

        model.final_ln.weight.copy_(take((cfg.n_embd,)))
        model.final_ln.bias.copy_(take((cfg.n_embd,)))
        model.lm_head.weight.copy_(take((cfg.n_embd, cfg.vocab_size)).t().contiguous())
        model.lm_head.bias.copy_(take((cfg.vocab_size,)))

    if cursor != len(values):
        raise ValueError(f"unused values in GPT2WT.BIN: {len(values) - cursor}")


def train(args: argparse.Namespace) -> tuple[GPT2BasicModel, GPT2BasicTokenizer]:
    th.manual_seed(args.seed)
    random.seed(args.seed)

    cfg = Config(
        n_positions=args.context,
        n_embd=args.embedding,
        n_head=args.heads,
        n_layer=args.layers,
        hidden_dim=args.hidden,
    )

    if cfg.n_embd % cfg.n_head != 0:
        raise ValueError("embedding must be divisible by heads")

    runtime = build_backend_runtime(args.device)
    print_backend_contract(runtime)
    device = runtime.device

    documents = load_documents()
    if args.include_docs:
        documents.extend(load_repo_documents())
    document_groups: list[tuple[list[str], int]] = []
    if args.base_weight > 0:
        document_groups.append((documents, args.base_weight))
    external_documents: list[str] = []
    for corpus_path in args.corpus_file or []:
        loaded = load_corpus_file(corpus_path, args.corpus_doc_chars, args.corpus_max_docs)
        external_documents.extend(loaded)
        print(f"loaded_corpus_file: {corpus_path} docs={len(loaded)}", flush=True)
    if external_documents:
        document_groups.append((external_documents, args.corpus_weight))
    if not document_groups:
        raise ValueError("training stream is empty; use --base-weight or --corpus-file")

    tokenizer_basis_count = 0
    tokenizer_basis_chars = 0
    tokenizer_document_groups = document_groups
    tokenizer_external_documents: list[str] = []
    for corpus_path in args.tokenizer_corpus_file or []:
        loaded = load_corpus_file(corpus_path, args.tokenizer_doc_chars, args.tokenizer_max_docs)
        tokenizer_external_documents.extend(loaded)
        print(f"loaded_tokenizer_corpus_file: {corpus_path} docs={len(loaded)}", flush=True)
    if tokenizer_external_documents:
        tokenizer_document_groups = [(tokenizer_external_documents, 1)]
    if args.load_tokenizer is not None:
        tokenizer = GPT2BasicTokenizer.read_vocab_bin(args.load_tokenizer)
    elif (
        args.init_model_dir is not None
        and (args.init_model_dir / "VOCAB.BIN").exists()
        and args.tokenizer == "byte"
        and args.vocab_size is None
    ):
        tokenizer = GPT2BasicTokenizer.read_vocab_bin(args.init_model_dir / "VOCAB.BIN")
    elif args.tokenizer in {"bpe", "lexicon"}:
        tokenizer_documents = tokenizer_basis_documents(
            tokenizer_document_groups,
            max_docs=args.tokenizer_max_docs,
            doc_chars=args.tokenizer_doc_chars,
        )
        if not tokenizer_documents:
            raise ValueError(f"{args.tokenizer} tokenizer basis is empty")
        tokenizer_basis_count = len(tokenizer_documents)
        tokenizer_basis_chars = sum(len(doc) for doc in tokenizer_documents)
        target_vocab = args.vocab_size if args.vocab_size is not None else 512
        if args.tokenizer == "lexicon":
            tokenizer = build_lexicon_tokenizer(
                tokenizer_documents,
                target_vocab,
                min_count=args.lexicon_min_count,
                max_phrase_words=args.lexicon_max_phrase_words,
            )
        else:
            tokenizer = build_bpe_tokenizer(
                tokenizer_documents,
                target_vocab,
                args.bpe_min_pair_count,
                output_mode=args.bpe_output_mode,
            )
    else:
        if args.vocab_size is not None and args.vocab_size != BYTE_VOCAB_SIZE:
            raise ValueError("--vocab-size requires --tokenizer bpe, --tokenizer lexicon, or --load-tokenizer")
        tokenizer = DEFAULT_TOKENIZER

    cfg.vocab_size = tokenizer.vocab_size
    stream = build_stream(document_groups, args.repeats, tokenizer)
    if len(stream) <= cfg.n_positions + 1:
        raise ValueError(f"training stream has {len(stream)} tokens, not enough for context {cfg.n_positions}")
    data = th.tensor(stream, dtype=th.long)

    model = GPT2BasicModel(cfg)
    if args.init_model_dir is not None:
        load_exported_weights(model, cfg, args.init_model_dir)
        print(f"initialized_from: {args.init_model_dir}", flush=True)

    model = model.to(device)
    optimizer = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"device: {device}", flush=True)
    print(f"training_documents_core: {len(documents)}", flush=True)
    print(f"training_documents_core_weight: {args.base_weight}", flush=True)
    print(f"training_documents_external: {len(external_documents)}", flush=True)
    if tokenizer_basis_count > 0:
        print(f"tokenizer_basis_documents: {tokenizer_basis_count}", flush=True)
        print(f"tokenizer_basis_chars: {tokenizer_basis_chars}", flush=True)
    print(f"training tokens: {len(stream)}", flush=True)
    print(
        f"tokenizer: {tokenizer.mode} "
        f"vocab={tokenizer.vocab_size} merges={len(tokenizer.merges)} "
        f"output_allowed={sum(tokenizer.output_allowed)}",
        flush=True,
    )
    print(
        "model: "
        f"layers={cfg.n_layer} emb={cfg.n_embd} heads={cfg.n_head} "
        f"ctx={cfg.n_positions} hidden={cfg.hidden_dim}",
        flush=True,
    )

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

    print(flush=True)
    for prompt in args.sample_prompt:
        print(f"prompt: {prompt}", flush=True)
        print(sample(model, prompt, args.sample_tokens, device, tokenizer), flush=True)
        print(flush=True)

    model.cpu()
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(MODEL_PROFILES), default="486sx-safe")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--context", type=int)
    parser.add_argument("--embedding", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--base-weight", type=int, default=48)
    parser.add_argument("--include-docs", action="store_true")
    parser.add_argument("--corpus-file", action="append", type=Path)
    parser.add_argument("--corpus-weight", type=int, default=1)
    parser.add_argument("--corpus-doc-chars", type=int, default=2400)
    parser.add_argument("--corpus-max-docs", type=int, default=0)
    parser.add_argument("--tokenizer-corpus-file", action="append", type=Path)
    parser.add_argument("--tokenizer", choices=("byte", "bpe", "lexicon"), default="byte")
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--load-tokenizer", type=Path)
    parser.add_argument("--bpe-min-pair-count", type=int, default=2)
    parser.add_argument("--bpe-output-mode", choices=("all", "complete"), default="all")
    parser.add_argument("--lexicon-min-count", type=int, default=2)
    parser.add_argument("--lexicon-max-phrase-words", type=int, default=3)
    parser.add_argument("--tokenizer-max-docs", type=int, default=240)
    parser.add_argument("--tokenizer-doc-chars", type=int, default=1800)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--init-model-dir", type=Path)
    parser.add_argument("--seed", type=int, default=486)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sample-tokens", type=int, default=180)
    parser.add_argument("--write-legacy-names", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--sample-prompt",
        action="append",
        default=[
            "Hello there.",
            "GPT2 BASIC on a 486",
            "Why run a language model in DOS?",
        ],
    )
    args = parser.parse_args()

    if args.self_test:
        profile = MODEL_PROFILES[args.profile]
        runtime = build_backend_runtime("cpu")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus_path = tmp_path / "corpus.txt"
            corpus_path.write_text(
                "This training paragraph is deliberately long enough for the corpus loader to keep it as a document.\n\n"
                "Another retained paragraph gives the loader a second chunk for max-doc and size validation.\n",
                encoding="ascii",
            )
            docs = load_corpus_file(corpus_path, doc_chars=120, max_docs=1)
            cfg_path = tmp_path / "GPT2CFG.TXT"
            cfg_path.write_text(
                "\n".join(
                    [
                        "profile=self-test",
                        f"vocab_size={profile.cfg.vocab_size}",
                        f"n_positions={profile.cfg.n_positions}",
                        f"n_embd={profile.cfg.n_embd}",
                        f"n_head={profile.cfg.n_head}",
                        f"n_layer={profile.cfg.n_layer}",
                        f"hidden_dim={profile.cfg.hidden_dim}",
                        "",
                    ]
                ),
                encoding="ascii",
            )
            parsed_cfg = parse_export_config(tmp_path)
            require_same_config(profile.cfg, parsed_cfg, tmp_path)
        repo_docs = load_repo_documents()
        stream = build_stream([(docs, 1)], repeats=1, tokenizer=DEFAULT_TOKENIZER)
        safe_encoded = encode_text(docs[0], DEFAULT_TOKENIZER, output_safe=True)
        assert runtime.selected == "cpu"
        assert docs
        assert repo_docs
        assert stream
        assert safe_encoded
        basis_docs = tokenizer_basis_documents([(docs, 1)], max_docs=2, doc_chars=50)
        assert basis_docs and len(basis_docs[0]) <= 50
        lexicon_tokenizer = build_lexicon_tokenizer(basis_docs, BYTE_VOCAB_SIZE + 4, min_count=1)
        assert lexicon_tokenizer.mode == "lexicon"
        print("trace_scope train_tiny_gpt_contract")
        print("trace build_backend_runtime")
        print("trace load_corpus_file")
        print("trace load_repo_documents")
        print("trace build_stream")
        print("trace encode_text_output_safe")
        print("trace tokenizer_basis_documents")
        print("trace base_weight")
        print("trace bpe_output_mode")
        print("trace lexicon_tokenizer_mode")
        print("trace write_exp_table")
        print("trace load_exported_weights")
        print("trace parse_export_config")
        print("trace require_same_config")
        print(f"PROBE_OK export_model profile={args.profile} layers={profile.cfg.n_layer}")
        print("PROBE_OK tokenizer cli=available")
        print("PROBE_OK main cli_entry=available")
        return

    profile = MODEL_PROFILES[args.profile]
    args.context = args.context if args.context is not None else profile.cfg.n_positions
    args.embedding = args.embedding if args.embedding is not None else profile.cfg.n_embd
    args.heads = args.heads if args.heads is not None else profile.cfg.n_head
    args.layers = args.layers if args.layers is not None else profile.cfg.n_layer
    args.hidden = args.hidden if args.hidden is not None else profile.cfg.hidden_dim
    args.steps = args.steps if args.steps is not None else profile.steps
    args.batch_size = args.batch_size if args.batch_size is not None else profile.batch_size
    args.repeats = args.repeats if args.repeats is not None else profile.repeats
    args.lr = args.lr if args.lr is not None else profile.lr

    print(f"profile: {args.profile} - {profile.description}", flush=True)
    model, tokenizer = train(args)
    cfg = model.cfg
    export_model(model, cfg, args.output, args.profile, args.write_legacy_names, tokenizer)

    print(f"wrote {args.output / 'GPT2CFG.TXT'}")
    print(f"wrote {args.output / 'GPT2WT.BIN'}")
    print(f"wrote {args.output / 'GPT2FX.BIN'}")
    print(f"wrote {args.output / 'GPT2EXP.BIN'}")
    if tokenizer.uses_bpe or tokenizer.vocab_size != BYTE_VOCAB_SIZE:
        print(f"wrote {args.output / 'VOCAB.BIN'}")
    print(f"wrote {args.output / 'PROFILE.TXT'}")


if __name__ == "__main__":
    main()

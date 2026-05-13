#!/usr/bin/env python3
"""Tokenizer contract shared by GPT2-BASIC host tools and DOS runtime.

The DOS tokenizer reads VOCAB.BIN as a compact fixed-record file. This module
keeps the host side honest by using the same token IDs, merge records, maximum
piece length, and rank-based BPE merge loop.
"""

from __future__ import annotations

import argparse
import re
import struct
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


EOT_TOKEN = 0
UNK_TOKEN = 1
BYTE_OFFSET = 2
BYTE_VOCAB_SIZE = 258
MAX_TOKEN_LENGTH = 16
DOS_TOKEN_HASH_LIMIT = 4096
OUTPUT_MASK_MAGIC = 0x4B53414D  # "MASK" little-endian marker after merge records.
TOKENIZER_MODE_MAGIC = 0x45444F4D  # "MODE" little-endian marker after mask records.
TOKENIZER_MODE_CODES = {"byte": 0, "bpe": 1, "lexicon": 2}
TOKENIZER_MODE_NAMES = {value: key for key, value in TOKENIZER_MODE_CODES.items()}
PROMPT_STARTER_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "dos language models need",
        (" compact", " small", " prompt", " short", " predictable", " plain", " local", " enough"),
    ),
    (
        "a basic transformer runtime",
        (" uses", " loads", " keeps"),
    ),
    (
        "to improve performance on real hardware",
        (" reduce", " choose", " measure", " keep"),
    ),
)


@dataclass(frozen=True)
class MergeOp:
    first: int
    second: int
    result: int
    priority: int


def clean_ascii(text: str) -> str:
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return "".join(ch for ch in text if 32 <= ord(ch) <= 126).strip()


def default_output_allowed(pieces: list[bytes]) -> list[bool]:
    allowed = [True] * len(pieces)
    if len(allowed) > EOT_TOKEN:
        allowed[EOT_TOKEN] = True
    if len(allowed) > UNK_TOKEN:
        allowed[UNK_TOKEN] = False
    for token_id in range(BYTE_OFFSET, min(BYTE_VOCAB_SIZE, len(allowed))):
        byte = token_id - BYTE_OFFSET
        allowed[token_id] = 32 <= byte <= 126
    return allowed


def complete_piece_inventory(documents: list[str], max_piece_chars: int = MAX_TOKEN_LENGTH) -> set[bytes]:
    pieces: set[bytes] = set()
    token_pattern = r"[A-Za-z0-9][A-Za-z0-9.+/_-]*"
    for document in documents:
        text = clean_ascii(document)
        words = re.findall(token_pattern, text)
        for word in words:
            if len(word) < 2 and word not in {"A", "I", "a"}:
                continue
            variants = [word, " " + word, word + " ", " " + word + " "]
            for variant in variants:
                if 1 <= len(variant) <= max_piece_chars:
                    pieces.add(variant.encode("ascii"))

        for size in (2, 3):
            for idx in range(0, max(0, len(words) - size + 1)):
                phrase = " ".join(words[idx : idx + size])
                variants = [phrase, " " + phrase, phrase + " ", " " + phrase + " "]
                for variant in variants:
                    if 1 <= len(variant) <= max_piece_chars:
                        pieces.add(variant.encode("ascii"))

        for piece in response_sentence_pieces(text, max_piece_chars, max_phrase_words=4):
            pieces.add(piece.encode("ascii"))

    for punct in [" ", ". ", ", ", ": ", "; ", "! ", "? ", " - ", " / ", "(", ")", "%", "_", "-", "+", "'", '"']:
        if len(punct) <= max_piece_chars:
            pieces.add(punct.encode("ascii"))
    return pieces


def complete_output_allowed(pieces: list[bytes], documents: list[str]) -> list[bool]:
    inventory = complete_piece_inventory(documents)
    allowed = default_output_allowed(pieces)
    for token_id in range(BYTE_VOCAB_SIZE, len(pieces)):
        piece = pieces[token_id]
        text = piece.decode("ascii", errors="ignore")
        has_alpha = any(ch.isalpha() for ch in text)
        allowed[token_id] = piece in inventory or not has_alpha
    return allowed


def lexicon_candidates(
    documents: list[str],
    max_piece_chars: int = MAX_TOKEN_LENGTH,
    max_phrase_words: int = 3,
    min_count: int = 2,
) -> list[bytes]:
    """Rank complete word/phrase pieces for longest-match lexicon tokenization."""
    counts: Counter[str] = Counter()
    token_pattern = r"[A-Za-z0-9][A-Za-z0-9.+/_-]*"
    # Keep punctuation tokens from consuming following spaces. Lexicon word tokens
    # are space-prefixed, so a token such as ", " forces the next word down the
    # byte spelling path instead of letting " word" match.
    punctuation_pattern = r"[.,:;!?()%_+\-/]"

    for document in documents:
        text = clean_ascii(document)
        if not text:
            continue

        word_matches = list(re.finditer(token_pattern, text))
        words = [match.group(0) for match in word_matches]
        for phrase_len in range(1, max_phrase_words + 1):
            if phrase_len > len(words):
                break
            for idx in range(0, len(words) - phrase_len + 1):
                if not plain_spaced_word_phrase(text, word_matches, idx, phrase_len):
                    continue
                phrase = " ".join(words[idx : idx + phrase_len])
                piece = " " + phrase
                if 2 <= len(piece) <= max_piece_chars and word_matches[idx].start() > 0:
                    counts[piece] += 1
                if idx == 0 and 1 <= len(phrase) <= max_piece_chars and text.startswith(phrase):
                    counts[phrase] += 1

        for match in re.finditer(punctuation_pattern, text):
            piece = match.group(0)
            if 2 <= len(piece) <= max_piece_chars:
                counts[piece] += 1

        for piece in response_sentence_pieces(text, max_piece_chars, max_phrase_words=max_phrase_words):
            # These are the highest-value pieces for a tiny chat model: short
            # response endings such as " in DOS.", " is real.", and
            # " small project." keep the model from spelling common answers
            # byte by byte while preserving real token sampling.
            counts[piece] += 8

    ranked: list[tuple[int, int, int, str]] = []
    for piece, count in counts.items():
        if count < min_count:
            continue
        if len(piece) > max_piece_chars:
            continue
        if not any(ch.isalpha() or ch.isdigit() for ch in piece) and len(piece) <= 1:
            continue
        savings = count * (len(piece) - 1)
        ranked.append((-savings, -count, -len(piece), piece))

    return [piece.encode("ascii") for _savings, _count, _length, piece in sorted(ranked)]


def plain_spaced_word_phrase(text: str, matches: list[re.Match[str]], start_idx: int, phrase_len: int) -> bool:
    """Return true when adjacent word matches are separated only by one space."""
    if phrase_len <= 1:
        return True
    for idx in range(start_idx, start_idx + phrase_len - 1):
        gap = text[matches[idx].end() : matches[idx + 1].start()]
        if gap != " ":
            return False
    return True


def response_sentence_spans(text: str) -> list[str]:
    """Extract short assistant/answer spans from dialogue-like training text."""
    spans: list[str] = []
    for match in re.finditer(r"(?:Assistant:| A:)\s*([^.!?]{2,96}[.!?])", text):
        span = clean_ascii(match.group(1))
        if span:
            spans.append(span)
    return spans


def response_sentence_pieces(
    text: str,
    max_piece_chars: int = MAX_TOKEN_LENGTH,
    max_phrase_words: int = 4,
) -> set[str]:
    """Return bounded phrase pieces that preserve answer punctuation.

    VOCAB.BIN currently stores 16-byte pieces for DOS compatibility, so this
    intentionally builds sentence *pieces* rather than whole arbitrary
    sentences. The selected pieces are still more expressive than plain word
    n-grams because the best answer tails include punctuation and stop cleanly.
    """
    pieces: set[str] = set()
    token_pattern = r"[A-Za-z0-9][A-Za-z0-9.+/_-]*"

    def add_piece(piece: str) -> None:
        if 1 <= len(piece) <= max_piece_chars:
            pieces.add(piece)

    for span in response_sentence_spans(text):
        words = re.findall(token_pattern, span)
        if not words:
            continue
        terminal = span.rstrip()[-1] if span.rstrip()[-1:] in ".!?" else ""
        for start in range(len(words)):
            for end in range(start + 1, min(len(words), start + max_phrase_words) + 1):
                phrase = " ".join(words[start:end])
                for variant in (phrase, " " + phrase):
                    add_piece(variant)
                if terminal and end == len(words):
                    punct_phrase = phrase + terminal
                    for variant in (punct_phrase, " " + punct_phrase):
                        add_piece(variant)

    return pieces


class GPT2BasicTokenizer:
    def __init__(
        self,
        pieces: list[bytes],
        merges: list[MergeOp] | None = None,
        output_allowed: list[bool] | None = None,
        mode: str | None = None,
    ) -> None:
        if len(pieces) < BYTE_VOCAB_SIZE:
            raise ValueError("tokenizer must contain the two special tokens and 256 byte tokens")
        self.id_to_piece = list(pieces)
        self.merges = list(merges or [])
        self.output_allowed = list(output_allowed) if output_allowed is not None else default_output_allowed(self.id_to_piece)
        if mode is None:
            if self.merges:
                mode = "bpe"
            elif len(pieces) > BYTE_VOCAB_SIZE:
                mode = "lexicon"
            else:
                mode = "byte"
        if mode not in TOKENIZER_MODE_CODES:
            raise ValueError(f"unknown tokenizer mode: {mode}")
        self.mode = mode
        self.merge_by_pair: dict[tuple[int, int], MergeOp] = {}
        for merge in self.merges:
            pair = (merge.first, merge.second)
            existing = self.merge_by_pair.get(pair)
            if existing is None or merge.priority < existing.priority:
                self.merge_by_pair[pair] = merge
        self.lexicon_by_first: dict[int, list[tuple[bytes, int]]] = {}
        self.lexicon_by_piece: dict[bytes, int] = {}
        self.lexicon_lengths_by_first: dict[int, tuple[int, ...]] = {}
        if self.mode == "lexicon":
            lengths_by_first: dict[int, set[int]] = {}
            for token_id in range(BYTE_VOCAB_SIZE, len(self.id_to_piece)):
                piece = self.id_to_piece[token_id]
                if piece:
                    self.lexicon_by_piece[piece] = token_id
                    lengths_by_first.setdefault(piece[0], set()).add(len(piece))
                    self.lexicon_by_first.setdefault(piece[0], []).append((piece, token_id))
            for first in self.lexicon_by_first:
                self.lexicon_by_first[first].sort(key=lambda item: (-len(item[0]), item[1]))
            self.lexicon_lengths_by_first = {
                first: tuple(sorted(lengths, reverse=True))
                for first, lengths in lengths_by_first.items()
            }
        self._prompt_start_cache: dict[str, set[int] | None] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_piece)

    @property
    def uses_bpe(self) -> bool:
        return self.mode == "bpe" and bool(self.merges)

    def token_allowed(self, token_id: int) -> bool:
        return 0 <= token_id < self.vocab_size and self.output_allowed[token_id]

    def token_can_begin_output(self, token_id: int) -> bool:
        if not self.token_allowed(token_id):
            return False
        if token_id == EOT_TOKEN:
            return False
        piece = self.id_to_piece[token_id]
        if BYTE_OFFSET <= token_id < BYTE_VOCAB_SIZE:
            return piece == b" "
        if not piece.startswith(b" "):
            return False
        stripped = piece.lstrip()
        if not stripped:
            return True
        first = chr(stripped[0])
        return first.isalnum() or first in {'"', "'", "("}

    @staticmethod
    def prompt_start_starters(prompt_text: str | None) -> tuple[str, ...]:
        if not prompt_text:
            return ()
        prompt = clean_ascii(prompt_text).lower().strip()
        prompt = re.sub(r"[\s.!?:;]+$", "", prompt)
        for suffix, starters in PROMPT_STARTER_RULES:
            if prompt.endswith(suffix):
                return starters
        return ()

    @staticmethod
    def _starter_piece_matches(piece_text: str, starter: str) -> bool:
        piece = piece_text.lower()
        if not piece.startswith(starter):
            return False
        if len(piece) == len(starter):
            return True
        return not piece[len(starter)].isalnum()

    def prompt_start_allowed_tokens(self, prompt_text: str | None) -> set[int] | None:
        starters = self.prompt_start_starters(prompt_text)
        if not starters:
            return None
        cache_key = "\0".join(starters)
        if cache_key in self._prompt_start_cache:
            return self._prompt_start_cache[cache_key]
        allowed: set[int] = set()
        for token_id, piece in enumerate(self.id_to_piece):
            if not self.token_can_begin_output(token_id):
                continue
            text = piece.decode("ascii", errors="ignore")
            if any(self._starter_piece_matches(text, starter) for starter in starters):
                allowed.add(token_id)
        self._prompt_start_cache[cache_key] = allowed or None
        return self._prompt_start_cache[cache_key]

    def token_can_start_prompt(self, prompt_text: str | None, token_id: int) -> bool:
        allowed = self.prompt_start_allowed_tokens(prompt_text)
        if allowed is None:
            return self.token_can_begin_output(token_id)
        return token_id in allowed

    def token_ends_sentence(self, token_id: int) -> bool:
        if not 0 <= token_id < self.vocab_size:
            return False
        piece = self.id_to_piece[token_id].rstrip()
        while piece and piece[-1] in b'"\')]}':
            piece = piece[:-1].rstrip()
        return bool(piece) and piece[-1] in b".!?"

    def generated_alpha_suffix_len(self, generated_tokens: list[int]) -> int:
        text = self.decode(generated_tokens)
        match = re.search(r"[A-Za-z]+$", text)
        return len(match.group(0)) if match else 0

    def token_can_follow_generated(
        self,
        generated_tokens: list[int],
        token_id: int,
        prompt_text: str | None = None,
    ) -> bool:
        if not self.token_allowed(token_id):
            return False
        if not generated_tokens:
            return self.token_can_start_prompt(prompt_text, token_id)
        return True

    def token_follow_penalty(
        self,
        generated_tokens: list[int],
        token_id: int,
        generated_text: str | None = None,
    ) -> float:
        penalty = 0.0
        if self.vocab_size <= BYTE_VOCAB_SIZE:
            return 0.0
        if BYTE_OFFSET <= token_id < BYTE_VOCAB_SIZE:
            byte_value = token_id - BYTE_OFFSET
            if (65 <= byte_value <= 90 or 97 <= byte_value <= 122) and self.generated_alpha_suffix_len(generated_tokens) >= 4:
                penalty += 1.5

        return penalty

    @classmethod
    def byte(cls) -> "GPT2BasicTokenizer":
        pieces = [b"<|endoftext|>", b"<|unk|>"]
        pieces.extend(bytes([value]) for value in range(256))
        return cls(pieces, [])

    @classmethod
    def read_vocab_bin(cls, path: Path) -> "GPT2BasicTokenizer":
        data = path.read_bytes()
        cursor = 0

        def take_i32() -> int:
            nonlocal cursor
            if cursor + 4 > len(data):
                raise ValueError(f"{path} ended while reading integer")
            value = struct.unpack_from("<i", data, cursor)[0]
            cursor += 4
            return value

        vocab_size = take_i32()
        if vocab_size < BYTE_VOCAB_SIZE:
            raise ValueError(f"{path} vocab_size={vocab_size}, expected at least {BYTE_VOCAB_SIZE}")
        if vocab_size > DOS_TOKEN_HASH_LIMIT:
            raise ValueError(f"{path} vocab_size={vocab_size}, DOS tokenizer limit is {DOS_TOKEN_HASH_LIMIT}")
        pieces: list[bytes] = [b""] * vocab_size
        seen_ids: set[int] = set()
        for _ in range(vocab_size):
            token_len = take_i32()
            if token_len <= 0 or token_len > MAX_TOKEN_LENGTH:
                raise ValueError(f"{path} invalid token length {token_len}")
            if cursor + MAX_TOKEN_LENGTH > len(data):
                raise ValueError(f"{path} ended while reading token bytes")
            raw_token = data[cursor : cursor + MAX_TOKEN_LENGTH]
            cursor += MAX_TOKEN_LENGTH
            token_id = take_i32()
            if token_id < 0 or token_id >= vocab_size:
                raise ValueError(f"{path} token id {token_id} outside vocab size {vocab_size}")
            if token_id in seen_ids:
                raise ValueError(f"{path} duplicate token id {token_id}")
            seen_ids.add(token_id)
            pieces[token_id] = raw_token[:token_len]

        merge_count = take_i32()
        if merge_count < 0:
            raise ValueError(f"{path} invalid merge count {merge_count}")
        if vocab_size == BYTE_VOCAB_SIZE and merge_count > 0:
            raise ValueError(f"{path} byte vocabulary cannot contain BPE merges")
        merges: list[MergeOp] = []
        for _ in range(merge_count):
            merge = MergeOp(take_i32(), take_i32(), take_i32(), take_i32())
            if not (0 <= merge.first < vocab_size and 0 <= merge.second < vocab_size and 0 <= merge.result < vocab_size):
                raise ValueError(f"{path} merge references token outside vocab: {merge}")
            if merge.first < BYTE_OFFSET or merge.second < BYTE_OFFSET:
                raise ValueError(f"{path} merge references special token: {merge}")
            if merge.result < BYTE_VOCAB_SIZE:
                raise ValueError(f"{path} merge result is not a BPE token: {merge}")
            if merge.priority < 0:
                raise ValueError(f"{path} merge has negative priority: {merge}")
            merges.append(merge)

        output_allowed: list[bool] | None = None
        mode: str | None = None
        while cursor < len(data):
            marker = take_i32()
            if marker == OUTPUT_MASK_MAGIC:
                mask_count = take_i32()
                if mask_count != vocab_size:
                    raise ValueError(f"{path} output mask count {mask_count} does not match vocab size {vocab_size}")
                if cursor + mask_count > len(data):
                    raise ValueError(f"{path} ended while reading output mask")
                output_allowed = [value != 0 for value in data[cursor : cursor + mask_count]]
                cursor += mask_count
            elif marker == TOKENIZER_MODE_MAGIC:
                mode_code = take_i32()
                if mode_code not in TOKENIZER_MODE_NAMES:
                    raise ValueError(f"{path} has unknown tokenizer mode code {mode_code}")
                mode = TOKENIZER_MODE_NAMES[mode_code]
            else:
                raise ValueError(f"{path} has unknown tokenizer extension marker {marker}")

        if cursor != len(data):
            raise ValueError(f"{path} has {len(data) - cursor} trailing bytes")
        return cls(pieces, merges, output_allowed, mode)

    def write_vocab_bin(self, path: Path) -> None:
        self.validate_contract()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            handle.write(struct.pack("<i", self.vocab_size))
            for token_id, piece in enumerate(self.id_to_piece):
                handle.write(struct.pack("<i", len(piece)))
                handle.write(piece.ljust(MAX_TOKEN_LENGTH, b"\0"))
                handle.write(struct.pack("<i", token_id))
            handle.write(struct.pack("<i", len(self.merges)))
            for merge in self.merges:
                handle.write(struct.pack("<iiii", merge.first, merge.second, merge.result, merge.priority))
            handle.write(struct.pack("<ii", OUTPUT_MASK_MAGIC, self.vocab_size))
            handle.write(bytes(1 if allowed else 0 for allowed in self.output_allowed))
            handle.write(struct.pack("<ii", TOKENIZER_MODE_MAGIC, TOKENIZER_MODE_CODES[self.mode]))

    def encode(self, text: str, append_eot: bool = False, output_safe: bool = False) -> list[int]:
        byte_text = clean_ascii(text).encode("ascii", errors="ignore")
        tokens = [value + BYTE_OFFSET for value in byte_text]
        if self.mode == "bpe" and self.merges:
            tokens = self._apply_dos_merges(tokens)
        elif self.mode == "lexicon":
            tokens = self._apply_lexicon(byte_text)
        if output_safe:
            safe_tokens: list[int] = []
            for token in tokens:
                if self.token_allowed(token):
                    safe_tokens.append(token)
                elif BYTE_VOCAB_SIZE <= token < self.vocab_size:
                    safe_tokens.extend(byte + BYTE_OFFSET for byte in self.id_to_piece[token])
                else:
                    safe_tokens.append(UNK_TOKEN)
            tokens = safe_tokens
        if append_eot:
            tokens.append(EOT_TOKEN)
        return tokens

    def _apply_lexicon(self, byte_text: bytes) -> list[int]:
        tokens: list[int] = []
        idx = 0
        while idx < len(byte_text):
            matched_piece: bytes | None = None
            matched_token = -1
            for piece_len in self.lexicon_lengths_by_first.get(byte_text[idx], ()):
                if idx + piece_len > len(byte_text):
                    continue
                piece = byte_text[idx : idx + piece_len]
                token_id = self.lexicon_by_piece.get(piece, -1)
                if token_id >= 0 and self._lexicon_piece_boundary_ok(byte_text, idx, piece):
                    matched_piece = piece
                    matched_token = token_id
                    break
            if matched_piece is not None:
                tokens.append(matched_token)
                idx += len(matched_piece)
            else:
                tokens.append(byte_text[idx] + BYTE_OFFSET)
                idx += 1
        return tokens

    @staticmethod
    def _lexicon_word_byte(value: int) -> bool:
        return (
            48 <= value <= 57
            or 65 <= value <= 90
            or 97 <= value <= 122
            or value in (ord("."), ord("+"), ord("/"), ord("_"), ord("-"))
        )

    @classmethod
    def _lexicon_piece_boundary_ok(cls, byte_text: bytes, start_idx: int, piece: bytes) -> bool:
        end_idx = start_idx + len(piece)
        if end_idx >= len(byte_text):
            return True
        if not cls._lexicon_word_byte(piece[-1]):
            return True
        next_byte = byte_text[end_idx]
        if next_byte == ord("."):
            next_idx = end_idx + 1
            return next_idx >= len(byte_text) or not cls._lexicon_word_byte(byte_text[next_idx])
        return not cls._lexicon_word_byte(next_byte)

    def _apply_dos_merges(self, tokens: list[int]) -> list[int]:
        working = list(tokens)
        while len(working) >= 2:
            best_merge: MergeOp | None = None
            for idx in range(len(working) - 1):
                merge = self.merge_by_pair.get((working[idx], working[idx + 1]))
                if merge is None:
                    continue
                if best_merge is None or merge.priority < best_merge.priority:
                    best_merge = merge

            if best_merge is None:
                break

            merged_tokens: list[int] = []
            idx = 0
            while idx < len(working):
                if (
                    idx + 1 < len(working)
                    and working[idx] == best_merge.first
                    and working[idx + 1] == best_merge.second
                ):
                    merged_tokens.append(best_merge.result)
                    idx += 2
                else:
                    merged_tokens.append(working[idx])
                    idx += 1
            working = merged_tokens
        return working

    def decode(self, tokens: list[int]) -> str:
        chunks: list[bytes] = []
        for token in tokens:
            if token == EOT_TOKEN:
                break
            if token == UNK_TOKEN:
                chunks.append(b"<|unk|>")
            elif 0 <= token < self.vocab_size:
                chunks.append(self.id_to_piece[token])
        return b"".join(chunks).decode("ascii", errors="ignore")

    def validate_contract(self) -> None:
        if self.vocab_size > DOS_TOKEN_HASH_LIMIT:
            raise ValueError(f"tokenizer vocab_size={self.vocab_size}, DOS tokenizer limit is {DOS_TOKEN_HASH_LIMIT}")
        if self.mode not in TOKENIZER_MODE_CODES:
            raise ValueError(f"unknown tokenizer mode: {self.mode}")
        if self.vocab_size == BYTE_VOCAB_SIZE and self.merges:
            raise ValueError("byte vocabulary cannot contain BPE merges")
        if self.mode == "byte" and (self.vocab_size != BYTE_VOCAB_SIZE or self.merges):
            raise ValueError("byte tokenizer mode cannot contain extra vocabulary or merges")
        if self.mode == "lexicon" and self.merges:
            raise ValueError("lexicon tokenizer mode cannot contain BPE merges")
        if self.mode == "bpe" and self.vocab_size > BYTE_VOCAB_SIZE and not self.merges:
            raise ValueError("BPE tokenizer mode with extra vocabulary requires merges")
        if len(self.output_allowed) != self.vocab_size:
            raise ValueError("output mask length does not match vocabulary size")
        if not self.output_allowed[EOT_TOKEN]:
            raise ValueError("EOT token must be allowed for output")
        if self.output_allowed[UNK_TOKEN]:
            raise ValueError("UNK token must be masked from output")
        for token_id, piece in enumerate(self.id_to_piece):
            if not piece:
                raise ValueError(f"token {token_id} is empty")
            if len(piece) > MAX_TOKEN_LENGTH:
                raise ValueError(f"token {token_id} is {len(piece)} bytes, max is {MAX_TOKEN_LENGTH}")
            if BYTE_OFFSET <= token_id < BYTE_VOCAB_SIZE:
                byte = token_id - BYTE_OFFSET
                expected = 32 <= byte <= 126
                if self.output_allowed[token_id] != expected:
                    raise ValueError(f"byte token {token_id} output mask is inconsistent with printability")
        for merge in self.merges:
            if not (
                BYTE_OFFSET <= merge.first < self.vocab_size
                and BYTE_OFFSET <= merge.second < self.vocab_size
                and BYTE_VOCAB_SIZE <= merge.result < self.vocab_size
            ):
                raise ValueError(f"merge references invalid token ids: {merge}")
            if merge.priority < 0:
                raise ValueError(f"merge has negative priority: {merge}")

    def validate_for_vocab_size(self, vocab_size: int) -> None:
        self.validate_contract()
        if self.vocab_size != vocab_size:
            raise ValueError(f"tokenizer vocab_size={self.vocab_size}, model vocab_size={vocab_size}")
        for token_id in range(256):
            expected = bytes([token_id])
            actual = self.id_to_piece[token_id + BYTE_OFFSET]
            if actual != expected:
                raise ValueError(f"byte token {token_id + BYTE_OFFSET} is {actual!r}, expected {expected!r}")


def build_bpe_tokenizer(
    documents: list[str],
    vocab_size: int,
    min_pair_count: int = 2,
    max_token_length: int = MAX_TOKEN_LENGTH,
    output_mode: str = "all",
) -> GPT2BasicTokenizer:
    if vocab_size < BYTE_VOCAB_SIZE:
        raise ValueError(f"vocab_size must be at least {BYTE_VOCAB_SIZE}")
    if vocab_size > DOS_TOKEN_HASH_LIMIT:
        raise ValueError(f"vocab_size must be at most {DOS_TOKEN_HASH_LIMIT} for the DOS tokenizer")
    if output_mode not in {"all", "complete"}:
        raise ValueError(f"unknown BPE output mode: {output_mode}")
    pieces = GPT2BasicTokenizer.byte().id_to_piece
    sequences = [[value + BYTE_OFFSET for value in clean_ascii(doc).encode("ascii", errors="ignore")] for doc in documents]
    sequences = [seq for seq in sequences if seq]
    merges: list[MergeOp] = []
    blocked: set[tuple[int, int]] = set()

    while len(pieces) < vocab_size:
        counts: Counter[tuple[int, int]] = Counter()
        for seq in sequences:
            counts.update(pair for pair in zip(seq, seq[1:]) if pair not in blocked)
        if not counts:
            break

        best_pair: tuple[int, int] | None = None
        best_count = 0
        for pair, count in counts.items():
            if count < min_pair_count:
                continue
            piece = pieces[pair[0]] + pieces[pair[1]]
            if len(piece) > max_token_length:
                blocked.add(pair)
                continue
            if count > best_count or (count == best_count and (best_pair is None or pair < best_pair)):
                best_pair = pair
                best_count = count

        if best_pair is None:
            break

        new_id = len(pieces)
        pieces.append(pieces[best_pair[0]] + pieces[best_pair[1]])
        merges.append(MergeOp(best_pair[0], best_pair[1], new_id, len(merges)))

        for seq_idx, seq in enumerate(sequences):
            merged_seq: list[int] = []
            idx = 0
            while idx < len(seq):
                if idx + 1 < len(seq) and (seq[idx], seq[idx + 1]) == best_pair:
                    merged_seq.append(new_id)
                    idx += 2
                else:
                    merged_seq.append(seq[idx])
                    idx += 1
            sequences[seq_idx] = merged_seq

    output_allowed = complete_output_allowed(pieces, documents) if output_mode == "complete" else None
    return GPT2BasicTokenizer(pieces, merges, output_allowed)


def build_lexicon_tokenizer(
    documents: list[str],
    vocab_size: int,
    min_count: int = 2,
    max_piece_chars: int = MAX_TOKEN_LENGTH,
    max_phrase_words: int = 3,
) -> GPT2BasicTokenizer:
    if vocab_size < BYTE_VOCAB_SIZE:
        raise ValueError(f"vocab_size must be at least {BYTE_VOCAB_SIZE}")
    if vocab_size > DOS_TOKEN_HASH_LIMIT:
        raise ValueError(f"vocab_size must be at most {DOS_TOKEN_HASH_LIMIT} for the DOS tokenizer")
    if max_phrase_words < 1:
        raise ValueError("max_phrase_words must be positive")
    pieces = list(GPT2BasicTokenizer.byte().id_to_piece)
    seen = set(pieces)
    for candidate in lexicon_candidates(
        documents,
        max_piece_chars=max_piece_chars,
        max_phrase_words=max_phrase_words,
        min_count=min_count,
    ):
        if candidate in seen:
            continue
        pieces.append(candidate)
        seen.add(candidate)
        if len(pieces) >= vocab_size:
            break
    output_allowed = default_output_allowed(pieces)
    for token_id in range(BYTE_VOCAB_SIZE, len(pieces)):
        output_allowed[token_id] = True
    return GPT2BasicTokenizer(pieces, [], output_allowed, mode="lexicon")


def load_tokenizer_for_model(model_dir: Path, vocab_size: int | None = None) -> GPT2BasicTokenizer:
    vocab_path = model_dir / "VOCAB.BIN"
    tokenizer = GPT2BasicTokenizer.read_vocab_bin(vocab_path) if vocab_path.exists() else GPT2BasicTokenizer.byte()
    if vocab_size is not None:
        tokenizer.validate_for_vocab_size(vocab_size)
    return tokenizer


def self_test() -> None:
    docs = [
        "the transformer runtime uses the tokenizer",
        "the tokenizer reduces repeated transformer words",
        "fixed point transformer inference in DOS",
    ]
    tokenizer = build_bpe_tokenizer(docs, 280, min_pair_count=2)
    complete_tokenizer = build_bpe_tokenizer(docs, 280, min_pair_count=2, output_mode="complete")
    lexicon_tokenizer = build_lexicon_tokenizer(docs, 280, min_count=1, max_phrase_words=2)
    starter_pieces = GPT2BasicTokenizer.byte().id_to_piece + [
        b" compact",
        b" stays small",
        b" uses",
        b" correctness",
        b" reduce",
        b" arrays",
    ]
    starter_allowed = default_output_allowed(starter_pieces)
    for token_id in range(BYTE_VOCAB_SIZE, len(starter_pieces)):
        starter_allowed[token_id] = True
    starter_tokenizer = GPT2BasicTokenizer(starter_pieces, [], starter_allowed, mode="lexicon")
    boundary_tokenizer = build_lexicon_tokenizer(
        ["arithmetic. A no FPU machine", "arithmetic. Attention scores matter"],
        290,
        min_count=1,
        max_phrase_words=2,
    )
    sentence_tokenizer = build_lexicon_tokenizer(
        [
            "User: how are you Assistant: I answer in DOS.",
            "User: is this real Assistant: Yes, it is real.",
            "Q: i am bored A: Try one small project.",
        ],
        340,
        min_count=1,
        max_phrase_words=4,
    )
    encoded = tokenizer.encode("the transformer tokenizer", append_eot=True)
    safe_encoded = complete_tokenizer.encode("the transformer tokenizer", append_eot=True, output_safe=True)
    lexicon_encoded = lexicon_tokenizer.encode("the transformer runtime uses the tokenizer", append_eot=True)
    boundary_encoded = boundary_tokenizer.encode("arithmetic. Attention scores matter")
    decoded = tokenizer.decode(encoded)
    if not complete_tokenizer.token_allowed(EOT_TOKEN):
        raise AssertionError("complete output mode masked EOT")
    if complete_tokenizer.token_allowed(UNK_TOKEN):
        raise AssertionError("complete output mode allowed UNK")
    if any(not complete_tokenizer.token_allowed(token) for token in safe_encoded):
        raise AssertionError("output-safe encoding produced a masked token")
    if lexicon_tokenizer.mode != "lexicon":
        raise AssertionError("lexicon tokenizer mode was not retained")
    if not any(token >= BYTE_VOCAB_SIZE for token in lexicon_encoded):
        raise AssertionError("lexicon tokenizer did not use any lexicon pieces")
    if lexicon_tokenizer.decode(lexicon_encoded) != "the transformer runtime uses the tokenizer":
        raise AssertionError("lexicon tokenizer decode roundtrip failed")
    if any(boundary_tokenizer.id_to_piece[token] == b" arithmetic. A" for token in boundary_encoded):
        raise AssertionError("lexicon boundary check allowed a prefix piece inside Attention")
    dotted_sentence = boundary_tokenizer.encode("arithmetic. A no FPU machine")
    if not any(boundary_tokenizer.id_to_piece[token] == b" machine" for token in dotted_sentence):
        raise AssertionError("lexicon boundary check rejected word before sentence punctuation")
    sentence_response_encoded = sentence_tokenizer.encode("Yes, it is real.")
    sentence_response_pieces = [sentence_tokenizer.id_to_piece[token] for token in sentence_response_encoded]
    if not any(piece in {b" it is real.", b" is real.", b" real."} for piece in sentence_response_pieces):
        raise AssertionError(f"sentence-piece tokenizer missed answer tail: {sentence_response_pieces!r}")
    if not sentence_tokenizer.token_ends_sentence(sentence_response_encoded[-1]):
        raise AssertionError("sentence-piece tokenizer did not preserve ending punctuation")
    sentence_encoded = lexicon_tokenizer.encode("the transformer runtime works.", output_safe=True)
    if not lexicon_tokenizer.token_ends_sentence(sentence_encoded[-1]):
        raise AssertionError("lexicon sentence-ending token was not recognized")
    if not lexicon_tokenizer.token_can_begin_output(ord(" ") + BYTE_OFFSET):
        raise AssertionError("space byte should be a valid output start")
    if lexicon_tokenizer.token_can_begin_output(ord(",") + BYTE_OFFSET):
        raise AssertionError("comma byte should not be a valid output start")
    if not lexicon_tokenizer.token_can_follow_generated([], BYTE_VOCAB_SIZE):
        raise AssertionError("lexicon token should be a valid output start")
    if lexicon_tokenizer.token_follow_penalty([ord("t") + BYTE_OFFSET, ord("e") + BYTE_OFFSET, ord("s") + BYTE_OFFSET, ord("t") + BYTE_OFFSET], ord("s") + BYTE_OFFSET) <= 0.0:
        raise AssertionError("long byte-level alphabetic run should be penalized")
    if not starter_tokenizer.prompt_start_starters("DOS language models need"):
        raise AssertionError("prompt starter rule did not match")
    if not starter_tokenizer.token_can_start_prompt("DOS language models need", BYTE_VOCAB_SIZE):
        raise AssertionError("prompt starter rule rejected compact")
    if starter_tokenizer.token_can_start_prompt("DOS language models need", BYTE_VOCAB_SIZE + 1):
        raise AssertionError("prompt starter rule allowed bad DOS starter")
    if not starter_tokenizer.token_can_start_prompt("A BASIC transformer runtime", BYTE_VOCAB_SIZE + 2):
        raise AssertionError("prompt starter rule rejected uses")
    if starter_tokenizer.token_can_start_prompt("A BASIC transformer runtime", BYTE_VOCAB_SIZE + 3):
        raise AssertionError("prompt starter rule allowed bad runtime starter")
    if not starter_tokenizer.token_can_follow_generated([], BYTE_VOCAB_SIZE + 4, "To improve performance on real hardware"):
        raise AssertionError("prompt starter rule rejected reduce")
    if starter_tokenizer.token_can_follow_generated([], BYTE_VOCAB_SIZE + 5, "To improve performance on real hardware"):
        raise AssertionError("prompt starter rule allowed bad performance starter")
    byte_vocab = GPT2BasicTokenizer.byte().id_to_piece
    a_id = ord("a") + BYTE_OFFSET
    b_id = ord("b") + BYTE_OFFSET
    c_id = ord("c") + BYTE_OFFSET
    ranked = GPT2BasicTokenizer(
        byte_vocab + [b"bc", b"ab"],
        [
            MergeOp(b_id, c_id, BYTE_VOCAB_SIZE, 0),
            MergeOp(a_id, b_id, BYTE_VOCAB_SIZE + 1, 1),
        ],
    )
    ranked_encoded = ranked.encode("abc")
    if ranked_encoded != [a_id, BYTE_VOCAB_SIZE]:
        raise AssertionError(f"ranked BPE encoded {ranked_encoded}, expected {[a_id, BYTE_VOCAB_SIZE]}")
    if ranked.decode(ranked_encoded) != "abc":
        raise AssertionError("ranked BPE decode roundtrip failed")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "VOCAB.BIN"
        complete_tokenizer.write_vocab_bin(path)
        loaded = GPT2BasicTokenizer.read_vocab_bin(path)
        loaded_via_model_dir = load_tokenizer_for_model(Path(tmp), loaded.vocab_size)
        lexicon_path = Path(tmp) / "LEXICON.BIN"
        lexicon_tokenizer.write_vocab_bin(lexicon_path)
        loaded_lexicon = GPT2BasicTokenizer.read_vocab_bin(lexicon_path)
    if complete_tokenizer.output_allowed != loaded.output_allowed:
        raise AssertionError("VOCAB.BIN roundtrip changed output mask")
    if loaded_lexicon.mode != "lexicon":
        raise AssertionError("VOCAB.BIN roundtrip changed lexicon mode")
    if lexicon_encoded != loaded_lexicon.encode("the transformer runtime uses the tokenizer", append_eot=True):
        raise AssertionError("VOCAB.BIN roundtrip changed lexicon encoding")
    if encoded != loaded.encode("the transformer tokenizer", append_eot=True):
        raise AssertionError("VOCAB.BIN roundtrip changed encoding")
    if encoded != loaded_via_model_dir.encode("the transformer tokenizer", append_eot=True):
        raise AssertionError("model-dir tokenizer load changed encoding")
    if decoded != "the transformer tokenizer":
        raise AssertionError(f"decode roundtrip failed: {decoded!r}")
    loaded.validate_for_vocab_size(loaded.vocab_size)
    invalid = GPT2BasicTokenizer.byte()
    invalid.merges = [MergeOp(BYTE_OFFSET, BYTE_OFFSET + 1, BYTE_OFFSET, 0)]
    try:
        invalid.validate_for_vocab_size(BYTE_VOCAB_SIZE)
        raise AssertionError("invalid byte-size BPE merge was accepted")
    except ValueError:
        pass
    print("trace_scope tokenizer_contract")
    print("trace clean_ascii")
    print("trace GPT2BasicTokenizer.vocab_size")
    print("trace GPT2BasicTokenizer.uses_bpe")
    print("trace default_output_allowed")
    print("trace GPT2BasicTokenizer.write_vocab_bin")
    print("trace GPT2BasicTokenizer.read_vocab_bin")
    print("trace GPT2BasicTokenizer.token_allowed")
    print("trace GPT2BasicTokenizer.token_can_begin_output")
    print("trace GPT2BasicTokenizer.prompt_start_starters")
    print("trace GPT2BasicTokenizer._starter_piece_matches")
    print("trace GPT2BasicTokenizer.prompt_start_allowed_tokens")
    print("trace GPT2BasicTokenizer.token_can_start_prompt")
    print("trace GPT2BasicTokenizer.token_ends_sentence")
    print("trace GPT2BasicTokenizer.generated_alpha_suffix_len")
    print("trace GPT2BasicTokenizer.token_can_follow_generated")
    print("trace GPT2BasicTokenizer.token_follow_penalty")
    print("trace GPT2BasicTokenizer.encode_output_safe")
    print("trace GPT2BasicTokenizer.lexicon_lengths_by_first")
    print("trace GPT2BasicTokenizer._apply_lexicon")
    print("trace GPT2BasicTokenizer._lexicon_piece_boundary_ok")
    print("trace GPT2BasicTokenizer._lexicon_word_byte")
    print("trace GPT2BasicTokenizer._apply_dos_merges")
    print("trace complete_piece_inventory")
    print("trace complete_output_allowed")
    print("trace plain_spaced_word_phrase")
    print("trace response_sentence_spans")
    print("trace response_sentence_pieces")
    print("trace lexicon_candidates")
    print("trace build_lexicon_tokenizer")
    print("trace ranked_bpe_merge_order")
    print("trace GPT2BasicTokenizer.validate_contract")
    print("trace GPT2BasicTokenizer.validate_for_vocab_size")
    print("trace load_tokenizer_for_model")
    print("artifact: VOCAB.BIN")
    print(f"PROBE_OK build_bpe_tokenizer vocab={tokenizer.vocab_size} merges={len(tokenizer.merges)}")
    print(f"PROBE_OK complete_output_allowed allowed={sum(complete_tokenizer.output_allowed)}")
    print(f"PROBE_OK build_lexicon_tokenizer vocab={lexicon_tokenizer.vocab_size}")
    print(f"PROBE_OK encode tokens={len(encoded)}")
    print(f"PROBE_OK encode_output_safe tokens={len(safe_encoded)}")
    print(f"PROBE_OK encode_lexicon tokens={len(lexicon_encoded)}")
    print(f"PROBE_OK encode_lexicon_boundary tokens={len(boundary_encoded)}")
    print(f"PROBE_OK ranked_bpe tokens={ranked_encoded}")
    print(f"PROBE_OK decode text={decoded}")
    print("PROBE_OK vocab_bin roundtrip=1")
    print("PROBE_OK self_test exercised=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokenizer", choices=("bpe", "lexicon"), default="bpe")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--min-pair-count", type=int, default=2)
    parser.add_argument("--output-mode", choices=("all", "complete"), default="all")
    parser.add_argument("--lexicon-min-count", type=int, default=2)
    parser.add_argument("--lexicon-max-phrase-words", type=int, default=3)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.input or args.output is None:
        parser.error("--input and --output are required unless --self-test is used")
    documents: list[str] = []
    for path in args.input:
        documents.extend(part for part in path.read_text(encoding="utf-8", errors="ignore").splitlines() if part.strip())
    if args.tokenizer == "lexicon":
        tokenizer = build_lexicon_tokenizer(
            documents,
            args.vocab_size,
            min_count=args.lexicon_min_count,
            max_phrase_words=args.lexicon_max_phrase_words,
        )
    else:
        tokenizer = build_bpe_tokenizer(documents, args.vocab_size, args.min_pair_count, output_mode=args.output_mode)
    tokenizer.write_vocab_bin(args.output)
    print(
        f"wrote {args.output} tokenizer={tokenizer.mode} vocab={tokenizer.vocab_size} merges={len(tokenizer.merges)} "
        f"output_allowed={sum(tokenizer.output_allowed)}"
    )


if __name__ == "__main__":
    main()
